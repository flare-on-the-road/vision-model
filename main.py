from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import base64
import logging
import os
import re

import numpy as np
import onnxruntime as ort
import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw
import io

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _ensure_logging():
    """uvicorn default log_config가 root logger를 WARNING으로 덮어쓰므로 lifespan에서 복구."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
VISION_DEVICE_POLICY = os.getenv("VISION_DEVICE_POLICY", "").strip().lower()
if not VISION_DEVICE_POLICY:
    VISION_DEVICE_POLICY = "auto"
REQUIRE_CUDA = VISION_DEVICE_POLICY in {"cuda", "require-cuda", "gpu", "require-gpu"}
FORCE_CPU = VISION_DEVICE_POLICY in {"cpu", "force-cpu"}
PRELOAD_CUDA_DLLS = os.getenv("VISION_PRELOAD_CUDA_DLLS", "true").lower() == "true"
VISION_DEBUG = os.getenv("VISION_DEBUG", "false").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _int_env("PORT", 8000)
INPUT_IMAGE_SIZE = 640
NMS_IOU_THRESHOLD = float(os.getenv("VISION_YOLO_NMS_IOU_THRESHOLD", "0.45"))

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
    2: "carlight",
}

RISK_CLASSES = {"fire", "smoke"}
FALSE_POSITIVE_CLASSES = {"carlight"}

VLM_CONF_LOW = 0.6
VLM_CONF_HIGH = 0.8

MODEL_CONFIGS = {
    "rt-detr": {
        "display_name": "RT-DETRv2",
        "path": os.getenv("RT_DETR_MODEL_PATH", MODEL_PATH),
        "type": "rtdetr",
    },
    "yolov8": {
        "display_name": "YOLOv8",
        "path": os.getenv("YOLOV8_MODEL_PATH", "models/YOLOv8l_fp32.onnx"),
        "type": "yolo",
    },
    "yolov11": {
        "display_name": "YOLOv11",
        "path": os.getenv("YOLOV11_MODEL_PATH", "models/YOLOv11lbest_fp32.onnx"),
        "type": "yolo",
    },
}


class OnnxPredictor:
    def __init__(self, model_key: str, display_name: str, model_path: str, model_type: str):
        self.model_key = model_key
        self.display_name = display_name
        self.model_path = model_path
        self.model_type = model_type
        self.session = None
        self.device = None
        self.input_names = []
        self.output_names = []

    def load(self):
        if PRELOAD_CUDA_DLLS and hasattr(ort, "preload_dlls"):
            # Allows onnxruntime-gpu[cuda,cudnn] NVIDIA wheels to be discovered.
            try:
                ort.preload_dlls(directory="")
            except Exception as exc:
                logger.warning(f"[ORT] CUDA DLL preload skipped: {exc}")

        available_providers = ort.get_available_providers()

        if not FORCE_CPU and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif REQUIRE_CUDA:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"Available providers: {available_providers}. "
                "Install a CUDA-compatible onnxruntime-gpu environment or set "
                "VISION_DEVICE_POLICY=auto/cpu to allow CPU fallback."
            )
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"
            logger.warning(
                "[ORT] CUDAExecutionProvider is not available for %s. "
                "Falling back to CPU. Available providers: %s",
                self.display_name,
                available_providers,
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.display_name} model file not found: {self.model_path}")

        print(f"[INFO] Loading ONNX model ({self.model_key}): {self.model_path}")
        print(f"[INFO] Providers: {providers}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers,
        )

        active_providers = self.session.get_providers()
        if REQUIRE_CUDA and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider was requested but is not active after "
                "creating the ONNX Runtime session. "
                f"Requested providers: {providers}. "
                f"Active providers: {active_providers}. "
                "Check CUDA/cuDNN runtime libraries and LD_LIBRARY_PATH. "
                "For GTX 1060 with onnxruntime-gpu==1.18.0, CUDA 11.8/cuDNN 8 "
                "runtime libraries such as libcublasLt.so.11 must be available."
            )

        self.device = (
            "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[INFO] Input names: {self.input_names}")
        print(f"[INFO] Output names: {self.output_names}")
        print(f"[INFO] Active ONNX Runtime providers: {active_providers}")
        self._validate_model_io()
        print(f"[INFO] {self.display_name} loaded on {self.device}")

    def _validate_model_io(self):
        if self.model_type == "yolo":
            if not self.input_names or not self.output_names:
                raise ValueError(
                    f"{self.display_name} has invalid ONNX IO. "
                    f"Inputs: {self.input_names}, outputs: {self.output_names}"
                )
            return

        required_inputs = {"images", "orig_target_sizes"}
        required_outputs = {"labels", "boxes", "scores"}

        missing_inputs = required_inputs - set(self.input_names)
        missing_outputs = required_outputs - set(self.output_names)

        if missing_inputs:
            raise ValueError(
                f"Missing required ONNX inputs: {missing_inputs}. "
                f"Actual inputs: {self.input_names}"
            )

        if missing_outputs:
            raise ValueError(
                f"Missing required ONNX outputs: {missing_outputs}. "
                f"Actual outputs: {self.output_names}"
            )

    def predict(
        self,
        image_bytes: bytes,
        confidence: float = 0.25,
        max_detections: int = 100,
    ) -> Dict[str, Any]:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_width, original_height = pil_image.size

        if self.model_type == "yolo":
            detections = self._predict_yolo(
                pil_image=pil_image,
                confidence=confidence,
                max_detections=max_detections,
            )
        else:
            input_image = _preprocess_image(pil_image)
            orig_target_sizes = np.array([[original_width, original_height]], dtype=np.int64)

            input_feed = {
                name: (input_image if name == "images" else orig_target_sizes)
                for name in self.input_names
            }

            outputs = self.session.run(self.output_names, input_feed)
            output_map = dict(zip(self.output_names, outputs))

            detections = _postprocess_rtdetr(
                labels=output_map["labels"],
                boxes=output_map["boxes"],
                scores=output_map["scores"],
                confidence=confidence,
                max_detections=max_detections,
                original_width=original_width,
                original_height=original_height,
            )

        risk_detections = [d for d in detections if d["class_name"] in RISK_CLASSES]
        false_positive_hints = [d for d in detections if d["class_name"] in FALSE_POSITIVE_CLASSES]

        max_confidence = max((d["confidence"] for d in detections), default=0.0)
        risk_max_confidence = max((d["confidence"] for d in risk_detections), default=0.0)
        return {
            "success": True,
            "model": self.model_key,
            "model_name": self.display_name,
            "model_path": self.model_path,
            "device": self.device,
            "image": {
                "width": original_width,
                "height": original_height,
            },
            "thresholds": {
                "confidence": confidence,
                "max_detections": max_detections,
            },
            "summary": {
                "total_detections": len(detections),
                "risk_detection_count": len(risk_detections),
                "false_positive_hint_count": len(false_positive_hints),
                "max_confidence": max_confidence,
                "risk_max_confidence": risk_max_confidence,
            },
            "detections": detections,
            "risk_detections": risk_detections,
            "false_positive_hints": false_positive_hints,
        }

    def _predict_yolo(
        self,
        pil_image: Image.Image,
        confidence: float,
        max_detections: int,
    ) -> List[Dict[str, Any]]:
        original_width, original_height = pil_image.size
        input_image = _preprocess_image(pil_image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_image})
        return _postprocess_yolo(
            output=outputs[0],
            confidence=confidence,
            max_detections=max_detections,
            original_width=original_width,
            original_height=original_height,
        )

class PredictorRegistry:
    def __init__(self):
        self.predictors: Dict[str, OnnxPredictor] = {}

    def load(self):
        for model_key, config in MODEL_CONFIGS.items():
            predictor = OnnxPredictor(
                model_key=model_key,
                display_name=config["display_name"],
                model_path=config["path"],
                model_type=config["type"],
            )
            predictor.load()
            self.predictors[model_key] = predictor

    def get(self, model_key: str) -> OnnxPredictor:
        if model_key not in self.predictors:
            raise KeyError(model_key)
        return self.predictors[model_key]

    def health(self):
        return {
            key: {
                "display_name": predictor.display_name,
                "device": predictor.device,
                "model_path": predictor.model_path,
                "type": predictor.model_type,
            }
            for key, predictor in self.predictors.items()
        }


predictors = PredictorRegistry()

# ── VLM 설정 ─────────────────────────────────────────────────────────────────

_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "")
_OPENROUTER_TIMEOUT = _int_env("OPENROUTER_TIMEOUT", 30)
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

def _build_vlm_prompt(detections: list) -> str:
    lines = [
        "이 이미지는 터널 CCTV 화재 감지 AI의 탐지 결과입니다.",
        "이미지에는 바운딩 박스(BBox)와 레이블이 표시되어 있습니다.",
        "",
        "탐지된 항목:",
    ]
    for i, det in enumerate(detections, 1):
        lines.append(f"{i}. {det['class_name']} (신뢰도 {det['confidence']:.2f})")
    lines += [
        "",
        "각 항목이 오탐(잘못된 탐지)인지 판단하세요.",
        "",
        "오탐 판단 기준:",
        "- 실제 해당 객체가 없는데 감지된 경우",
        "- 조명·반사·햇빛 등 유사 시각 패턴을 화염(fire)으로 잘못 분류한 경우",
        "- 차량 전조등(carlight)을 화염으로 잘못 분류한 경우",
        "- 터널 환경 특성상 구름 없음 (매연·분말·증기·먼지를 smoke로 잘못 분류 포함)",
        "",
        "반드시 아래 형식으로만 답변하세요 (번호 순서대로):",
    ]
    for i in range(1, len(detections) + 1):
        lines.append(f"{i}. 오탐 여부: yes(오탐) 또는 no(정상) / 판단 근거")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_logging()
    predictors.load()
    yield


app = FastAPI(title="FLARE Vision API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": predictors.health(),
        "providers": ort.get_available_providers(),
    }


_BBOX_COLORS = {
    "fire": (255, 60, 60),
    "smoke": (180, 180, 180),
    "carlight": (255, 210, 0),
}


def _draw_bboxes(image_bytes: bytes, detections: list) -> bytes:
    """탐지 결과(bbox + 레이블)를 이미지에 그려 JPEG bytes로 반환."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        color = _BBOX_COLORS.get(class_name, (255, 255, 255))

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{class_name} {confidence:.2f}"
        draw.text((x1, max(0, y1 - 16)), label, fill=color)

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _call_vlm(image_bytes: bytes, detections: list) -> Optional[list]:
    """
    OpenRouter VLM 호출. 탐지 항목별 오탐 여부를 반환한다.
    성공 시 [{"class_name": str, "is_false_positive": bool, "reason": str}, ...] 반환.
    실패 시 None.
    """
    if not _OPENROUTER_API_KEY or not _OPENROUTER_MODEL:
        logger.warning("[VLM] OPENROUTER_API_KEY 또는 OPENROUTER_MODEL 미설정 → 생략")
        return None

    prompt = _build_vlm_prompt(detections)

    if VISION_DEBUG:
        logger.info(f"[VLM] 호출 시작 (model={_OPENROUTER_MODEL}, image={len(image_bytes)}B)")
        logger.info(f"[VLM] 프롬프트:\n{prompt}")
    else:
        logger.info(f"[VLM] 호출 시작 (model={_OPENROUTER_MODEL}, items={len(detections)})")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": _OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 128 * len(detections),
        "temperature": 0,
    }
    try:
        resp = http_requests.post(
            _OPENROUTER_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {_OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            timeout=_OPENROUTER_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if VISION_DEBUG:
            logger.info(f"[VLM] 응답 원문 전체:\n{raw}")
        else:
            logger.info(f"[VLM] 응답 원문: {raw[:300]}")
    except Exception as e:
        logger.error(f"[VLM] 호출 실패: {e}")
        if VISION_DEBUG:
            logger.exception("[VLM] 스택 트레이스:")
        return None

    result = _parse_vlm_response(raw, detections)
    if result is None:
        logger.warning("[VLM] 파싱 실패 — 응답에서 번호별 yes/no 패턴을 찾지 못했습니다.")
    else:
        logger.info(f"[VLM] 파싱 결과: {result}")
    return result


def _parse_vlm_response(raw: str, detections: list) -> Optional[list]:
    results = []
    for i, det in enumerate(detections, 1):
        # "1. 오탐 여부: yes / 근거" 또는 "1. yes / 근거" 형태 모두 허용
        pattern = rf"{i}\.\s*(?:오탐\s*여부\s*[:：]\s*)?(yes|no)\s*[/／、,]\s*(.+?)(?=\n\s*\d+\.|$)"
        match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if not match:
            logger.warning(f"[VLM] {i}번 항목({det['class_name']}) 파싱 실패")
            return None
        is_false_positive = match.group(1).lower() == "yes"
        reason = match.group(2).strip()
        results.append({
            "class_name": det["class_name"],
            "is_false_positive": is_false_positive,
            "reason": reason,
        })
    return results if results else None


@app.post("/vlm")
async def vlm(
    image: UploadFile = File(...),
    detections: str = Form(default="[]"),
):
    """이미지와 탐지 목록을 받아 항목별 오탐 여부를 반환한다."""
    if not _OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY 미설정")
    if not _OPENROUTER_MODEL:
        raise HTTPException(status_code=503, detail="OPENROUTER_MODEL 미설정")

    import json as _json
    try:
        det_list = _json.loads(detections)
    except Exception:
        raise HTTPException(status_code=400, detail="detections 필드가 유효한 JSON이 아닙니다.")

    if not isinstance(det_list, list) or not det_list:
        raise HTTPException(status_code=400, detail="detections는 비어있지 않은 배열이어야 합니다.")

    image_bytes = await image.read()
    result = _call_vlm(image_bytes, det_list)
    if result is None:
        raise HTTPException(status_code=502, detail="VLM 호출 또는 파싱 실패")
    return result


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model_key: str = Form(default="rt-detr"),
    confidence: float = Form(default=0.25, ge=0.0, le=1.0),
    max_detections: int = Form(default=100, ge=1, le=300),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image 파일만 허용됩니다.")

    image_bytes = await image.read()

    try:
        predictor = predictors.get(model_key)
        result = predictor.predict(
            image_bytes=image_bytes,
            confidence=confidence,
            max_detections=max_detections,
        )
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 model_key입니다: {model_key}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 신뢰도 구간(0.6~0.8) 탐지 시 bbox 이미지 생성 → VLM 2차 판단 (항목별 오탐 여부)
    vlm_candidates = [
        d for d in result["detections"]
        if VLM_CONF_LOW <= d["confidence"] <= VLM_CONF_HIGH
    ]
    if vlm_candidates:
        annotated_bytes = _draw_bboxes(image_bytes, vlm_candidates)
        result["annotated_image_b64"] = base64.b64encode(annotated_bytes).decode("utf-8")
        result["vlm"] = _call_vlm(annotated_bytes, vlm_candidates)
    else:
        result["annotated_image_b64"] = None
        result["vlm"] = None

    return result


# ── 추론 유틸 ────────────────────────────────────────────────────────────────

def _preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(arr)


def _postprocess_rtdetr(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    confidence: float,
    max_detections: int,
    original_width: int,
    original_height: int,
) -> List[Dict[str, Any]]:
    labels = labels[0]
    boxes = boxes[0]
    scores = scores[0]

    detections: List[Dict[str, Any]] = []

    for label, box, score in zip(labels, boxes, scores):
        score_float = float(score)
        if score_float < confidence:
            continue

        class_id = int(label)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = _clip(x1, 0, original_width)
        y1 = _clip(y1, 0, original_height)
        x2 = _clip(x2, 0, original_width)
        y2 = _clip(y2, 0, original_height)

        if x2 <= x1 or y2 <= y1:
            continue

        box_width = x2 - x1
        box_height = y2 - y1

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(score_float, 6),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(box_width, 2),
                "height": round(box_height, 2),
                "area": round(box_width * box_height, 2),
            },
            "bbox_normalized": {
                "x1": round(x1 / original_width, 6) if original_width else 0.0,
                "y1": round(y1 / original_height, 6) if original_height else 0.0,
                "x2": round(x2 / original_width, 6) if original_width else 0.0,
                "y2": round(y2 / original_height, 6) if original_height else 0.0,
            },
            "is_risk_class": class_name in RISK_CLASSES,
            "is_false_positive_hint": class_name in FALSE_POSITIVE_CLASSES,
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:max_detections]


def _postprocess_yolo(
    output: np.ndarray,
    confidence: float,
    max_detections: int,
    original_width: int,
    original_height: int,
) -> List[Dict[str, Any]]:
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {predictions.shape}")

    # Ultralytics ONNX usually returns [classes+4, anchors]. Convert to [anchors, attrs].
    if predictions.shape[0] <= len(CLASS_NAMES) + 5 and predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T

    candidates: List[Dict[str, Any]] = []
    class_count = len(CLASS_NAMES)

    for prediction in predictions:
        if prediction.shape[0] < 4 + class_count:
            continue

        box = prediction[:4].astype(np.float32)
        class_scores = prediction[4:4 + class_count].astype(np.float32)

        # Some exports include objectness before class scores: [x, y, w, h, obj, cls...].
        if prediction.shape[0] >= 5 + class_count:
            objectness = float(prediction[4])
            objectness_class_scores = prediction[5:5 + class_count].astype(np.float32)
            if objectness <= 1.0 and objectness_class_scores.size == class_count:
                combined_scores = objectness * objectness_class_scores
                if float(np.max(combined_scores)) > float(np.max(class_scores)):
                    class_scores = combined_scores

        class_id = int(np.argmax(class_scores))
        score_float = float(class_scores[class_id])
        if score_float < confidence:
            continue

        cx, cy, width, height = [float(v) for v in box.tolist()]

        if max(abs(cx), abs(cy), abs(width), abs(height)) <= 1.5:
            x1 = (cx - width / 2) * original_width
            y1 = (cy - height / 2) * original_height
            x2 = (cx + width / 2) * original_width
            y2 = (cy + height / 2) * original_height
        else:
            scale_x = original_width / INPUT_IMAGE_SIZE
            scale_y = original_height / INPUT_IMAGE_SIZE
            x1 = (cx - width / 2) * scale_x
            y1 = (cy - height / 2) * scale_y
            x2 = (cx + width / 2) * scale_x
            y2 = (cy + height / 2) * scale_y

        x1 = _clip(x1, 0, original_width)
        y1 = _clip(y1, 0, original_height)
        x2 = _clip(x2, 0, original_width)
        y2 = _clip(y2, 0, original_height)

        if x2 <= x1 or y2 <= y1:
            continue

        candidates.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "confidence": score_float,
                "xyxy": [x1, y1, x2, y2],
            }
        )

    kept = _nms(candidates, NMS_IOU_THRESHOLD)[:max_detections]
    detections: List[Dict[str, Any]] = []

    for candidate in kept:
        x1, y1, x2, y2 = candidate["xyxy"]
        box_width = x2 - x1
        box_height = y2 - y1
        class_name = candidate["class_name"]

        detections.append({
            "class_id": candidate["class_id"],
            "class_name": class_name,
            "confidence": round(candidate["confidence"], 6),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(box_width, 2),
                "height": round(box_height, 2),
                "area": round(box_width * box_height, 2),
            },
            "bbox_normalized": {
                "x1": round(x1 / original_width, 6) if original_width else 0.0,
                "y1": round(y1 / original_height, 6) if original_height else 0.0,
                "x2": round(x2 / original_width, 6) if original_width else 0.0,
                "y2": round(y2 / original_height, 6) if original_height else 0.0,
            },
            "is_risk_class": class_name in RISK_CLASSES,
            "is_false_positive_hint": class_name in FALSE_POSITIVE_CLASSES,
        })

    return detections


def _nms(candidates: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    candidates = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []

    while candidates:
        current = candidates.pop(0)
        kept.append(current)
        candidates = [
            candidate
            for candidate in candidates
            if candidate["class_id"] != current["class_id"]
            or _iou(current["xyxy"], candidate["xyxy"]) < iou_threshold
        ]

    return kept


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
