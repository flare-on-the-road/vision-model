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
from PIL import Image
import io

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
REQUIRE_CUDA = os.getenv("VISION_REQUIRE_CUDA", "true").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _int_env("PORT", 8000)
INPUT_IMAGE_SIZE = 640

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
    2: "carlight",
}

RISK_CLASSES = {"fire", "smoke"}
FALSE_POSITIVE_CLASSES = {"carlight"}


class OnnxPredictor:
    def __init__(self):
        self.session = None
        self.device = None
        self.input_names = []
        self.output_names = []

    def load(self):
        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "cuda"
        elif REQUIRE_CUDA:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"Available providers: {available_providers}. "
                "Install a CUDA-compatible onnxruntime-gpu environment or set "
                "VISION_REQUIRE_CUDA=false for local CPU testing."
            )
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"

        print(f"[INFO] Loading ONNX model: {MODEL_PATH}")
        print(f"[INFO] Providers: {providers}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=session_options,
            providers=providers,
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[INFO] Input names: {self.input_names}")
        print(f"[INFO] Output names: {self.output_names}")
        self._validate_model_io()
        print(f"[INFO] Model loaded on {self.device}")

    def _validate_model_io(self):
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

        input_image = _preprocess_image(pil_image)
        orig_target_sizes = np.array([[original_width, original_height]], dtype=np.int64)

        input_feed = {
            name: (input_image if name == "images" else orig_target_sizes)
            for name in self.input_names
        }

        outputs = self.session.run(self.output_names, input_feed)
        output_map = dict(zip(self.output_names, outputs))

        detections = _postprocess(
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
        risk_score = _calculate_risk_score(risk_detections)

        return {
            "success": True,
            "model": "rtdetrv2-onnx",
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
                "risk_candidate": len(risk_detections) > 0,
                "risk_detection_count": len(risk_detections),
                "false_positive_hint_count": len(false_positive_hints),
                "max_confidence": max_confidence,
                "risk_max_confidence": risk_max_confidence,
                "risk_score": risk_score,
            },
            "detections": detections,
            "risk_detections": risk_detections,
            "false_positive_hints": false_positive_hints,
        }


predictor = OnnxPredictor()

# ── VLM 설정 ─────────────────────────────────────────────────────────────────

_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "")
_OPENROUTER_TIMEOUT = _int_env("OPENROUTER_TIMEOUT", 30)
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_VLM_PROMPT = (
    "이 이미지는 Vision AI가 객체를 탐지한 결과입니다. 이미지에는 바운딩 박스(BBox)와 레이블이 표시되어 있습니다.\n\n"
    "각 BBox의 레이블이 실제 이미지 내용과 일치하는지 검증해주세요.\n\n"
    "오탐으로 판정하는 기준:\n"
    "- BBox 레이블이 실제 객체와 다른 경우 (예: 불꽃인데 carlight로 표기, 구름인데 smoke로 표기)\n"
    "- 실제로 해당 객체가 없는데 감지된 경우\n"
    "- 조명·반사·햇빛 등 유사 시각 패턴을 화염으로 잘못 분류한 경우\n"
    "- 이 시스템은 터널 환경에서 운영되며 터널 내 구름은 발생하지 않음. "
    "매연·소화 분말·증기·먼지 등을 화재 연기로 잘못 분류한 경우\n\n"
    "다음 두 가지만 한국어로 간결하게 답변해주세요:\n"
    "1. 오탐 여부: yes(오탐) 또는 no(정상 탐지)\n"
    "2. 오탐이 맞을 경우 오탐 원인 분석 / 오탐이 아닐 경우 \"pass\""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="FLARE Vision API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": predictor.device,
        "model_path": MODEL_PATH,
        "providers": ort.get_available_providers(),
    }


def _call_vlm(image_bytes: bytes) -> Optional[dict]:
    """OpenRouter VLM 호출 내부 함수. 실패 시 None 반환 (파이프라인 중단 방지)."""
    if not _OPENROUTER_API_KEY or not _OPENROUTER_MODEL:
        logger.warning("[VLM] OPENROUTER_API_KEY 또는 OPENROUTER_MODEL 미설정 → 생략")
        return None
    logger.info(f"[VLM] 호출 시작 (model={_OPENROUTER_MODEL})")
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": _OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 256,
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
        logger.info(f"[VLM] 응답 원문: {raw[:200]}")
    except Exception as e:
        logger.error(f"[VLM] 호출 실패: {e}")
        return None
    result = _parse_vlm_response(raw)
    logger.info(f"[VLM] 파싱 결과: {result}")
    return result


@app.post("/vlm")
async def vlm(image: UploadFile = File(...)):
    """이미지를 OpenRouter VLM으로 분석해 오탐 여부를 반환한다."""
    if not _OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY 미설정")
    if not _OPENROUTER_MODEL:
        raise HTTPException(status_code=503, detail="OPENROUTER_MODEL 미설정")

    image_bytes = await image.read()
    result = _call_vlm(image_bytes)
    if result is None:
        raise HTTPException(status_code=502, detail="VLM 호출 또는 파싱 실패")
    return result


def _parse_vlm_response(raw: str):
    match = re.search(r"1\.\s*오탐\s*여부\s*[:：]\s*(yes|no)", raw, re.IGNORECASE)
    if not match:
        return None
    is_false_positive = match.group(1).lower() == "yes"
    reason_match = re.search(r"2\.\s*(.+)", raw)
    reason = reason_match.group(1).strip() if reason_match else ""
    return {"is_fire": not is_false_positive, "reason": reason}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    confidence: float = Form(default=0.25, ge=0.0, le=1.0),
    max_detections: int = Form(default=100, ge=1, le=300),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image 파일만 허용됩니다.")

    image_bytes = await image.read()

    try:
        result = predictor.predict(
            image_bytes=image_bytes,
            confidence=confidence,
            max_detections=max_detections,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # fire/smoke 탐지 시 VLM 2차 판단을 내부에서 즉시 수행
    if result["summary"]["risk_candidate"]:
        result["vlm"] = _call_vlm(image_bytes)
    else:
        result["vlm"] = None

    return result


# ── 추론 유틸 ────────────────────────────────────────────────────────────────

def _preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(arr)


def _postprocess(
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


def _calculate_risk_score(risk_detections: List[Dict[str, Any]]) -> int:
    if not risk_detections:
        return 0

    max_conf = max(d["confidence"] for d in risk_detections)
    count = len(risk_detections)
    score = int(max_conf * 100)

    if count >= 2:
        score += 10
    if count >= 3:
        score += 10

    return min(score, 100)


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
