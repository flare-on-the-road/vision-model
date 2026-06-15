from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
import io


MODEL_PATH = "models/best.onnx"
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
        print(f"[INFO] Model loaded on {self.device}")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="FLARE Vision API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": predictor.device}


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
