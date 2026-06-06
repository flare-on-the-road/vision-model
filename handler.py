import base64
import io
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

import runpod


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.onnx")
INPUT_IMAGE_SIZE = 640

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
    2: "carlight",
}

RISK_CLASSES = {"fire", "smoke"}
FALSE_POSITIVE_CLASSES = {"carlight"}


class RTDETRv2OnnxDetector:
    def __init__(self) -> None:
        self.device = "unknown"
        self.session = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        self._load_model()

    def _load_model(self) -> None:
        available_providers = ort.get_available_providers()
        require_cuda = os.getenv("REQUIRE_CUDA", "1") == "1"

        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "cuda"
        else:
            if require_cuda:
                raise RuntimeError(
                    "CUDAExecutionProvider is not available. "
                    f"Available providers: {available_providers}. "
                    "RunPod GPU worker should normally provide CUDAExecutionProvider. "
                    "For local CPU testing, set REQUIRE_CUDA=0."
                )

            providers = ["CPUExecutionProvider"]
            self.device = "cpu"

        print(f"[INFO] Loading ONNX model from: {MODEL_PATH}")
        print(f"[INFO] Available ONNX Runtime providers: {available_providers}")
        print(f"[INFO] Selected ONNX Runtime providers: {providers}")

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

        print("[INFO] ONNX model loaded successfully")

    def _validate_model_io(self) -> None:
        required_inputs = {"images", "orig_target_sizes"}
        required_outputs = {"labels", "boxes", "scores"}

        input_name_set = set(self.input_names)
        output_name_set = set(self.output_names)

        missing_inputs = required_inputs - input_name_set
        missing_outputs = required_outputs - output_name_set

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
        image: Image.Image,
        confidence: float = 0.25,
        max_detections: int = 100,
    ) -> Dict[str, Any]:
        pil_image = image.convert("RGB")
        original_width, original_height = pil_image.size

        input_image = self._preprocess_image(pil_image)

        # ONNX 스펙상 원본 크기 순서는 [width, height]
        orig_target_sizes = np.array(
            [[original_width, original_height]],
            dtype=np.int64,
        )

        input_feed = self._build_input_feed(
            images=input_image,
            orig_target_sizes=orig_target_sizes,
        )

        outputs = self.session.run(
            self.output_names,
            input_feed,
        )

        parsed_outputs = self._parse_outputs(outputs)

        detections = self._postprocess(
            labels=parsed_outputs["labels"],
            boxes=parsed_outputs["boxes"],
            scores=parsed_outputs["scores"],
            confidence=confidence,
            max_detections=max_detections,
            original_width=original_width,
            original_height=original_height,
        )

        risk_detections = [
            det for det in detections
            if det["class_name"] in RISK_CLASSES
        ]

        false_positive_hints = [
            det for det in detections
            if det["class_name"] in FALSE_POSITIVE_CLASSES
        ]

        max_confidence = max(
            [det["confidence"] for det in detections],
            default=0.0,
        )

        risk_max_confidence = max(
            [det["confidence"] for det in risk_detections],
            default=0.0,
        )

        risk_score = self._calculate_risk_score(risk_detections)

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

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        resized = image.resize(
            (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.array(resized).astype(np.float32) / 255.0

        # HWC -> CHW
        image_array = np.transpose(image_array, (2, 0, 1))

        # CHW -> NCHW
        image_array = np.expand_dims(image_array, axis=0)

        image_array = np.ascontiguousarray(image_array)

        return image_array.astype(np.float32)

    def _build_input_feed(
        self,
        images: np.ndarray,
        orig_target_sizes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        input_feed: Dict[str, np.ndarray] = {}

        for name in self.input_names:
            if name == "images":
                input_feed[name] = images
            elif name == "orig_target_sizes":
                input_feed[name] = orig_target_sizes
            else:
                raise ValueError(
                    f"Unexpected ONNX input name: {name}. "
                    f"Expected 'images' or 'orig_target_sizes'."
                )

        return input_feed

    def _parse_outputs(
        self,
        outputs: List[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        output_map = {
            name: value
            for name, value in zip(self.output_names, outputs)
        }

        return {
            "labels": output_map["labels"],
            "boxes": output_map["boxes"],
            "scores": output_map["scores"],
        }

    def _postprocess(
        self,
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

            x1 = self._clip(x1, 0, original_width)
            y1 = self._clip(y1, 0, original_height)
            x2 = self._clip(x2, 0, original_width)
            y2 = self._clip(y2, 0, original_height)

            if x2 <= x1 or y2 <= y1:
                continue

            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height

            detection = {
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
                    "area": round(area, 2),
                },
                "bbox_normalized": {
                    "x1": round(x1 / original_width, 6)
                    if original_width
                    else 0.0,
                    "y1": round(y1 / original_height, 6)
                    if original_height
                    else 0.0,
                    "x2": round(x2 / original_width, 6)
                    if original_width
                    else 0.0,
                    "y2": round(y2 / original_height, 6)
                    if original_height
                    else 0.0,
                },
                "is_risk_class": class_name in RISK_CLASSES,
                "is_false_positive_hint": class_name in FALSE_POSITIVE_CLASSES,
            }

            detections.append(detection)

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return detections[:max_detections]

    def _calculate_risk_score(
        self,
        risk_detections: List[Dict[str, Any]],
    ) -> int:
        if not risk_detections:
            return 0

        max_conf = max(det["confidence"] for det in risk_detections)
        count = len(risk_detections)

        score = int(max_conf * 100)

        if count >= 2:
            score += 10

        if count >= 3:
            score += 10

        return min(score, 100)

    def _clip(
        self,
        value: float,
        min_value: float,
        max_value: float,
    ) -> float:
        return max(min_value, min(value, max_value))


def load_image_from_base64(image_base64: str) -> Image.Image:
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_image_from_url(image_url: str) -> Image.Image:
    response = requests.get(
        image_url,
        timeout=30,
        headers={
            "User-Agent": "rtdetrv2-runpod-worker/1.0"
        },
    )
    response.raise_for_status()

    return Image.open(io.BytesIO(response.content)).convert("RGB")


def load_image_from_input(job_input: Dict[str, Any]) -> Image.Image:
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    if image_url:
        return load_image_from_url(image_url)

    if image_base64:
        return load_image_from_base64(image_base64)

    raise ValueError(
        "Missing image input. Provide either 'image_url' or 'image_base64'."
    )


# 모델은 worker 시작 시 한 번만 로드한다.
# RunPod 문서/템플릿에서도 무거운 모델은 handler 내부가 아니라
# 스크립트 시작 시점에 로드하는 방식을 권장한다.
DETECTOR = RTDETRv2OnnxDetector()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless entrypoint.

    Expected input:

    {
      "input": {
        "image_url": "https://example.com/frame.jpg",
        "confidence": 0.25,
        "max_detections": 100
      }
    }

    또는

    {
      "input": {
        "image_base64": "...",
        "confidence": 0.25,
        "max_detections": 100
      }
    }
    """

    try:
        job_input = event.get("input", {})

        confidence = float(job_input.get("confidence", 0.25))
        max_detections = int(job_input.get("max_detections", 100))

        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if max_detections < 1 or max_detections > 300:
            raise ValueError("max_detections must be between 1 and 300")

        image = load_image_from_input(job_input)

        result = DETECTOR.predict(
            image=image,
            confidence=confidence,
            max_detections=max_detections,
        )

        return result

    except Exception as exc:
        error_trace = traceback.format_exc()
        print("[ERROR] Inference failed")
        print(error_trace)

        return {
            "success": False,
            "error": str(exc),
            "traceback": error_trace,
        }


runpod.serverless.start({"handler": handler})