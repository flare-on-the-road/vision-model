from cog import BasePredictor, Input, Path

from typing import Any, Dict, List
from PIL import Image
import numpy as np
import onnxruntime as ort


MODEL_PATH = "models/best.onnx"

INPUT_IMAGE_SIZE = 640

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
    2: "carlight",
}

RISK_CLASSES = {"fire", "smoke"}
FALSE_POSITIVE_CLASSES = {"carlight"}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Replicate 컨테이너가 시작될 때 한 번만 실행된다.
        여기서 ONNX 모델을 메모리에 로드한다.
        """

        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"Available providers: {available_providers}. "
                "This model is configured for GPU inference. "
                "Check cog.yaml build.gpu or onnxruntime-gpu installation."
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.device = "cuda"

        print(f"[INFO] Loading ONNX model from: {MODEL_PATH}")
        print(f"[INFO] Available ONNX Runtime providers: {available_providers}")
        print(f"[INFO] Selected ONNX Runtime providers: {providers}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

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

    def predict(
        self,
        image: Path = Input(description="Input CCTV frame image"),
        confidence: float = Input(
            description="Confidence threshold",
            default=0.25,
            ge=0.0,
            le=1.0,
        ),
        max_detections: int = Input(
            description="Maximum number of detections to return",
            default=100,
            ge=1,
            le=300,
        ),
    ) -> Dict[str, Any]:
        """
        이미지 1장을 입력받아 ONNX RT-DETR 추론 결과를 JSON으로 반환한다.

        ONNX input:
        - images: [1, 3, 640, 640] float32
        - orig_target_sizes: [1, 2] int64, [width, height]

        ONNX output:
        - labels: [1, 300] int64
        - boxes: [1, 300, 4] float32, x1 y1 x2 y2, 원본 이미지 기준
        - scores: [1, 300] float32
        """

        image_path = str(image)

        pil_image = Image.open(image_path).convert("RGB")
        original_width, original_height = pil_image.size

        input_image = self._preprocess_image(pil_image)

        # ONNX 스펙상 원본 이미지 크기 순서는 [width, height]
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

    def _validate_model_io(self) -> None:
        """
        모델 입출력 이름이 예상과 다른 경우 초기에 바로 확인하기 위한 검증.
        """

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

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        ONNX input spec:
        images: [1, 3, 640, 640] float32

        학습/검증 설정에서 Resize 640x640 후 float32 scale=True 형태였으므로
        RGB → resize → /255.0 → CHW → NCHW 순서로 변환한다.
        """

        resized = image.resize(
            (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.array(resized).astype(np.float32) / 255.0

        # HWC -> CHW
        image_array = np.transpose(image_array, (2, 0, 1))

        # CHW -> NCHW
        image_array = np.expand_dims(image_array, axis=0)

        # ONNXRuntime에 안정적으로 넘기기 위해 contiguous 보장
        image_array = np.ascontiguousarray(image_array)

        return image_array.astype(np.float32)

    def _build_input_feed(
        self,
        images: np.ndarray,
        orig_target_sizes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        ONNX 모델 input 이름에 맞춰 입력 dict를 만든다.
        """

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
        """
        ONNX output을 이름 기준으로 labels, boxes, scores에 매핑한다.

        Output spec:
        - labels: [1, 300]
        - boxes: [1, 300, 4]
        - scores: [1, 300]
        """

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
        """
        labels, boxes, scores를 JSON-friendly detection list로 변환한다.

        boxes는 이미 원본 이미지 기준 x1, y1, x2, y2 픽셀 좌표로 나온다는
        스펙이므로 별도의 scale 변환을 하지 않는다.
        """

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

            # 모델 출력이 이미지 범위를 조금 벗어나는 경우 방어적으로 clip
            x1 = self._clip(x1, 0, original_width)
            y1 = self._clip(y1, 0, original_height)
            x2 = self._clip(x2, 0, original_width)
            y2 = self._clip(y2, 0, original_height)

            # 유효하지 않은 box는 제외
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
        """
        MVP용 위험도 계산.
        fire/smoke 탐지 confidence와 개수를 기반으로 단순 계산한다.
        이후에는 연속 프레임 감지 여부, CCTV 위치, VLM 결과 등을 반영하면 된다.
        """

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