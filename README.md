# RT-DETRv2 ONNX Fire & Smoke Detector for Replicate

커스텀 학습된 RT-DETRv2 기반 객체 탐지 모델을 ONNXRuntime GPU로 실행하고, Replicate/Cog를 통해 API 형태로 서빙하기 위한 프로젝트입니다.

이 모델은 CCTV 프레임 이미지 1장을 입력받아 `fire`, `smoke`, `carlight` 객체 후보를 탐지하고, bounding box, class name, confidence, risk score를 JSON 형태로 반환합니다.

---

## 1. 프로젝트 목적

이 프로젝트는 ITS CCTV 기반 화재 감지 시스템에서 1차 AI 탐지 모델로 사용하기 위해 구성되었습니다.

전체 시스템에서 이 모델의 역할은 다음과 같습니다.

```text
Python Worker
↓
CCTV 프레임 캡처
↓
Replicate RT-DETRv2 ONNX 모델 호출
↓
fire / smoke / carlight 후보 탐지
↓
fire 또는 smoke 후보가 있으면 VLM 2차 판단
↓
이벤트 저장
```

이 모델은 CCTV 영상을 계속 스트리밍으로 분석하는 서버가 아니라, **프레임 이미지 1장을 입력받아 추론 결과를 반환하는 API형 모델**입니다.

---

## 2. 모델 개요

| 항목            | 내용                          |
| ------------- | --------------------------- |
| 모델            | RT-DETRv2                   |
| Backbone      | HGNetv2-L                   |
| Export Format | ONNX                        |
| Runtime       | ONNXRuntime GPU             |
| Serving       | Replicate + Cog             |
| Input         | Image                       |
| Output        | JSON                        |
| Classes       | `fire`, `smoke`, `carlight` |

---

## 3. 클래스 정의

| class_id | class_name | 의미           | 처리 방식    |
| -------: | ---------- | ------------ | -------- |
|        0 | `fire`     | 화염 후보        | 위험 후보    |
|        1 | `smoke`    | 연기 후보        | 위험 후보    |
|        2 | `carlight` | 차량 등화류/조명 후보 | 오탐 판단 보조 |

`fire`, `smoke`는 위험 후보로 분류합니다.

`carlight`는 화재 위험으로 직접 판단하지 않고, 후미등/조명류로 인한 오탐 가능성을 줄이기 위한 보조 클래스로 사용합니다.

---

## 4. 프로젝트 구조

```text
rtdetr-onnx-replicate/
├─ cog.yaml
├─ predict.py
├─ requirements.txt
├─ README.md
└─ models/
   └─ model.onnx
```

### 주요 파일 설명

| 파일                  | 설명                          |
| ------------------- | --------------------------- |
| `cog.yaml`          | Replicate/Cog 빌드 환경 설정 파일   |
| `predict.py`        | ONNX 모델 로드, 전처리, 추론, 후처리 로직 |
| `requirements.txt`  | Python 패키지 의존성              |
| `models/model.onnx` | RT-DETRv2 ONNX 모델 파일        |
| `README.md`         | 프로젝트 설명 문서                  |

---

## 5. 모델 파일 위치

기본 모델 경로는 다음과 같습니다.

```text
models/model.onnx
```

`predict.py` 상단의 `MODEL_PATH` 값과 실제 파일 위치가 일치해야 합니다.

```python
MODEL_PATH = "models/model.onnx"
```

모델 파일을 다른 위치에 둘 경우 `MODEL_PATH`를 수정해야 합니다.

예시:

```python
MODEL_PATH = "weights/rtdetrv2_tunnel.onnx"
```

---

## 6. requirements.txt

현재 ONNXRuntime 기반 서빙에 필요한 최소 패키지는 다음과 같습니다.

```txt
onnxruntime-gpu
pillow
numpy
```

이미지 시각화, OpenCV 기반 전처리, bbox drawing 등을 추가할 경우 다음 패키지를 추가할 수 있습니다.

```txt
opencv-python-headless
```

---

## 7. cog.yaml

Replicate/Cog 실행 환경 설정입니다.

```yaml
build:
  gpu: true
  python_version: "3.11"

  system_packages:
    - "libgl1"
    - "libglib2.0-0"
    - "libgomp1"

  python_requirements: requirements.txt

predict: "predict.py:Predictor"
```

이 설정은 GPU 환경에서 ONNXRuntime을 사용해 모델을 실행하는 것을 전제로 합니다.

---

## 8. ONNX 입출력 스펙

### Inputs

| 이름                  | Shape              | Type      | 설명                          |
| ------------------- | ------------------ | --------- | --------------------------- |
| `images`            | `[1, 3, 640, 640]` | `float32` | 전처리된 입력 이미지                 |
| `orig_target_sizes` | `[1, 2]`           | `int64`   | 원본 이미지 크기 `[width, height]` |

### Outputs

| 이름       | Shape         | Type      | 설명                                |
| -------- | ------------- | --------- | --------------------------------- |
| `labels` | `[1, 300]`    | `int64`   | 클래스 인덱스                           |
| `boxes`  | `[1, 300, 4]` | `float32` | `x1, y1, x2, y2`, 원본 이미지 기준 픽셀 좌표 |
| `scores` | `[1, 300]`    | `float32` | confidence score                  |

---

## 9. 입력값

Replicate 모델은 이미지 파일 1장을 입력받습니다.

### Input Parameters

| 이름               |    타입 |      기본값 | 설명                   |
| ---------------- | ----: | -------: | -------------------- |
| `image`          |  file | required | 추론할 CCTV 프레임 이미지     |
| `confidence`     | float |   `0.25` | confidence threshold |
| `max_detections` |   int |    `100` | 반환할 최대 탐지 개수         |

현재 `predict.py`에서는 `iou`, `image_size`를 외부 입력으로 받지 않습니다.

이미지 크기는 내부에서 `640x640`으로 resize되며, ONNX 모델 출력 bbox는 `orig_target_sizes`를 이용해 원본 이미지 기준 좌표로 반환됩니다.

---

## 10. 전처리 방식

입력 이미지는 다음 순서로 전처리됩니다.

```text
RGB 변환
↓
640x640 resize
↓
float32 변환
↓
0~1 범위로 scale
↓
HWC → CHW 변환
↓
batch 차원 추가
↓
[1, 3, 640, 640] 형태로 ONNX 모델에 입력
```

`orig_target_sizes`에는 원본 이미지 크기를 `[width, height]` 순서로 입력합니다.

예시:

```python
orig_target_sizes = np.array(
    [[original_width, original_height]],
    dtype=np.int64,
)
```

---

## 11. 출력값

모델은 JSON 형태로 추론 결과를 반환합니다.

### Output Example

```json
{
  "success": true,
  "model": "rtdetrv2-onnx",
  "device": "cuda",
  "image": {
    "width": 1280,
    "height": 720
  },
  "thresholds": {
    "confidence": 0.25,
    "max_detections": 100
  },
  "summary": {
    "total_detections": 2,
    "risk_candidate": true,
    "risk_detection_count": 1,
    "false_positive_hint_count": 1,
    "max_confidence": 0.874321,
    "risk_max_confidence": 0.874321,
    "risk_score": 87
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "fire",
      "confidence": 0.874321,
      "bbox": {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 240.1,
        "y2": 190.7,
        "width": 119.6,
        "height": 110.4,
        "area": 13203.84
      },
      "bbox_normalized": {
        "x1": 0.094141,
        "y1": 0.111528,
        "x2": 0.187578,
        "y2": 0.264861
      },
      "is_risk_class": true,
      "is_false_positive_hint": false
    },
    {
      "class_id": 2,
      "class_name": "carlight",
      "confidence": 0.812345,
      "bbox": {
        "x1": 640.0,
        "y1": 420.0,
        "x2": 700.0,
        "y2": 455.0,
        "width": 60.0,
        "height": 35.0,
        "area": 2100.0
      },
      "bbox_normalized": {
        "x1": 0.5,
        "y1": 0.583333,
        "x2": 0.546875,
        "y2": 0.631944
      },
      "is_risk_class": false,
      "is_false_positive_hint": true
    }
  ],
  "risk_detections": [
    {
      "class_id": 0,
      "class_name": "fire",
      "confidence": 0.874321,
      "bbox": {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 240.1,
        "y2": 190.7,
        "width": 119.6,
        "height": 110.4,
        "area": 13203.84
      },
      "bbox_normalized": {
        "x1": 0.094141,
        "y1": 0.111528,
        "x2": 0.187578,
        "y2": 0.264861
      },
      "is_risk_class": true,
      "is_false_positive_hint": false
    }
  ],
  "false_positive_hints": [
    {
      "class_id": 2,
      "class_name": "carlight",
      "confidence": 0.812345,
      "bbox": {
        "x1": 640.0,
        "y1": 420.0,
        "x2": 700.0,
        "y2": 455.0,
        "width": 60.0,
        "height": 35.0,
        "area": 2100.0
      },
      "bbox_normalized": {
        "x1": 0.5,
        "y1": 0.583333,
        "x2": 0.546875,
        "y2": 0.631944
      },
      "is_risk_class": false,
      "is_false_positive_hint": true
    }
  ]
}
```

---

## 12. 출력 필드 설명

### `summary`

| 필드                          | 설명                                           |
| --------------------------- | -------------------------------------------- |
| `total_detections`          | confidence threshold를 통과한 전체 탐지 개수           |
| `risk_candidate`            | `fire` 또는 `smoke` 탐지 여부                      |
| `risk_detection_count`      | 위험 후보 탐지 개수                                  |
| `false_positive_hint_count` | `carlight` 탐지 개수                             |
| `max_confidence`            | 전체 detection 중 가장 높은 confidence              |
| `risk_max_confidence`       | `fire`, `smoke` detection 중 가장 높은 confidence |
| `risk_score`                | MVP용 위험 점수                                   |

### `detections`

confidence threshold를 통과한 전체 탐지 결과 목록입니다.

각 detection은 다음 정보를 포함합니다.

| 필드                       | 설명                        |
| ------------------------ | ------------------------- |
| `class_id`               | 클래스 ID                    |
| `class_name`             | 클래스 이름                    |
| `confidence`             | 탐지 신뢰도                    |
| `bbox`                   | 원본 이미지 기준 bounding box    |
| `bbox_normalized`        | 0~1 범위로 정규화된 bounding box |
| `is_risk_class`          | `fire`, `smoke` 여부        |
| `is_false_positive_hint` | `carlight` 여부             |

### `risk_detections`

`fire`, `smoke` 클래스만 필터링한 목록입니다.

위험 후보 판단에는 이 필드를 사용합니다.

### `false_positive_hints`

`carlight` 클래스만 필터링한 목록입니다.

이 필드는 차량 등화류, 조명류로 인한 화재 오탐 가능성을 줄이기 위한 보조 정보로 사용합니다.

---

## 13. 위험도 계산 방식

현재 `risk_score`는 MVP용 단순 계산입니다.

기본 방식은 다음과 같습니다.

```text
risk_score = fire/smoke 중 가장 높은 confidence × 100
```

추가로 위험 후보 개수에 따라 점수를 보정합니다.

```text
risk detection 2개 이상: +10
risk detection 3개 이상: +10
최대값: 100
```

예시:

```text
fire confidence 0.87
smoke confidence 0.72
risk_detection_count = 2

risk_score = 87 + 10 = 97
```

향후에는 다음 정보를 반영해 고도화할 수 있습니다.

```text
- 연속 프레임 감지 여부
- CCTV 위치
- 터널/교량/일반도로 구분
- 시간대
- VLM 2차 판단 결과
- 이전 이벤트와의 중복 여부
```

---

## 14. 로컬 테스트

### 14.1 Cog 설치

Linux/macOS 기준:

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

설치 확인:

```bash
cog --version
```

---

### 14.2 프로젝트 구조 확인

```text
rtdetr-onnx-replicate/
├─ cog.yaml
├─ predict.py
├─ requirements.txt
├─ README.md
└─ models/
   └─ model.onnx
```

---

### 14.3 로컬 추론 테스트

테스트 이미지가 `test.jpg`라고 가정합니다.

```bash
cog predict -i image=@test.jpg
```

confidence 값을 지정하려면 다음과 같이 실행합니다.

```bash
cog predict \
  -i image=@test.jpg \
  -i confidence=0.3 \
  -i max_detections=100
```

---

## 15. Replicate에 배포하기

### 15.1 Replicate 로그인

```bash
cog login
```

---

### 15.2 모델 Push

Replicate 모델 이름이 `rtdetrv2-fire-smoke-detector`라고 가정하면 다음과 같이 push합니다.

```bash
cog push r8.im/YOUR_USERNAME/rtdetrv2-fire-smoke-detector
```

`YOUR_USERNAME`은 본인의 Replicate 계정명으로 변경해야 합니다.

예시:

```bash
cog push r8.im/eunjae/rtdetrv2-fire-smoke-detector
```

---

## 16. Python에서 호출하기

Replicate Python SDK를 설치합니다.

```bash
pip install replicate
```

호출 예시는 다음과 같습니다.

```python
import replicate

output = replicate.run(
    "YOUR_USERNAME/rtdetrv2-fire-smoke-detector:MODEL_VERSION",
    input={
        "image": open("test.jpg", "rb"),
        "confidence": 0.25,
        "max_detections": 100,
    },
)

print(output)
```

`MODEL_VERSION`은 Replicate에 push한 뒤 생성되는 version hash로 교체해야 합니다.

---

## 17. Worker에서 사용하는 예시

CCTV 프레임을 캡처한 뒤 Replicate 모델에 전달하는 예시입니다.

```python
import replicate


MODEL_NAME = "YOUR_USERNAME/rtdetrv2-fire-smoke-detector:MODEL_VERSION"


def run_detection(frame_path: str):
    result = replicate.run(
        MODEL_NAME,
        input={
            "image": open(frame_path, "rb"),
            "confidence": 0.25,
            "max_detections": 100,
        },
    )

    return result


result = run_detection("frame.jpg")

if result["summary"]["risk_candidate"]:
    print("위험 후보 발견")
    print(result["risk_detections"])
else:
    print("위험 후보 없음")
```

---

## 18. 시스템 연동 방식

이 모델은 전체 시스템에서 다음 위치에 연결됩니다.

```text
ITS CCTV API
↓
CCTV 목록 수집
↓
Python Worker
↓
CCTV 프레임 캡처
↓
Replicate RT-DETRv2 ONNX 모델 호출
↓
fire/smoke 위험 후보 탐지
↓
위험 후보가 있으면 VLM 2차 판단
↓
Event DB 저장
↓
Dashboard 표시
```

권장 방식은 모든 CCTV 영상을 Replicate에 계속 스트리밍하는 것이 아니라, Worker가 일정 주기로 프레임을 캡처하고 해당 프레임 이미지만 모델에 전달하는 구조입니다.

---

## 19. 운영 시 주의사항

### 19.1 GPU 실행 확인

현재 `predict.py`는 `CUDAExecutionProvider`가 없으면 에러를 발생시키도록 구성되어 있습니다.

Replicate GPU 환경에서 정상 실행된다면 응답의 `device` 필드는 다음과 같이 표시됩니다.

```json
{
  "device": "cuda"
}
```

만약 로컬 CPU 환경에서도 테스트하고 싶다면 `predict.py`의 `setup()`에서 CPU fallback을 허용하도록 수정해야 합니다.

---

### 19.2 모델 파일 크기

`model.onnx` 파일 크기가 큰 경우 Cog build 및 Replicate push 시간이 길어질 수 있습니다.

모델을 자주 교체한다면 모델 버전과 파일명을 명확히 관리하는 것이 좋습니다.

예시:

```text
models/
├─ rtdetrv2_tunnel_v3_20260606.onnx
└─ model.onnx
```

---

### 19.3 콜드스타트

Replicate는 모델 컨테이너가 처음 실행될 때 ONNX 모델을 로드해야 하므로 첫 요청이 느릴 수 있습니다.

데모나 운영 안정성이 중요하다면 Replicate Deployment에서 minimum instance 설정을 검토할 수 있습니다.

---

### 19.4 동시 요청 처리

`predict.py` 내부에서 직접 멀티스레딩이나 비동기 처리를 구현하지 않는 것을 권장합니다.

현재 구조는 다음 원칙을 따릅니다.

```text
요청 1개
↓
이미지 1장 추론
↓
JSON 결과 반환
```

동시 요청 처리는 Replicate Deployment의 autoscaling 또는 외부 Worker/Queue 구조에서 해결하는 것이 좋습니다.

---

### 19.5 실시간 스트리밍 처리

이 프로젝트는 프레임 단위 추론에 맞춰져 있습니다.

Replicate 모델을 CCTV 스트리밍 서버처럼 계속 연결해두는 구조는 권장하지 않습니다.

권장 구조:

```text
CCTV stream
↓
frame capture
↓
image inference
↓
event decision
```

---

## 20. 향후 개선 방향

향후 다음 기능을 추가할 수 있습니다.

```text
- class별 threshold 분리
- fire/smoke 전용 risk score 고도화
- 연속 프레임 기반 위험도 계산
- 동일 CCTV 내 중복 이벤트 제거
- VLM 판단 결과와 risk score 통합
- bbox 시각화 이미지 반환
- Cloud Storage/S3에 스냅샷 저장
- 모델 버전별 성능 비교
- Replicate Deployment 기반 minimum instance 운영
- Queue 기반 다중 CCTV 처리
```

---

## 21. License

이 프로젝트는 내부 실험 및 연구/개발 목적으로 사용됩니다.

상용 배포 또는 외부 공개 시 다음 항목을 별도로 검토해야 합니다.

```text
- 학습 데이터셋 사용 권한
- 모델 가중치 배포 가능 여부
- CCTV 영상 활용 정책
- Replicate 배포 정책
- 개인정보 및 영상정보 처리 정책
```
