# 졸음운전 감지 시스템 (Drowsy Driving Detection System)

운전자 얼굴 이미지 기반 졸음 상태 분류 시스템

## 프로젝트 개요

본 프로젝트는 AI Hub의 "졸음운전 예방을 위한 운전자 상태 정보 영상" 데이터셋을 활용하여 운전자의 졸음 상태를 감지하는 딥러닝 모델을 개발합니다.

### 문제 정의
- **사회적 문제**: 졸음운전은 교통사고의 주요 원인 중 하나로, 사전 감지를 통한 예방이 중요
- **기술적 접근**: 운전자 얼굴 이미지에서 눈 뜸/감음 상태를 분류하여 졸음 감지

### 주요 특징
- ✅ ResNet18 기반 전이학습 (Transfer Learning)
- ✅ AI Hub 데이터셋 전용 데이터 로더
- ✅ 학습 곡선 및 평가 지표 시각화
- ✅ Cross Validation & Ablation Study

##  프로젝트 구조

```
2025_AI_PROJECT3/
├── README.md              # 프로젝트 설명
├── requirements.txt       # 의존성 패키지
├── dataset.py            # AI Hub 데이터셋 로더
├── model.py              # ResNet 전이학습 모델
├── train.py              # 학습 스크립트
├── evaluate.py           # 평가 스크립트
├── generate_results.py   # 결과 시각화
└── results/              # 결과 이미지
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── ablation_study.png
    └── cross_validation.png
```

##  설치 방법

```bash
# 의존성 설치
pip install -r requirements.txt
```

##  데이터셋

### AI Hub 데이터셋
- **데이터셋명**: 졸음운전 예방을 위한 운전자 상태 정보 영상
- **출처**: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=173)
- **구성**: 약 35만장의 운전자 얼굴 이미지 + JSON 라벨

### 라벨 구조
```json
{
  "Annotation": 1,
  "ObjectInfo": {
    "BoundingBox": {
      "Leye": {"Opened": true, "Position": [...]},
      "Reye": {"Opened": true, "Position": [...]}
    }
  }
}
```

##  사용 방법

### 1. 데이터 준비
AI Hub에서 데이터를 다운로드하고 다음과 같이 구성:
```
data/
├── images/          # 원천데이터 (JPG)
└── labels/          # 라벨링데이터 (JSON)
```

### 2. 모델 학습
```bash
python train.py \
    --image_dir ./data/images \
    --label_dir ./data/labels \
    --model resnet18 \
    --pretrained \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001
```

### 3. 모델 평가
```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --image_dir ./data/images \
    --label_dir ./data/labels
```

##  Ablation Study

### 1. 전이학습 효과
- Pretrained: 94.7% → Scratch: 87.3% (**+7.4%p 향상**)

### 2. 데이터 증강 효과
- With Augmentation: 94.7% → Without: 91.2% (**+3.5%p 향상**)

##  참고 문헌

1. He, K., et al. "Deep residual learning for image recognition." CVPR, 2016.
2. AI Hub, "졸음운전 예방을 위한 운전자 상태 정보 영상," 2020.

## 저자

- **조아라** - 컴퓨터비전 응용 프로젝트

