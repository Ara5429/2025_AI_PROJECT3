# 전이학습 기반 졸음운전 감지 시스템
**Drowsy Driving Detection using Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI Hub의 운전자 상태 영상 데이터셋을 활용한 딥러닝 기반 졸음운전 감지 시스템입니다.
ImageNet으로 사전학습된 ResNet18을 전이학습하여 운전자의 눈 상태를 분류합니다.

## 성능 결과

| Metric | Value |
|--------|-------|
| Validation Accuracy | **78.2%** |
| ROC-AUC | **0.852** |
| 5-Fold CV Mean | **77.6% ± 2.8%** |
| Inference Time | **~15ms/image** |

### 전이학습 효과
- Pretrained ResNet18: **78.2%**
- From Scratch: **67.4%**
- **+10.8%p 성능 향상**

## 프로젝트 구조

```
drowsy-driving-detection/
├── README.md                      # 프로젝트 설명
├── requirements.txt               # 의존성 패키지
│
├── dataset.py                     # 데이터셋 클래스
├── model.py                       # 모델 정의 (ResNet Transfer Learning)
├── train.py                       # 학습 스크립트
│
├── evaluate.py                    # 모델 평가 (Confusion Matrix, ROC)
├── cross_validation.py            # 5-Fold Cross Validation
├── ablation_study.py              # Ablation Study 실험



```

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/Ara5429/drowsy-driving-detection.git
cd drowsy-driving-detection

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터셋 준비

[AI Hub](https://aihub.or.kr)에서 '졸음운전 예방을 위한 운전자 상태 정보 영상' 데이터셋을 다운로드합니다.

```
data/
├── images/           # 운전자 얼굴 이미지 (.jpg)
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/           # JSON 라벨 파일
    ├── image_001.json
    ├── image_002.json
    └── ...
```

JSON 라벨 형식:
```json
{
    "annotation": 0  // 0: Drowsy, 1: Normal
}
```

### 3. 모델 학습

```bash
python train.py \
    --image_dir ./data/images \
    --label_dir ./data/labels \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001 \
    --pretrained \
    --save_dir ./checkpoints
```

### 4. 모델 평가

```bash
python evaluate.py \
    --image_dir ./data/images \
    --label_dir ./data/labels \
```


노트북 내용:
1. 환경 설정 및 라이브러리
2. 데이터셋 로드 및 탐색
3. 모델 정의 (ResNet18 + Transfer Learning)
4. 모델 학습
5. 성능 평가 (Confusion Matrix, ROC-AUC)
6. 5-Fold Cross Validation
7. Ablation Study
8. 추론

## 실험 재현

### Cross Validation
```bash
python cross_validation.py \
    --image_dir ./data/images \
    --label_dir ./data/labels \
    --n_folds 5 \
    --epochs 30 \
    --pretrained \
    --save_dir ./cv_results
```

### Ablation Study
```bash
python ablation_study.py \
    --image_dir ./data/images \
    --label_dir ./data/labels \
    --epochs 30 \
    --save_dir ./ablation_results
```

## 모델 구조

```
ResNet18 (ImageNet Pretrained)
├── Conv1 (7×7, 64 filters, stride 2)
├── MaxPool (3×3, stride 2)
├── Layer1 (2 × BasicBlock, 64 filters)
├── Layer2 (2 × BasicBlock, 128 filters)
├── Layer3 (2 × BasicBlock, 256 filters)
├── Layer4 (2 × BasicBlock, 512 filters)
├── Global Average Pooling
└── Custom Classifier
    ├── Dropout(0.5)
    ├── Linear(512 → 256)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(256 → 2)
```

**파라미터 수**: ~11.7M

## 학습 설정

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 30 |
| Early Stopping Patience | 7 |
| LR Scheduler | ReduceLROnPlateau |
| Data Augmentation | HFlip, Rotation(±10°), ColorJitter |

## 실험 결과

### Table 1: Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ResNet18 (Scratch) | 67.4% | 0.65 | 0.68 | 0.66 |
| ResNet18 (Pretrained) | **78.2%** | 0.76 | 0.78 | 0.77 |
| ResNet50 (Pretrained) | 79.5% | 0.78 | 0.80 | 0.79 |

### Table 2: Ablation Study

| Configuration | Accuracy | Δ Accuracy |
|--------------|----------|------------|
| Full Model (Ours) | 78.2% | - |
| w/o Transfer Learning | 67.4% | -10.8%p |
| w/o Data Augmentation | 74.5% | -3.7%p |

## 한계점 및 향후 연구

**현재 한계:**
- 전체 35만 장 중 5,000장만 사용
- 단일 프레임 기반 분류 (시간적 패턴 미활용)
- 다양한 환경 변수에 대한 강건성 검증 부족

**향후 연구 방향:**
- LSTM/Transformer를 결합한 시계열 분석
- PERCLOS, EAR 등 전통적 지표와 하이브리드 방식
- MobileNet, EfficientNet-Lite 등 경량화 모델
- 실제 운전 환경 필드 테스트


## Acknowledgements

- [AI Hub](https://aihub.or.kr) - 데이터셋 제공
- [PyTorch](https://pytorch.org) - 딥러닝 프레임워크
- [torchvision](https://pytorch.org/vision) - Pretrained 모델