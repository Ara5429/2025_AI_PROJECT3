## 전이학습 기반 졸음운전 감지 시스템  
**Drowsy Driving Detection using Transfer Learning**

본 저장소는 AI Hub에서 제공하는 *운전자 상태 영상 데이터셋*을 활용하여  
전이학습(Transfer Learning) 기반의 졸음운전 감지 모델을 구현한 코드이다.  
ImageNet으로 사전학습된 CNN 모델을 이용하여 운전자의 눈 상태를 분석하고,  
정상 상태와 졸음 상태를 분류하는 것을 목표로 한다.

---

### 모델 구조 (Model)
- **Backbone**: ResNet18 (ImageNet pretrained)
- **분류 문제**: 이진 분류  
  - 0: 졸음 상태 (Drowsy)  
  - 1: 정상 상태 (Normal)

---

### 학습 설정 (Training Setup)
- **데이터 분할**: Train / Validation = 80% / 20%
- **최적화 기법**: AdamW Optimizer
- **학습 에포크 수**: 30 epochs
- **데이터 증강(Data Augmentation)**  
  - Random Horizontal Flip  
  - Random Rotation (±10°)  
  - Color Jitter  

---

### 성능 결과 (Performance)
- **검증 정확도(Validation Accuracy)**: **78.2%**

---

### 실행 방법 (Run)
아래 명령어를 통해 모델 학습을 실행할 수 있다.

```bash
python train.py
