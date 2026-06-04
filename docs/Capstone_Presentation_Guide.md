본 문서는 아마존 패션 데이터셋을 활용하여 진행된 평점 예측 멀티모달 모델들의 버전별 아키텍처 및 핵심 변경 사항을 정리한 요약본입니다.

## 📊 1. 모델 버전별 비교 테이블

| Model | Text Backbone | Image Backbone | Fusion Module | Key Modification |
| :--- | :--- | :--- | :--- | :--- |
| **V1: Baseline** | RoBERTa-base | EfficientNet-B0 | **Simple Concatenation** (단순 결합) | 2단계 전이학습 도입<br>*(글/사진을 무시하고 가격 수치에만 의존하는 Shortcut Learning 발생)* |
| **V2: Multi-task** | RoBERTa-base | EfficientNet-B0 | **3-Way GMU** (Gated Multimodal Unit) | 가격 편식 방지를 위해 이미지/텍스트 각각 독립된 손실을 부여하는 **Multi-task Loss** 도입 |
| **V3: Isolation** | RoBERTa-base | EfficientNet-B0 | **3-Way GMU** | 텍스트 편향을 억제하고 이미지 학습을 강제하기 위한 **Targeted Modality Dropout** (텍스트 80% 마스킹) 도입 |
| **V4: Mobile-V2** | RoBERTa-base | MobileNet-V2 | **3-Way GMU** | 실무 모바일 서빙에 적합하도록 이미지 백본을 **초경량 MobileNet-V2**로 대체 |
| **V5: Colab TF** | RoBERTa-base | MobileNet-V2 | **3-Way GMU** / **Cross-Attention** | TensorFlow / Keras 포팅 및 Google Colab 분산 학습 파이프라인 (Sequence 기반 데이터 배포) 최적화 |
| **V6: Intra-Self** | RoBERTa-base | MobileNet-V3 Large | **Intra-Modality Self-Attention**<br>+ **Inter-Modality Cross-Attention** | 단어 시퀀스와 이미지 패치 내부의 문맥/공간 관계를 먼저 분석하는 **자체 어텐션(Self-Attention)** 선행 처리 및 **MobileNet-V3 Large** 업그레이드 |
| **V7: Cross-Attn** | RoBERTa-base | MobileNet-V3 Large | **Intra-Self + Inter-Cross Attention** | 어텐션의 정밀한 정렬을 강제하는 대조 학습 기법인 **CCR (대조 함량 재구성)** & **CCS (대조 함량 스와핑) Loss** 도입 |
| **V8: Multi-Layer** | RoBERTa-base | MobileNet-V3 Large | **Multi-Layer Cross-Attention (2층)** | 상호 참조 블록을 **다층(2층) 구조화**하고, 배치 내에서 가장 헷갈리는 오답을 Negative로 샘플링하는 **CCS Hard Negative Mining** 탑재 |
| **V9: Ultimate** | RoBERTa-base | MobileNet-V3 Large | **Multi-Layer Cross-Attention (2층)** | V8 아키텍처에 차등 학습률(Differential LR) 및 코사인 어닐링 스케줄러 기반 **10에폭 최적 학습(Full Unfreeze)**을 적용하여 최고 정확도(최종 MAE `0.3532`) 달성 |
| **V10: 4-Way Modal** | RoBERTa-base | MobileNet-V3 Large | **4-Way GMU + Multi-Layer Cross-Attention** | 리뷰(Review)와 판매자(Seller) 텍스트를 분리하여 4개 모달리티(Seller, Review, Image, Tabular) 융합. 성능 향상(최종 MAE `0.3150`)은 있었으나, 게이트가 리뷰 텍스트에 극단적으로 쏠리는(81%) 과의존 현상 발생 |
| **V12: Balanced Fusion (λ=0.02)** | RoBERTa-base | MobileNet-V3 Large | **Concatenate-Attend-Split Cross-Attention** | V10의 리뷰 과의존을 해결하기 위해 **Review Modality Dropout (P=0.3)** 및 **Gate Entropy Loss (λ=0.02)** 도입. 모달리티 간 균형 잡힌 게이트 가중치(약 25%씩)를 달성하며 크게 개선된 정확도 달성 (최종 MAE `0.2877`) |
| **V12: Balanced Fusion (λ=0.01)** | RoBERTa-base | MobileNet-V3 Large | **Concatenate-Attend-Split Cross-Attention** | Gate Entropy Loss 비중을 **λ=0.01**로 조정하여 규제 강도를 완화. 모달리티 균형을 안정적으로 유지하면서도 융합 성능을 최적화하여 최고 정확도 경신 (최종 MAE `0.2872`) |

---

## ⚡ 2. 핵심 고도화 기술 설명

### 1) 3-Way Gated Multimodal Unit (GMU)
* 단순 피처 결합(`torch.cat`) 시 발생하기 쉬운 지름길 학습(Shortcut Learning)을 방지하기 위해 도입되었습니다.
* 텍스트, 이미지, 정형 데이터 각각의 게이트 가중치(Softmax)를 동적으로 연산하여 매 입력마다 최적의 융합 비율을 설정합니다.

### 2) 타겟형 모달리티 드롭아웃 (Targeted Modality Dropout)
* 텍스트 인코더(RoBERTa)의 성능이 지나치게 강해 이미지를 학습하지 않는 문제를 해결하기 위해, 학습 중 **80% 확률로 텍스트 입력을 강제로 마스킹**합니다. 텍스트가 없는 환경에서도 사진만으로 평점을 맞추게 훈련함으로써 모델의 이미지 표현력을 대폭 향상시킵니다.

### 3) 하이브리드 멀티태스크 학습 (Multi-task Loss)
* 융합 모델의 최종 평점 예측뿐만 아니라, `텍스트 단독 예측` 및 `이미지 단독 예측`을 동시에 수행하게 하여 각 모달리티의 도메인 특성이 융합 과정에서 지워지거나 희석되지 않도록 강력하게 제어합니다.

### 4) 다층 상호 참조 어텐션 (Multi-Layer Cross-Attention)
* 단어 시퀀스와 이미지의 패치 레벨 영역이 서로 교차해가며 어텐션을 주고받아 시너지를 극대화합니다. 2개 층으로 쌓아 깊고 세밀한 정렬이 가능합니다.

### 5) Hard Negative Mining 대조 학습
* 기존 배치 셔플 기반의 단순 대조 학습을 발전시켜, **현재 학습 배치 내에서 의미가 가장 유사하여 헷갈리는 이미지**를 오답(Negative)으로 지정해 훈련합니다. 모델이 아주 세부적인 질감이나 로고까지 꼼꼼히 확인하고 판단하게 강제합니다.

### 6) 텍스트 분리 및 4-Way 융합 (V10)
* 하나의 텍스트로 합쳐서 처리하던 방식을 개선하여, **판매자 텍스트(Seller Text)**와 **리뷰 텍스트(Review Text)**를 분리 입력합니다. 이를 통해 이미지, 정형 데이터(Tabular)와 함께 총 4개의 독립적인 모달리티로 세분화하여 모델이 각 정보의 특성을 더 정밀하게 학습하도록 유도합니다.

### 7) 모달리티 균형화 (Modality Balancing) (V12)
* **Concatenate-Attend-Split**: Seller와 Review 텍스트를 결합(Concatenate)하여 이미지와 교차 어텐션을 수행한 후 다시 분리(Split)함으로써 파라미터와 연산량을 효율화합니다.
* **Review Modality Dropout**: V10에서는 모델이 평점을 예측할 때 가장 쉬운 힌트인 '리뷰 텍스트(Review)'에만 81%의 비중으로 과하게 의존하고(지름길 학습), 이미지나 판매자 정보 등은 무시하는 현상이 발생했습니다. 이를 막기 위해 학습 중 30% 확률로 리뷰 피처를 마스킹하여 모델이 다양한 모달리티 고르게 학습하도록 강제했습니다.
* **Gate Entropy Loss**: GMU의 게이트 가중치 분포에 엔트로피(Entropy) 손실을 추가하여 특정 모달리티로 게이트가 편중되는 현상을 방지했습니다. 이를 통해 4개의 모달리티(Seller, Review, Image, Tabular) 가중치가 각각 약 0.25(25%)씩 고르게 분배되었으며, 모델이 모든 정보를 종합적으로 판단하게 되었습니다.
* **규제 강도 조정 (λ=0.02 → 0.01)**: 초기에는 엔트로피 규제를 강하게(λ=0.02) 주어 25% 균형 분배를 엄격하게 강제했습니다. 하지만 규제를 약간 완화(λ=0.01)함으로써, 전체적인 평균 균형은 25% 내외로 유지하되 개별 샘플의 특성에 따라 더 중요한 모달리티에 유연하게 집중할 수 있는 여유를 모델에 부여했습니다. 이러한 융통성 확보가 최종 융합 성능 최적화(MAE 0.2872)로 이어졌습니다.
