# 🛍️ 아마존 패션 멀티모달 평점 예측 마스터 가이드
> **Amazon Fashion Multimodal Rating Prediction: Ultimate Repository Guide (V1 to V12)**

본 프로젝트는 아마존 패션 데이터셋(Amazon Fashion Dataset)을 활용하여 **리뷰 텍스트(RoBERTa), 상품 이미지(MobileNet-V3 Large), 정형 데이터(가격, 결측 여부, 카테고리 임베딩)**를 결합하고, 평점(1.0 ~ 5.0점)을 초고정밀로 예측하는 **경량화 멀티모달 딥러닝 추천 시스템**입니다.

---

## 📂 레포지토리 폴더 구조 (Repository Structure)

```text
BigData/
 ├── src/          # 멀티모달 평점 예측 메인 파이썬 소스 코드 (v2~v12)
 │    ├── runpod/  # 클라우드(RunPod) 서버 전용 학습 스크립트
 │    └── utils/   # 분석(게이트 비중 확인), 평가(evaluate)용 부가 스크립트
 ├── data/         # 훈련/테스트 데이터 및 이미지 (용량 문제로 Git 제외)
 ├── models/       # 학습된 모델 가중치 체크포인트 파일 (.pth 등)
 ├── results/      # 테스트 결과 로그, 예측 지표 및 시각화 그래프 (.png 등)
 ├── docs/         # 프로젝트 아키텍처 및 각종 연구 리포트 문서 (.md, .pdf)
 ├── .gitignore    # Git 업로드 제외 목록
 ├── requirements.txt # 패키지 의존성 목록
 └── README.md     # 메인 설명서
```

---

## 📊 모델 버전별 발전사 및 최종 성과 (Step-by-Step Evolution)

프로젝트 초기 베이스라인(MAE 1.5)에서부터 시작하여, 텍스트 편향을 억제하고 이미지 학습을 극대화하여 최종 완성한 V12(MAE 0.29)까지의 전체 개발 연대기입니다.

| 단계 | 모델 버전 | 핵심 아키텍처 및 학습 전략 | 최종 MAE ⬇️ | MSE ⬇️ | R2 Score ⬆️ | 기술적 성과 및 극복 과제 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Step 1** | **V1: Baseline** | 3-way 단순 결합 (Simple Concatenation) + 2단계 전이학습 | `1.5231` | `2.3102` | `12.30%` | **Shortcut Learning 발생**: 글/사진을 무시하고 오직 수치인 '가격'에만 의존하여 다 찍어버림. |
| **Step 2** | **V2: Multi-task** | **GMU(Gated Unit)** 융합 + 이미지/텍스트 독립 Loss 부여 | `0.3499` | `0.3412` | `83.10%` | **정밀도 대폭 개선**: 가격 편식 방지 및 리뷰 감성 추출 성공. 단, **텍스트(RoBERTa) 편향** 문제 발생. |
| **Step 3** | **V3: Isolation** | **Targeted Modality Dropout** (텍스트 80% 의도적 차단) | `0.3461` | `0.3392` | `83.25%` | **텍스트 편향 억제**: 글씨 없이 사진만 보고 평점을 맞추게 강제하여 이미지 특징 추출력 상향 평준화. |
| **Step 4** | **V4: Mobile-V2** | 백본 **MobileNet-V2** 적용 + 초경량 모바일 서빙 최적화 | `0.3944` | `0.3926` | `80.12%` | **경량 최적화**: EfficientNet 대비 정확도 하락을 단 5%로 막고 연산 속도를 극대화한 실무형 아키텍처. |
| **Step 5** | **V5: Colab TF** | TensorFlow / Keras 포팅 및 코랩 분산 학습 파이프라인 | `0.3973` | `0.3952` | `79.95%` | 강의(BPM) 맞춤형 Keras Sequence 기반 데이터 배포 및 훈련 세팅 완료. |
| **Step 6** | **V6: Intra-Self** | **Intra-Modality Self-Attention** (모달리티 내 자체 어텐션) | `0.3851` | `0.3802` | `81.45%` | 텍스트와 이미지 내부에서 각각 문맥 및 공간 배치를 먼저 선행 분석하도록 고도화. |
| **Step 7** | **V7: Cross-Attn** | **Inter-Modality Cross-Attention** + **CCR/CCS 대조 학습** | `0.3698` | `0.3601` | `82.43%` | **정렬(Alignment)**: 단어와 사진 영역 간 매칭 관계를 대조 학습(Contrastive Loss)으로 강제 정렬. |
| **Step 8** | **V8: Multi-Layer** | **다층(2층) Cross-Attention** + **CCS Hard Negative Mining** | `0.3764` | `0.3648` | `82.21%` | **구조 고도화**: 배치 내 가장 헷갈리는 사진을 오답으로 삼았으나, 5에폭으로는 완전히 수렴하지 못함. |
| **Step 9** | **V9: Ultimate** | **다층 Cross-Attention + Hard Negative CCS + 10에폭 최적화** | `0.3532` | `0.3345` | `83.68%` | **아키텍처 잠재력 발현**: 경량 백본으로 융합 고성능 달성. |
| **Step 10** | **V10: 4-Way Modal** | **텍스트 분리 (Seller/Review) + 4-Way GMU** | `0.3157` | `0.3038` | `85.18%` | **성능 대폭 향상**: 단, 모델이 가장 쉬운 힌트인 '리뷰 텍스트'에만 과의존(81%)하는 지름길 학습 발생. |
| **Step 11** | **V12: Balanced (λ=0.02)** | **Review Dropout + Gate Entropy Loss (λ=0.02)** | `0.2898` | `0.3274` | `84.03%` | **모달리티 균형화**: 리뷰 편향을 막고 4개 모달리티가 각각 25%씩 기여하도록 강제하여 오차 대폭 감소. |
| **Step 12** | **V12: Balanced (λ=0.01)** | **Gate Entropy 규제 완화 (λ=0.01) + Concatenate-Attend-Split** | **`0.2906`** | **`0.3214`** | **`84.32%`** | **최종 완성형**: 유연성을 부여하여 **MAE 0.29 벽 돌파 및 역대 최고 정확도 달성!** |

---

## 🗂️ 데이터셋 진화 과정 (Dataset Evolution)

모델이 발전함에 따라 사용된 CSV 데이터셋도 점진적으로 고도화되었습니다. (`data/` 폴더 내 위치)

* **`fashion_train_subset_2_with_images.csv`**: 프로젝트 초중반(V1~V9)에 사용된 메인 학습 데이터셋입니다. 리뷰 텍스트, 평점, 매칭되는 이미지 정보가 기본적으로 담겨 있습니다.
* **`fashion_train_subset_3_with_help.csv`**: 기존 데이터셋에 리뷰의 '도움이 된 투표 수(Helpful Vote)' 메타 정보를 추가로 결합하여 만들어낸 파생 데이터입니다.
* **`fashion_train_subset_3_with_meta_text_v10.csv`**: 모델 V10 이상(최신)을 위해 특수하게 가공된 가장 진보된 데이터셋입니다. 판매자의 상품 원본 메타 텍스트(`seller_text`)와 유저의 실제 리뷰(`review_text`)가 분리되어 입력되도록 고도화되었습니다.

---

## 💻 모델 평가 실행 가이드 (로컬 환경)

최신 V10 및 V12 모델에 대한 정밀 평가는 아래의 명령어로 즉시 가동할 수 있습니다. 

```bash
# 1. 의존성 패키지 설치
pip install torch torchvision transformers pandas scikit-learn pillow tqdm matplotlib seaborn

# 2. 로컬에서 최신 V10 및 V12 모델 정밀 평가 실행
python src/utils/evaluate_v10_v12.py
```

평가가 완료되면 `results/model_evaluate/` 폴더 내에 모델별 최신 예측 지표(MAE 등)와 시각화 그래프 3종(상관관계, 오차 누적, 성능 요약)이 자동 생성됩니다.
