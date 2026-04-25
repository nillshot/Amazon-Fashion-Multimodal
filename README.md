# Amazon Fashion 멀티모달 평점 예측 프로젝트 종합 가이드

이 프로젝트는 Amazon Fashion 데이터셋을 활용하여 리뷰 텍스트, 상품 이미지, 그리고 메타데이터(가격, 브랜드 등)를 모두 통합 분석해 상품 평점을 예측하는 멀티모달 딥러닝 시스템입니다.
본 프로젝트의 핵심은 **Gated Multimodal Unit (GMU)**을 도입하여, 시각적 정보와 텍스트 설명, 메타데이터가 **상호보완적으로 구매 결정(평점)에 미치는 영향**을 동적으로 분석하는 데 있습니다.

---

## 1. 프로젝트 디렉토리 구조 및 파일 설명

```text
/BigData
├── Amazon_Fashion.jsonl.gz         # 원본 리뷰 데이터 (Gzip)
├── meta_Amazon_Fashion.jsonl.gz    # 원본 상품 메타데이터 (Gzip)
├── joined_reviews.jsonl.gz         # Phase 1: 리뷰와 메타데이터가 병합된 중간 데이터
├── fashion_train_subset_2.csv      # Phase 2: 필터링 및 정제된 최종 학습용 데이터셋 (본문 설명 참고)
├── subset_100_with_images.csv      # Phase 3: 이미지 로컬 경로가 매핑된 샘플 데이터셋 (142건)
├── images/                         # 다운로드된 상품 이미지 파일 폴더
├── join_data.py                    # 데이터 병합 스크립트 (Reviews + Meta)
├── prepare_dataset.py              # 데이터 정제 및 특징 추출 스크립트 (CSV 생성)
├── download_fashion_images_full.py # 전체 3.8만 건 이미지 자동 다운로드 및 매핑 스크립트
├── multimodal_model_full.py        # 로컬 전체 데이터셋 학습 스크립트 (최신 튜닝 적용)
├── inference.py                    # 단일 샘플 평점 예측 추론 스크립트
└── README.md                       # 본 프로젝트 가이드 문서
```

---

## 2. 데이터 전처리 및 구축 과정

### Step 1: 데이터 병합 (`join_data.py`)
- **로직**: `parent_asin`을 키(Key)로 사용하여 수백만 건의 리뷰와 상품 정보를 하나로 합칩니다.
- **결과**: `joined_reviews.jsonl.gz` 생성. 메모리 효율을 위해 스트리밍 방식으로 처리됩니다.

### Step 2: 고품질 학습셋 추출 (`prepare_dataset.py`)
- **필터링 조건**: 리뷰의 신뢰도를 높이기 위해 '도움됨(helpful_vote)' 점수가 **5점 이상**인 리뷰만 추출합니다.
- **특징 추출(Feature Engineering)**:
    - **Sub-category**: 제목 키워드를 분석하여 카테고리를 자동 분류합니다.
    - **Price Cleaning**: 정규표현식을 통해 치환/정제 및 스케일링을 수행합니다. (브랜드 정보는 노이즈 최소화를 위해 제외)
    - **Input Text 생성**: RoBERTa 입력을 위해 [상품명 + 리뷰 제목 + 리뷰 본문 + 상세 스펙]을 결합합니다.
- **결과**: `fashion_train_subset_2.csv`.

### Step 3: 이미지 다운로드 파이프라인 (`download_fashion_images.py`)
- **연결 방식**: CSV의 `parent_asin`을 사용하여 `meta_Amazon_Fashion.jsonl`에서 해당 상품의 `hi_res(고해상도)` 또는 `large(대형)` 이미지 URL을 찾습니다.
- **작동 방식**: 1.4GB 메타데이터를 메모리에 다 올리지 않고 한 줄씩 읽는 **Streaming 검색** 방식을 사용.

---

## 3. 멀티모달 모델 실행 가이드

### 환경별 설정
1.  **로컬 스케일업 (Full Scale)**: `multimodal_model_full.py` 사용. 전체 데이터 3.8만 건을 학습하며 최신 튜닝 로직이 포함되어 있습니다.
2.  **코랩 샘플 테스트 (Colab)**: `multimodal_model_colab.py` 사용. 100건 샘플 테스트 및 구조 검증용.

### 필수 라이브러리 설치
```bash
pip install torch torchvision transformers scikit-learn pandas tqdm numpy pillow
```

### 아키텍처 개요 (3-way 하이브리드 결합 모델)
단순한 特性 이어붙이기(Concatenation)가 아닌 Gated Fusion 방식을 통해 모달리티 별 비중을 모델이 스스로 판단합니다.
- **Text Encoder**: `roberta-base` (사용자 경험, 텍스트)
- **Vision Encoder**: `EfficientNet-B0` (상품의 시각적 디테일 및 텍스쳐, 이미지넷의 구조 파악 활용)
- **Tabular Encoder**: 가격 수치(Float) + 카테고리 임베딩
- **Fusion (3-way GMU)**: 각 정보를 입력받아 Softmax 기반으로 가중치를 결정, 가장 유의미한 채널의 정보에 집중하여 최종 특징 추출.
- **Prediction Head**: MLP 기반 회귀 분석. 최종 출력에 Sigmoid를 활용하여 1~5점 사이로 스케일링하는 Bounded Output 기법 적용.

### 핵심 최적화 기법 (Tuning)
모델의 일반화 성능과 학습 안정성을 높이기 위해 다음의 기법이 적용되었습니다.
- **Modality Dropout (15%)**: 학습 중 무작위로 텍스트, 이미지, 또는 메타데이터 채널 중 하나를 0으로 마스킹하여 모델이 특정 모달리티에 편향되지 않도록 방지.
- **Weighted MSE Loss**: 데이터가 많은 5점에 비해 수가 적은 1~3점 리뷰에 가중치를 부여하여 데이터 불균형 문제 완화.
- **Differential Learning Rates**: 사전학습 모델(RoBERTa, EfficientNet)은 1e-5로 낮게, 신규 레이어(GMU, 회귀 모듈)는 1e-4로 설정하여 미세조정(Fine-Tuning) 효율 극대화.
- **Gradient Accumulation & Cosine Annealing**: 로컬 메모리 한계를 극복하기 위해 그래디언트를 누적(가상 배치 크기 16)시키고, 부드러운 수렴을 위해 코사인 스케줄러 적용.

---

## 4. 추론 (Inference)
학습이 완료된 가중치(`best_multimodal_model.pth`)를 통해 실시간으로 단일 상품 정보에 대한 평점을 예측할 수 있습니다.
```bash
python inference.py
```
입력된 텍스트, 이미지 경로, 가격, 카테고리를 분석하여 예측 평점과 각 모달리티의 활용 비중(Gate Weights)을 제공합니다.

---

## 4. 향후 개발을 위한 AI 프롬프트 (Development Prompt)

새로운 개발자나 AI 어시스턴트가 이 프로젝트를 이어서 작업할 때 아래 프롬프트를 입력하면 프로젝트 맥락을 즉시 파악할 수 있습니다.

> **[AI Assistant Prompt]**
> "현재 프로젝트는 Amazon Fashion 데이터(약 3.8만 건)를 활용하여 이미지, 리뷰, 가격-카테고리를 결합하는 하이브리드 멀티모달 평점 예측 시스템입니다. 
> 1. 영상처리에 EfficientNet-B0(1280 dim), 텍스트에 RoBERTa(768 dim), 메타데이터에 Tabular-MLP를 적용하고 이를 3-way Gated Multimodal Unit (GMU)으로 동적 결합합니다. 
> 2. `subset_100_with_images.csv` 샘플 검증을 넘어 전체 데이터셋으로 스케일업되었으며, Bounded Output(1~5점), Modality Dropout(15%), Weighted Loss, Differential LR 등의 튜닝 기법이 모두 적용되어 있습니다.
> 3. 브랜드 데이터는 노이즈로 파악하여 제외하였으며, 학습된 모델로 실시간 예측이 가능한 `inference.py`가 구축되어 있습니다.
> 위 환경을 바탕으로 향후 서비스 배포 및 API 연동 작업을 진행해 주세요."

---

## 5. 주의 사항
- `joined_reviews.jsonl.gz`는 용량이 매우 크므로 보관에 주의하세요.
- 이미지 다운로드 시 네트워크 오류가 날 수 있으므로, 실패한 이미지는 멀티모달 코드 내에서 자동으로 0(검정 화면) 처리되도록 예외 처리가 되어 있습니다.
- EfficientNet의 경우 입력 이미지 전처리 시 `torchvision` 공식 매개변수가 필수이므로 Custom Transform을 직접 수정하지 마세요.
