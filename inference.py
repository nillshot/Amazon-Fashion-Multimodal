import os
import torch
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torchvision import models
from transformers import RobertaModel

# multimodal_model_full에서 선언한 모델 클래스 구조를 그대로 가져오거나 임포트해야 합니다.
# 파일이 크지 않으므로 여기서는 독립 실행 가능하도록 클래스를 직접 선언/임포트합니다.
from multimodal_model_full import MultimodalRatingModelBounded

# ==========================================
# 1. 환경 및 설정
# ==========================================
MODEL_PATH = 'best_multimodal_model.pth'
CSV_FILE = 'fashion_train_subset_2_with_images.csv' # 카테고리 인코딩 기준을 맞추기 위함
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"추론 환경 초기화 중... (기기: {DEVICE})")

# ==========================================
# 2. 전처리 도구 로드
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 학습 당시 사용했던 카테고리를 기준으로 LabelEncoder 초기화
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["sub_category"].astype(str))
    num_categories = len(label_encoder.classes_)
else:
    print(f"경고: {CSV_FILE}이 없습니다. 카테고리 인코딩이 부정확할 수 있습니다.")
    num_categories = 100 # 임시 더미 값

# ==========================================
# 3. 모델 로드
# ==========================================
model = MultimodalRatingModelBounded(num_categories=num_categories).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("성공적으로 학습된 모델 가중치를 로드했습니다.")
else:
    print(f"경고: {MODEL_PATH} 파일이 없습니다. 랜덤 가중치로 추론을 진행합니다.")
    model.eval()

# ==========================================
# 4. 단일 샘플 추론 함수
# ==========================================
def predict_rating(text, image_path, price, sub_category):
    """
    단일 상품 정보를 바탕으로 평점을 예측합니다.
    """
    # 1) TEXT 처리
    if pd.isna(text) or text == "":
        text = "No review provided"
        
    enc = tokenizer(
        str(text),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # 2) IMAGE 처리
    try:
        img = Image.open(image_path).convert("RGB")
        img = image_transform(img).unsqueeze(0).to(DEVICE)
    except:
        print(f"경고: 이미지를 불러오지 못했습니다. ({image_path})")
        img = torch.zeros(1, 3, 224, 224).to(DEVICE)

    # 3) TABULAR 처리
    # price_missing 처리
    if pd.isna(price) or price == 0:
        p_val = 0.0
        p_miss = 1.0
    else:
        # 실제로는 StandardScaler로 스케일링을 해야 정확하지만, 여기선 임시 적용
        p_val = float(price)
        p_miss = 0.0

    p_tensor = torch.tensor([[p_val]], dtype=torch.float32).to(DEVICE)
    p_miss_tensor = torch.tensor([[p_miss]], dtype=torch.float32).to(DEVICE)

    # Category 처리
    cat_str = str(sub_category)
    if cat_str in label_encoder.classes_:
        cat_id = label_encoder.transform([cat_str])[0]
    else:
        # Unseen Category
        cat_id = 0

    cat_tensor = torch.tensor([cat_id], dtype=torch.long).to(DEVICE)

    # 4) 추론 수행
    with torch.no_grad():
        pred, gate = model(
            input_ids,
            attention_mask,
            img,
            p_tensor,
            p_miss_tensor,
            cat_tensor
        )
    
    predicted_score = pred.item()
    gate_weights = gate.cpu().numpy()[0]
    
    return predicted_score, gate_weights

# ==========================================
# 5. 테스트 실행
# ==========================================
if __name__ == "__main__":
    print("\n--- 샘플 데이터 추론 테스트 ---")
    sample_text = "The quality is amazing and it fits perfectly! Highly recommended."
    sample_image = "images/sample_dummy.jpg" # 실제 파일 경로로 대체
    sample_price = 25.99
    sample_category = "Dress"
    
    score, gates = predict_rating(sample_text, sample_image, sample_price, sample_category)
    
    print(f"입력 텍스트: '{sample_text}'")
    print(f"입력 카테고리: {sample_category} / 가격: ${sample_price}")
    print(f"예측 평점: {score:.2f} / 5.0")
    print(f"모달리티 가중치 활용도 - [텍스트: {gates[0]*100:.1f}%, 이미지: {gates[1]*100:.1f}%, 정보: {gates[2]*100:.1f}%]")
