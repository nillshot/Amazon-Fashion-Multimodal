# ==========================================
# Google Colab 전용 멀티모달 학습 스크립트
# ==========================================
# 실행 전 참고:
# 1. 코랩 첫 번째 셀에서 아래 명령어를 실행하여 라이브러리를 설치하세요:
#    !pip install transformers tqdm torchvision torch
# 2. 아래 USE_GOOGLE_DRIVE 변수를 설정하세요.
#    - 직접 파일 업로드 시: False
#    - 구글 드라이브 사용 시: True
# ==========================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, RobertaModel
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
# [설정] 구글 드라이브를 사용할지 여부를 선택하세요 (True 또는 False)
USE_GOOGLE_DRIVE = False

if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("구글 드라이브가 마운트되었습니다.")
        # 구글 드라이브 내 데이터 폴더 경로로 수정해주세요
        BASE_DIR = '/content/drive/MyDrive/BigData' 
    except:
        print("드라이브 마운트 실패. 로컬 경로를 사용합니다.")
        BASE_DIR = '.'
else:
    print("직접 파일 업로드 방식을 사용합니다. (현재 디렉토리 기준)")
    BASE_DIR = '/content'  # 코랩 세션에 직접 업로드 시 기본 경로 (또는 '.')

# 파일명만 올바르게 업로드 되었다고 가정하고 경로 구성
CSV_FILE = os.path.join(BASE_DIR, 'fashion_train_subset_2_with_images.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 기기: {DEVICE}")

# ==========================================
# 2. 데이터셋 및 전처리 클래스
# ==========================================
class FashionMultimodalDataset(Dataset):
    def __init__(self, df, text_tokenizer, image_transform, max_len=128):
        self.df = df.reset_index(drop=True)
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

        # 정형 데이터(특징) 정리 (결측치 처리) - 브랜드 제외
        self.prices = torch.tensor(self.df['price_clean'].fillna(self.df['price_clean'].mean()).values, dtype=torch.float32)
        self.categories = torch.tensor(self.df['sub_category_encoded'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df['prod_avg_rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 텍스트 처리
        text = str(row['input_text'])
        encoding = self.text_tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. 이미지 처리 (경로 보정 및 torchvision transforms 사용)
        original_img_path = row['image_path']
        img_filename = os.path.basename(original_img_path)
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.image_transform(image)
        except Exception as e:
            pixel_values = torch.zeros((3, 224, 224))
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values,
            'price': self.prices[idx],
            'category': self.categories[idx],
            'rating': self.ratings[idx]
        }

# ==========================================
# 3. 모델 아키텍처 정의 (3-way GMU)
# ==========================================
class GatedMultimodalUnit(nn.Module):
    def __init__(self, text_dim, image_dim, tabular_dim, hidden_dim=512):
        super(GatedMultimodalUnit, self).__init__()
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)
        
        self.gate_layer = nn.Linear(text_dim + image_dim + tabular_dim, 3)

    def forward(self, text_emb, image_emb, tabular_emb):
        h_text = torch.tanh(self.text_proj(text_emb))
        h_image = torch.tanh(self.image_proj(image_emb))
        h_tabular = torch.tanh(self.tabular_proj(tabular_emb))
        
        concat_emb = torch.cat([text_emb, image_emb, tabular_emb], dim=1)
        gate_weights = torch.softmax(self.gate_layer(concat_emb), dim=1)
        
        g_text = gate_weights[:, 0].unsqueeze(1)
        g_image = gate_weights[:, 1].unsqueeze(1)
        g_tabular = gate_weights[:, 2].unsqueeze(1)
        
        fused_emb = (g_text * h_text) + (g_image * h_image) + (g_tabular * h_tabular)
        return fused_emb, gate_weights

class FashionFusionModel(nn.Module):
    def __init__(self, num_categories, hidden_dim=512):
        super(FashionFusionModel, self).__init__()
        
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        text_dim = self.text_encoder.config.hidden_size # 768
        
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,
            nn.Flatten()
        )
        image_dim = 1280 
        
        self.category_embedding = nn.Embedding(num_categories, 16)
        tabular_dim = 1 + 16 # 가격(1) + 카테고리(16)
        
        self.gmu = GatedMultimodalUnit(text_dim, image_dim, tabular_dim, hidden_dim)
        
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values, price, category):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_outputs.pooler_output
        
        image_emb = self.image_encoder(pixel_values)
        
        cat_emb = self.category_embedding(category)
        price = price.unsqueeze(1)
        tabular_emb = torch.cat([price, cat_emb], dim=1)
        
        fused_emb, gate_weights = self.gmu(text_emb, image_emb, tabular_emb)
        output = self.prediction_head(fused_emb)
        return output.squeeze(), gate_weights

# ==========================================
# 4. 학습 실행
# ==========================================
def main():
    if not os.path.exists(CSV_FILE):
        print(f"오류: {CSV_FILE} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    df = pd.read_csv(CSV_FILE).dropna(subset=['image_path'])
    df['input_text'] = df['input_text'].fillna("No review")
    
    le_cat = LabelEncoder()
    df['sub_category_encoded'] = le_cat.fit_transform(df['sub_category'].astype(str))
    
    scaler = StandardScaler()
    prices = df['price_clean'].fillna(df['price_clean'].mean()).values.reshape(-1, 1)
    df['price_clean'] = scaler.fit_transform(prices)
    
    num_categories = len(le_cat.classes_)

    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    image_transform = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    
    model = FashionFusionModel(num_categories=num_categories).to(DEVICE)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = DataLoader(FashionMultimodalDataset(train_df, text_tokenizer, image_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FashionMultimodalDataset(val_df, text_tokenizer, image_transform), batch_size=BATCH_SIZE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            outputs, _ = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), 
                               batch['pixel_values'].to(DEVICE), batch['price'].to(DEVICE), 
                               batch['category'].to(DEVICE))
            loss = criterion(outputs.view(-1), batch['rating'].to(DEVICE).view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} 완료 | Avg Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()
