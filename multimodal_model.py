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
# 1. 하이퍼파라미터 및 설정
# ==========================================
CSV_FILE = 'subset_100_with_images.csv'
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
        
        # 타겟 값 (평점)
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
        
        # 2. 이미지 처리 (torchvision transforms 사용)
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.image_transform(image)
        except Exception as e:
            # 이미지 로드 실패 시 검정 배경 대입 (예외 방지)
            pixel_values = torch.zeros((3, 224, 224))
            
        # 3. 정형 특징 (Feature) 가져오기
        price = self.prices[idx]
        category = self.categories[idx]
        rating = self.ratings[idx]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values,
            'price': price,
            'category': category,
            'rating': rating
        }

# ==========================================
# 3. 멀티모달 모델 아키텍처 정의 (3-way GMU)
# ==========================================
class GatedMultimodalUnit(nn.Module):
    """
    3-way Gated Multimodal Unit (GMU)
    텍스트, 이미지, 메타데이터(가격, 카테고리)의 중요도를 동적으로 결정하여 결합합니다.
    """
    def __init__(self, text_dim, image_dim, tabular_dim, hidden_dim=512):
        super(GatedMultimodalUnit, self).__init__()
        
        # 각 모달리티를 동일한 hidden_dim으로 투영
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)
        
        # Gate 계산을 위한 레이어 (각 채널별 가중치)
        self.gate_layer = nn.Linear(text_dim + image_dim + tabular_dim, 3)

    def forward(self, text_emb, image_emb, tabular_emb):
        # 1. 공통 차원으로 매핑 (Tanh 활성화)
        h_text = torch.tanh(self.text_proj(text_emb))
        h_image = torch.tanh(self.image_proj(image_emb))
        h_tabular = torch.tanh(self.tabular_proj(tabular_emb))
        
        # 2. Gate 값 계산 (가중치) - 원래 특성들을 합쳐서 각 채널의 중요도를 판단
        concat_emb = torch.cat([text_emb, image_emb, tabular_emb], dim=1)
        gate_weights = torch.softmax(self.gate_layer(concat_emb), dim=1) # [Batch, 3]
        
        g_text = gate_weights[:, 0].unsqueeze(1)
        g_image = gate_weights[:, 1].unsqueeze(1)
        g_tabular = gate_weights[:, 2].unsqueeze(1)
        
        # 3. 가중합 (Weighted Sum) 적용
        fused_emb = (g_text * h_text) + (g_image * h_image) + (g_tabular * h_tabular)
        
        return fused_emb, gate_weights # 가중치(gate_weights)도 반환하여 나중에 분석 가능하도록 함

class FashionFusionModel(nn.Module):
    def __init__(self, num_categories, hidden_dim=512):
        super(FashionFusionModel, self).__init__()
        
        # 1. 텍스트 인코더 (RoBERTa)
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        text_dim = self.text_encoder.config.hidden_size # 768
        
        # 2. 이미지 인코더 (EfficientNet-B0)
        # torchvision의 EfficientNet_B0 사용, 분류기(classifier) 제거 후 특징(1280차원)만 추출
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # 특징 추출기(features)와 풀링(avgpool)을 묶어서 인코더로 사용
        self.image_encoder = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,
            nn.Flatten()
        )
        image_dim = 1280 # EfficientNet-B0의 feature_dim
        
        # 3. 정형 데이터 임베딩 (카테고리, 가격) - 브랜드 제거됨
        self.category_embedding = nn.Embedding(num_categories, 16)
        tabular_dim = 1 + 16 # 가격(1) + 카테고리(16)
        
        # 4. 결합(Fusion) 레이어 (3-way GMU)
        self.gmu = GatedMultimodalUnit(text_dim, image_dim, tabular_dim, hidden_dim)
        
        # 5. 평점을 예측하는 회귀(Regression) 파트
        self.prediction_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 최종 1개의 값 (평점) 출력
        )

    def forward(self, input_ids, attention_mask, pixel_values, price, category):
        # 텍스트 임베딩 추출 ([CLS] 토큰 사용)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_outputs.pooler_output
        
        # 이미지 임베딩 추출
        image_emb = self.image_encoder(pixel_values)
        
        # 정형 특징(Feature) 통합
        cat_emb = self.category_embedding(category)
        price = price.unsqueeze(1) # [Batch Size, 1]
        tabular_emb = torch.cat([price, cat_emb], dim=1) # 17 차원
        
        # 3-way GMU 융합
        fused_emb, gate_weights = self.gmu(text_emb, image_emb, tabular_emb)
        
        # 최종 평점 예측
        output = self.prediction_head(fused_emb)
        
        return output.squeeze(), gate_weights

# ==========================================
# 4. 학습 루프 (메인 실행 파트)
# ==========================================
def main():
    print("데이터 로딩 중...")
    df = pd.read_csv(CSV_FILE)
    
    # 누락된 이미지 행 제거
    df = df.dropna(subset=['image_path'])
    
    # 텍스트 결측치 처리
    df['input_text'] = df['input_text'].fillna("No review provided")
    
    # --- 추가 특징(Feature) 전처리 (카테고리, 가격) ---
    le_cat = LabelEncoder()
    df['sub_category_encoded'] = le_cat.fit_transform(df['sub_category'].astype(str))
    
    # 연속형 데이터(가격) 스케일링 설정
    scaler = StandardScaler()
    prices = df['price_clean'].fillna(df['price_clean'].mean()).values.reshape(-1, 1)
    df['price_clean'] = scaler.fit_transform(prices)
    
    num_categories = len(le_cat.classes_)
    # ---------------------------------

    print("모델 초기화 중 (RoBERTa & EfficientNet 다운로드 시 시간 소요)...")
    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # torchvision official EfficientNet-B0 Transforms 적용
    image_transform = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    
    model = FashionFusionModel(num_categories=num_categories)
    model.to(DEVICE)
    
    # Train / Val Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = FashionMultimodalDataset(train_df, text_tokenizer, image_transform)
    val_dataset = FashionMultimodalDataset(val_df, text_tokenizer, image_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 손실 함수 (MSE, 회귀 분석용) 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # 가중치 분석을 위한 변수
        epoch_gate_text_sum = 0
        epoch_gate_image_sum = 0
        epoch_gate_tabular_sum = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values = batch['pixel_values'].to(DEVICE)
            price = batch['price'].to(DEVICE)
            category = batch['category'].to(DEVICE)
            rating = batch['rating'].to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs, gate_weights = model(input_ids, attention_mask, pixel_values, price, category)
            loss = criterion(outputs.view(-1), rating.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # (디버깅) GMU gate 합계 (3 모달리티의 평균 중요도 계산용)
            epoch_gate_text_sum += gate_weights[:, 0].sum().item()
            epoch_gate_image_sum += gate_weights[:, 1].sum().item()
            epoch_gate_tabular_sum += gate_weights[:, 2].sum().item()

            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        total_samples = len(train_dataset)
        avg_g_text = epoch_gate_text_sum / total_samples
        avg_g_image = epoch_gate_image_sum / total_samples
        avg_g_tabular = epoch_gate_tabular_sum / total_samples
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values = batch['pixel_values'].to(DEVICE)
                price = batch['price'].to(DEVICE)
                category = batch['category'].to(DEVICE)
                rating = batch['rating'].to(DEVICE)
                
                outputs, _ = model(input_ids, attention_mask, pixel_values, price, category)
                loss = criterion(outputs.view(-1), rating.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} 완료 | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        print(f" ㄴ GMU 평균 가중치: Text({avg_g_text:.2f}), Image({avg_g_image:.2f}), Meta({avg_g_tabular:.2f})")
        
    print("학습이 완료되었습니다! (모델 저장 준비 코드 필요 시 추가 가능)")

if __name__ == "__main__":
    main()
