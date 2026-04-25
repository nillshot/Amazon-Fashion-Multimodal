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
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import random

warnings.filterwarnings("ignore")

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
CSV_FILE = 'fashion_train_subset_2_with_images.csv'
IMAGE_DIR = 'images'

BATCH_SIZE = 4
EPOCHS = 5  # 시간 소요가 클 경우를 대비하여 우선 5로 설정
ACCUMULATION_STEPS = 4  # 실질적인 배치 크기를 16(4*4)으로 만들어줍니다.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 기기: {DEVICE}")

# ============================================================
# 2. Tokenizer / Image Transform 설정
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# 3. Dataset 클래스 정의
# ============================================================
class AmazonDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) TEXT 처리
        text = str(row["input_text"])
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # 2) IMAGE 처리
        img_path = str(row["image_path"]).replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            img = image_transform(img)
        except:
            img = torch.zeros(3, 224, 224)

        # 3) TABULAR 처리
        price = torch.tensor([row["price_clean"]], dtype=torch.float32)
        price_missing = torch.tensor([row["price_missing"]], dtype=torch.float32)
        category = torch.tensor(row["category_id"], dtype=torch.long)

        # 4) TARGET 처리
        target = torch.tensor(row["target"], dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img,
            "price": price,
            "price_missing": price_missing,
            "category_id": category,
            "target": target
        }

# ============================================================
# 4. Modality Dropout 및 GMU 정의
# ============================================================
class ModalityDropout(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, text_feat, image_feat, tabular_feat):
        if not self.training:
            return text_feat, image_feat, tabular_feat

        mask = torch.ones((text_feat.size(0), 3), device=text_feat.device)
        for i in range(text_feat.size(0)):
            if random.random() < self.p:
                idx = random.randint(0, 2)
                mask[i, idx] = 0

        return (text_feat * mask[:, 0].unsqueeze(1), 
                image_feat * mask[:, 1].unsqueeze(1), 
                tabular_feat * mask[:, 2].unsqueeze(1))

class ThreeWayGMU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text_feat, image_feat, tabular_feat):
        concat = torch.cat([text_feat, image_feat, tabular_feat], dim=1)
        gate_weights = self.softmax(self.gate(concat))

        w_text = gate_weights[:, 0].unsqueeze(1)
        w_image = gate_weights[:, 1].unsqueeze(1)
        w_tab = gate_weights[:, 2].unsqueeze(1)

        fused = (
            w_text * text_feat +
            w_image * image_feat +
            w_tab * tabular_feat
        )
        return fused, gate_weights

# ============================================================
# 5. 모델 아키텍처 정의 (Bounded Model)
# ============================================================
class MultimodalRatingModelBounded(nn.Module):
    def __init__(self, num_categories, hidden_dim=256):
        super().__init__()

        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_fc = nn.Linear(768, hidden_dim)

        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.image_encoder = efficientnet.features
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(1280, hidden_dim)

        self.category_emb = nn.Embedding(num_categories, 32)
        self.tabular_fc = nn.Sequential(
            nn.Linear(1 + 1 + 32, hidden_dim),
            nn.ReLU()
        )

        self.modality_dropout = ModalityDropout(p=0.15)
        self.gmu = ThreeWayGMU(hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, image, price, price_missing, category_id):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_fc(text_out.last_hidden_state[:, 0, :])

        img_feat = self.image_encoder(image)
        img_feat = self.image_pool(img_feat).view(image.size(0), -1)
        img_feat = self.image_fc(img_feat)

        cat_emb = self.category_emb(category_id)
        tab = torch.cat([price, price_missing, cat_emb], dim=1)
        tab_feat = self.tabular_fc(tab)

        text_feat, img_feat, tab_feat = self.modality_dropout(text_feat, img_feat, tab_feat)
        fused, gate = self.gmu(text_feat, img_feat, tab_feat)

        out = torch.sigmoid(self.regressor(fused)).squeeze(1) * 4 + 1
        return out, gate

# ============================================================
# 6. 학습 / 평가 및 손실 함수
# ============================================================
def weighted_mse_loss(pred, target):
    weight_map = {
        1.0: 4.0,
        2.0: 4.0,
        3.0: 3.0,
        4.0: 2.0,
        5.0: 1.0
    }
    
    # 1.0 ~ 5.0 범위를 벗어나는 값이 혹시라도 들어오면 반올림하여 매핑
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor(
        [weight_map[t.item()] for t in target_rounded],
        dtype=torch.float32,
        device=target.device
    )

    loss = (pred - target) ** 2
    loss = loss * weights
    return loss.mean()

def train_one_epoch_weighted(model, loader, optimizer, scheduler, device, accumulation_steps=4):
    model.train()
    total_loss = 0

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        price = batch["price"].to(device)
        price_missing = batch["price_missing"].to(device)
        category_id = batch["category_id"].to(device)
        target = batch["target"].to(device)

        pred, gate = model(
            input_ids,
            attention_mask,
            image,
            price,
            price_missing,
            category_id
        )

        loss = weighted_mse_loss(pred, target)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    gates = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            price = batch["price"].to(device)
            price_missing = batch["price_missing"].to(device)
            category_id = batch["category_id"].to(device)
            target = batch["target"].to(device)

            pred, gate = model(
                input_ids,
                attention_mask,
                image,
                price,
                price_missing,
                category_id
            )

            loss = weighted_mse_loss(pred, target)
            total_loss += loss.item()

            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            gates.extend(gate.cpu().numpy())

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    avg_gate = np.mean(gates, axis=0)

    return total_loss / len(loader), mse, mae, avg_gate, preds, targets

# ============================================================
# 7. 메인 실행 함수
# ============================================================
def main():
    if not os.path.exists(CSV_FILE):
        print(f"오류: {CSV_FILE} 파일을 찾을 수 없습니다.")
        print("전체 데이터셋 이미지 다운로드 스크립트를 먼저 실행해주세요.")
        return

    print("데이터 로드 중...")
    df = pd.read_csv(CSV_FILE)
    
    # 텍스트 결측치 처리 (이미지는 download_full 단계에서 전처리됨)
    df['input_text'] = df['input_text'].fillna("No review provided")

    # Label Encoding for category
    label_encoder = LabelEncoder()
    df["category_id"] = label_encoder.fit_transform(df["sub_category"].astype(str))

    num_categories = df["category_id"].nunique()

    # Train / Validation Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = AmazonDataset(train_df.reset_index(drop=True))
    val_dataset = AmazonDataset(val_df.reset_index(drop=True))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    model = MultimodalRatingModelBounded(num_categories=num_categories).to(DEVICE)

    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "encoder" in n],
            "lr": 1e-5, # Pretrained 모델 (RoBERTa, EfficientNet)은 낮게
        },
        {
            "params": [p for n, p in model.named_parameters() if "encoder" not in n],
            "lr": 1e-4, # 신규 레이어 (GMU, Regressor)는 높게
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Cosine Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * (len(train_loader) // ACCUMULATION_STEPS))

    print("학습 시작...")
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch_weighted(model, train_loader, optimizer, scheduler, DEVICE, ACCUMULATION_STEPS)
        val_loss, val_mse, val_mae, avg_gate, preds, targets = evaluate(model, val_loader, DEVICE)

        print(f"\n[Weighted] Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"Validation MSE: {val_mse:.4f} | Validation MAE: {val_mae:.4f}")
        print(f"Gate → Text: {avg_gate[0]:.4f}, Image: {avg_gate[1]:.4f}, Tabular: {avg_gate[2]:.4f}")
        print("-" * 50)
        
        # Best 모델 저장
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print(f"Best model saved with MAE: {best_mae:.4f}")

    print("학습 완료! 일부 샘플 예측 결과:")
    for i in range(min(10, len(targets))):
        print(f"실제 평점: {targets[i]:.1f} / 예측 평점: {preds[i]:.2f}")

if __name__ == "__main__":
    main()
