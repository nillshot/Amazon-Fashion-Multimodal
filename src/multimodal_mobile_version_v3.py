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
import random
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
CSV_FILE = 'fashion_train_subset_2_with_images.csv'
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

EPOCHS = 5
PHASE_1_EPOCHS = 2 # 이미지 주도 학습 기간
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ==========================================
# 2. Tokenizer & Image Transform (Augmentation 추가)
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# MobileNet-V3-Large의 기본 Transform
v3_weights = models.MobileNet_V3_Large_Weights.DEFAULT
base_transform = v3_weights.transforms()

# V3 전략: 학습 시 강력한 증강(Augmentation) 추가 -> 모델이 이미지를 더 주의 깊게 보도록 강제
train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(), # 강력한 자동 증강
    transforms.RandomHorizontalFlip(),
    base_transform
])

val_transform = base_transform

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class AmazonFashionFullDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else val_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text
        text = str(row["input_text"])
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # Image
        img_path = str(row["image_path"]).replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(img)
        except:
            pixel_values = torch.zeros(3, 224, 224)

        # Tabular
        price = torch.tensor([row["price_clean"]], dtype=torch.float32)
        price_missing = torch.tensor([row["price_missing"]], dtype=torch.float32)
        category = torch.tensor(row["category_id"], dtype=torch.long)
        
        # Target
        target = torch.tensor(row["target"], dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "price": price,
            "price_missing": price_missing,
            "category_id": category,
            "target": target
        }

# ==========================================
# 4. 모델 아키텍처 (MobileNet-V3 기반)
# ==========================================
class TargetedModalityDropout(nn.Module):
    def __init__(self, text_drop_p=0.8, general_drop_p=0.2): 
        super().__init__()
        self.text_drop_p = text_drop_p
        self.general_drop_p = general_drop_p

    def forward(self, t, i, tab):
        if not self.training: return t, i, tab
        mask = torch.ones((t.size(0), 3), device=t.device)
        for idx in range(t.size(0)):
            if random.random() < self.text_drop_p:
                mask[idx, 0] = 0 # 텍스트 강력 차단
            elif random.random() < self.general_drop_p:
                mask[idx, random.randint(1, 2)] = 0
        return t * mask[:, 0].unsqueeze(1), i * mask[:, 1].unsqueeze(1), tab * mask[:, 2].unsqueeze(1)

class ThreeWayGMU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 3), nn.Softmax(dim=1))
    def forward(self, t, i, tab):
        weights = self.gate(torch.cat([t, i, tab], dim=1))
        fused = weights[:, 0].unsqueeze(1)*t + weights[:, 1].unsqueeze(1)*i + weights[:, 2].unsqueeze(1)*tab
        return fused, weights

class MultitaskFashionModelV3(nn.Module):
    def __init__(self, num_cat, hidden_dim=256):
        super().__init__()
        # Text Encoder
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_fc = nn.Linear(768, hidden_dim)
        
        # Image Encoder (Torchvision MobileNet-V3-Large 적용 - SE Block 포함으로 더 정교한 피처 추출)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.image_encoder = mobilenet.features
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(960, hidden_dim) # MobileNet-V3-Large의 출력 차원은 960
        
        # Tabular Encoder
        self.cat_emb = nn.Embedding(num_cat, 32)
        self.tab_fc = nn.Sequential(nn.Linear(1 + 1 + 32, hidden_dim), nn.ReLU())
        
        self.modality_dropout = TargetedModalityDropout(0.8, 0.2)
        self.gmu = ThreeWayGMU(hidden_dim)
        
        # Multi-task Regressors
        self.text_regressor = nn.Linear(hidden_dim, 1)
        self.image_regressor = nn.Linear(hidden_dim, 1)
        self.fused_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, ids, mask, pixels, price, miss, cat):
        # Text
        t_feat_raw = self.text_fc(self.text_encoder(ids, mask).pooler_output)
        
        # Image
        img_feat = self.image_encoder(pixels)
        img_feat = self.image_pool(img_feat).view(pixels.size(0), -1)
        i_feat_raw = self.image_fc(img_feat)
        
        # Tabular
        tab_feat_raw = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))
        
        # Individual Predictions
        out_text = torch.sigmoid(self.text_regressor(t_feat_raw)).squeeze() * 4 + 1
        out_image = torch.sigmoid(self.image_regressor(i_feat_raw)).squeeze() * 4 + 1
        
        # GMU Fusion & Final Prediction
        t_feat, i_feat, tab_feat = self.modality_dropout(t_feat_raw, i_feat_raw, tab_feat_raw)
        fused, gates = self.gmu(t_feat, i_feat, tab_feat)
        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze() * 4 + 1
        
        return out_fused, out_text, out_image, gates

# ==========================================
# 5. 유틸리티 (학습 및 평가)
# ==========================================
def weighted_mse_loss(pred, target):
    weight_map = {1.0: 4.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0}
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor([weight_map[t.item()] for t in target_rounded], device=target.device)
    return (weights * (pred - target)**2).mean()

def train_epoch(model, loader, optimizer, scheduler, device, acc_steps, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    for i, batch in enumerate(pbar):
        target = batch["target"].to(device)
        
        out_fused, out_text, out_image, _ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device), 
            batch["pixel_values"].to(device), batch["price"].to(device), 
            batch["price_missing"].to(device), batch["category_id"].to(device)
        )
        
        loss_fused = weighted_mse_loss(out_fused, target)
        loss_text = weighted_mse_loss(out_text, target)
        loss_image = weighted_mse_loss(out_image, target)
        
        # V3 전략: Image-Heavy Multi-task Loss (이미지 학습 비중 극대화)
        # 텍스트는 0.2로 낮추고, 이미지는 1.2로 대폭 상향
        loss = (1.0 * loss_fused + 0.2 * loss_text + 1.2 * loss_image) / acc_steps
        loss.backward()
        
        if (i+1) % acc_steps == 0 or (i+1) == len(loader):
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * acc_steps
        pbar.set_postfix({'loss': loss.item() * acc_steps})
    return total_loss / len(loader)

def evaluate(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    preds, targets, gates = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} Eval"):
            target = batch["target"].to(device)
            out_fused, out_text, out_image, gate = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), 
                batch["pixel_values"].to(device), batch["price"].to(device), 
                batch["price_missing"].to(device), batch["category_id"].to(device)
            )
            
            loss = weighted_mse_loss(out_fused, target)
            total_loss += loss.item()
            preds.extend(out_fused.cpu().numpy())
            targets.extend(batch["target"].numpy())
            gates.extend(gate.cpu().numpy())
            
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    avg_gate = np.mean(gates, axis=0)
    return total_loss / len(loader), mse, mae, avg_gate

# ==========================================
# 6. 메인 실행부 (Curriculum Learning 적용)
# ==========================================
def main():
    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # 학습셋에는 Augmentation이 포함된 transform 적용
    train_loader = DataLoader(AmazonFashionFullDataset(train_df, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AmazonFashionFullDataset(val_df, transform=val_transform), batch_size=BATCH_SIZE)
    
    model = MultitaskFashionModelV3(num_cat=df["category_id"].nunique()).to(DEVICE)
    best_mae = float('inf')

    print("\n=======================================================")
    print(" 🚀 V3 Training: Curriculum Learning (Image-First)")
    print("=======================================================")

    # [Phase 1] 텍스트 동결 (이미지 & 정형 데이터 주도 학습)
    print("\n--- [Phase 1] Text Encoder Frozen (Epoch 1 ~ 2) ---")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
    optimizer_p1 = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": 2e-5}, # 이미지 인코더는 적극적으로
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": 1e-4}
    ])
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=PHASE_1_EPOCHS * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(1, PHASE_1_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p1, scheduler_p1, DEVICE, ACCUMULATION_STEPS, epoch)
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch)
        
        print(f"\n[Phase 1 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")

    # [Phase 2] 텍스트 동결 해제 및 차등 학습률 적용
    print("\n--- [Phase 2] Text Encoder Unfrozen (Epoch 3 ~ 5) ---")
    for param in model.text_encoder.parameters():
        param.requires_grad = True

    # V2의 차등 학습률 재적용 (단, 텍스트는 1e-6으로 여전히 억제)
    optimizer_p2 = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": 1e-6},
        {"params": model.image_encoder.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n], "lr": 1e-4}
    ])
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=(EPOCHS - PHASE_1_EPOCHS) * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(PHASE_1_EPOCHS + 1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p2, scheduler_p2, DEVICE, ACCUMULATION_STEPS, epoch)
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch)
        
        print(f"\n[Phase 2 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_mobile_version_v3_model.pth")
            print(f"🌟 New Best Mobile Version V3 Model Saved (MAE: {best_mae:.4f})\n")

if __name__ == "__main__":
    main()
