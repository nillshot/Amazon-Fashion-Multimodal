import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

"""
Multimodal Mobile Version V5 — CCR & CCS Integration
=====================================================
이 버전은 V4의 Cross-Attention 구조를 계승하면서, 
논문 arXiv:2105.09597에서 제안된 CCR(대조적 콘텐츠 재확보) 및 
CCS(대조적 콘텐츠 교체) Loss를 추가하여 Attention의 정교함을 극대화했습니다.

주요 특징:
1. Backbone: RoBERTa-base & MobileNet-V2
2. Loss: Weighted MSE + CCR Loss + CCS Loss
3. Training: 2-Phase Curriculum Learning (Freeze/Unfreeze)
"""

warnings.filterwarnings("ignore")

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
CSV_FILE = 'fashion_train_subset_2_with_images.csv' # 기존 이미지 데이터셋 사용
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EPOCHS = 5
PHASE_1_EPOCHS = 2 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 512

# ==========================================
# 2. Tokenizer & Image Transform
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# MobileNet-V2용 표준 변환
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class AmazonFashionV5Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else base_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = str(row["input_text"])
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        img_path = str(row["image_path"]).replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(img)
        except:
            pixel_values = torch.zeros(3, 224, 224)

        price = torch.tensor([row.get("price_clean", 0.0)], dtype=torch.float32)
        price_missing = torch.tensor([row.get("price_missing", 1.0)], dtype=torch.float32)
        category = torch.tensor(row.get("category_id", 0), dtype=torch.long)
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
# 4. CCR & CCS Loss 클래스
# ==========================================
class ContrastiveAttentionLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin
        )

    def compute_ccr(self, query, keys, attn_weights, k=5):
        # CCR: 높은 어텐션 영역은 Query와 더 가깝게, 낮은 영역은 멀게
        dim = keys.size(-1)
        _, indices = torch.sort(attn_weights, dim=-1, descending=True)
        pos_idx, neg_idx = indices[:, :k], indices[:, -k:]
        
        pos_content = torch.gather(keys, 1, pos_idx.unsqueeze(-1).expand(-1,-1,dim)).mean(1)
        neg_content = torch.gather(keys, 1, neg_idx.unsqueeze(-1).expand(-1,-1,dim)).mean(1)
        
        return self.triplet_loss(query, pos_content, neg_content)

    def compute_ccs(self, query, attended_info):
        # CCS: 이미지 정보는 가짜 텍스트보다 진짜 텍스트와 더 가깝게
        swapped_query = torch.roll(query, shifts=1, dims=0)
        return self.triplet_loss(attended_info, query, swapped_query)

# ==========================================
# 5. 모델 아키텍처 (V5)
# ==========================================
class CrossAttentionFusionV5(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 3), nn.Softmax(dim=1)
        )

    def forward(self, t, i_keys, tab):
        # t: [B, D], i_keys: [B, Patch, D]
        Q = self.q_proj(t).unsqueeze(1)
        K = self.k_proj(i_keys)
        V = self.v_proj(i_keys)
        
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attended_i = torch.bmm(attn_weights, V).squeeze(1)
        
        # Gates
        weights = self.gate(torch.cat([t, attended_i, tab], dim=1))
        fused = weights[:, 0].unsqueeze(1)*t + weights[:, 1].unsqueeze(1)*attended_i + weights[:, 2].unsqueeze(1)*tab
        
        return fused, weights, attn_weights.squeeze(1), attended_i

class MultitaskFashionModelV5(nn.Module):
    def __init__(self, num_cat, hidden_dim=EMBED_DIM):
        super().__init__()
        # Text Encoder: RoBERTa
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_fc = nn.Linear(768, hidden_dim)
        
        # Image Encoder: MobileNet-V2
        mobilenet = models.mobilenet_v2(pretrained=True).features
        self.image_encoder = mobilenet
        self.image_proj = nn.Linear(1280, hidden_dim)
        
        # Tabular
        self.cat_emb = nn.Embedding(num_cat, 32)
        self.tab_fc = nn.Sequential(nn.Linear(1 + 1 + 32, hidden_dim), nn.ReLU())
        
        # Fusion
        self.fusion = CrossAttentionFusionV5(hidden_dim)
        
        # Regressors
        self.fused_regressor = nn.Linear(hidden_dim, 1)
        self.text_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, ids, mask, pixels, price, miss, cat):
        # Text
        t_feat = self.text_fc(self.text_encoder(ids, mask).pooler_output)
        
        # Image (Patch-level features for CCR)
        img_features = self.image_encoder(pixels) # [B, 1280, 7, 7]
        img_patches = img_features.flatten(2).transpose(1, 2) # [B, 49, 1280]
        i_keys = self.image_proj(img_patches) # [B, 49, D]
        
        # Tabular
        tab_feat = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))
        
        # Fusion with Attention Weights for CCR/CCS
        fused, gates, attn_weights, attended_i = self.fusion(t_feat, i_keys, tab_feat)
        
        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze() * 4 + 1
        out_text = torch.sigmoid(self.text_regressor(t_feat)).squeeze() * 4 + 1
        
        return out_fused, out_text, t_feat, i_keys, attn_weights, attended_i, gates

# ==========================================
# 6. 학습 및 평가 루틴
# ==========================================
def weighted_mse_loss(pred, target):
    weight_map = {1.0: 4.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0}
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor([weight_map[t.item()] for t in target_rounded], device=target.device)
    return (weights * (pred - target)**2).mean()

def train_epoch(model, loader, optimizer, scheduler, device, acc_steps, epoch):
    model.train()
    contrastive_criterion = ContrastiveAttentionLoss()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    
    for i, batch in enumerate(pbar):
        target = batch["target"].to(device)
        out_fused, out_text, t_feat, i_keys, attn_w, attn_i, _ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device), 
            batch["pixel_values"].to(device), batch["price"].to(device), 
            batch["price_missing"].to(device), batch["category_id"].to(device)
        )
        
        # 1. Base Task Loss
        loss_task = (weighted_mse_loss(out_fused, target) + 0.4 * weighted_mse_loss(out_text, target))
        
        # 2. V5 Contrastive Loss (CCR & CCS)
        loss_ccr = contrastive_criterion.compute_ccr(t_feat, i_keys, attn_w)
        loss_ccs = contrastive_criterion.compute_ccs(t_feat, attn_i)
        
        # 3. Total Loss Integration
        loss = (loss_task + 0.1 * loss_ccr + 0.1 * loss_ccs) / acc_steps
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
    preds, targets, gates = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} Eval"):
            out_fused, _, _, _, _, _, gate = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), 
                batch["pixel_values"].to(device), batch["price"].to(device), 
                batch["price_missing"].to(device), batch["category_id"].to(device)
            )
            preds.extend(out_fused.cpu().numpy())
            targets.extend(batch["target"].numpy())
            gates.extend(gate.cpu().numpy())
            
    mae = mean_absolute_error(targets, preds)
    avg_gate = np.mean(gates, axis=0)
    return mae, avg_gate

# ==========================================
# 7. 메인 실행부
# ==========================================
def main():
    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(AmazonFashionV5Dataset(train_df, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AmazonFashionV5Dataset(val_df), batch_size=BATCH_SIZE)
    
    model = MultitaskFashionModelV5(num_cat=df["category_id"].nunique()).to(DEVICE)
    best_mae = float('inf')

    print("\n=======================================================")
    print(" 🚀 V5 Training: Cross-Attention + CCR + CCS Loss")
    print("=======================================================")

    # [Phase 1] 텍스트 동결
    for param in model.text_encoder.parameters(): param.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": 1e-4}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE_1_EPOCHS * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(1, PHASE_1_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, scheduler, DEVICE, ACCUMULATION_STEPS, epoch)
        mae, gate = evaluate(model, val_loader, DEVICE, epoch)
        print(f"Epoch {epoch} MAE: {mae:.4f} | Gates: T:{gate[0]:.2f} I:{gate[1]:.2f} Tab:{gate[2]:.2f}")

    # [Phase 2] 동결 해제
    for param in model.text_encoder.parameters(): param.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": 5e-6},
        {"params": model.image_encoder.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n], "lr": 1e-4}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=(EPOCHS - PHASE_1_EPOCHS) * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(PHASE_1_EPOCHS + 1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, scheduler, DEVICE, ACCUMULATION_STEPS, epoch)
        mae, gate = evaluate(model, val_loader, DEVICE, epoch)
        print(f"Epoch {epoch} MAE: {mae:.4f} | Gates: T:{gate[0]:.2f} I:{gate[1]:.2f} Tab:{gate[2]:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "best_mobile_version_v5_model.pth")
            print(f"🌟 New Best Model Saved! (MAE: {best_mae:.4f})")

if __name__ == "__main__":
    main()
