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
PHASE_1_EPOCHS = 2 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ==========================================
# 2. Tokenizer & Image Transform
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
v3_weights = models.MobileNet_V3_Large_Weights.DEFAULT
base_transform = v3_weights.transforms()

train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(),
    base_transform
])

val_transform = base_transform

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class AmazonFashionV8Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else val_transform

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
# 4. CCR & CCS Loss 클래스 (V8: Hard Negative)
# ==========================================
class ContrastiveAttentionLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin
        )

    def compute_ccr(self, query, keys, attn_weights, k=5):
        # query: [B, D] (Text Pooled)
        # keys: [B, Patch, D] (Image sequence)
        dim = keys.size(-1)
        _, indices = torch.sort(attn_weights, dim=-1, descending=True)
        pos_idx, neg_idx = indices[:, :k], indices[:, -k:]
        
        pos_content = torch.gather(keys, 1, pos_idx.unsqueeze(-1).expand(-1,-1,dim)).mean(1)
        neg_content = torch.gather(keys, 1, neg_idx.unsqueeze(-1).expand(-1,-1,dim)).mean(1)
        
        return self.triplet_loss(query, pos_content, neg_content)

    def compute_ccs_hard_negative(self, query, attended_info):
        """ V8: 배치 내에서 가장 헷갈리는(유사도가 높은) 오답을 찾아 Negative로 활용 """
        B = query.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=query.device, requires_grad=True)
            
        # 1. 유사도 행렬 계산 (B x B)
        sim_matrix = F.cosine_similarity(query.unsqueeze(1), attended_info.unsqueeze(0), dim=-1)
        
        # 2. 자기 자신(정답) 제외
        mask = torch.eye(B, device=query.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)
        
        # 3. 가장 헷갈리는 샘플(Hard Negative) 인덱스 추출
        hard_negative_indices = sim_matrix.argmax(dim=-1)
        hard_negatives = attended_info[hard_negative_indices]
        
        # 4. Triplet Loss (Anchor, Positive, Hard Negative)
        return self.triplet_loss(query, attended_info, hard_negatives)

# ==========================================
# 5. 모델 아키텍처 (V8: Multi-Layer Cross-Attention)
# ==========================================
class IntraModalitySelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, padding_mask=None):
        attn_out, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=padding_mask)
        return self.norm(x + attn_out)

class CrossAttentionBlock(nn.Module):
    """ V8: 반복 가능한 단일 Cross-Attention 계층 """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.text_img_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.img_text_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.ffn_i = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        
        self.norm1_t = nn.LayerNorm(dim)
        self.norm2_t = nn.LayerNorm(dim)
        self.norm1_i = nn.LayerNorm(dim)
        self.norm2_i = nn.LayerNorm(dim)

    def forward(self, t_seq, i_seq, t_pad_mask=None):
        # 텍스트가 이미지를 봄
        t_attended, t_attn_weights = self.text_img_attn(query=t_seq, key=i_seq, value=i_seq, need_weights=True)
        t_seq = self.norm1_t(t_seq + t_attended)
        t_seq = self.norm2_t(t_seq + self.ffn_t(t_seq))
        
        # 이미지가 텍스트를 봄
        i_attended, _ = self.img_text_attn(query=i_seq, key=t_seq, value=t_seq, key_padding_mask=t_pad_mask)
        i_seq = self.norm1_i(i_seq + i_attended)
        i_seq = self.norm2_i(i_seq + self.ffn_i(i_seq))
        
        return t_seq, i_seq, t_attn_weights, t_attended

class InterModalityCrossAttentionV8(nn.Module):
    """ V8: Multi-Layer 구조 도입 """
    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim), 
            nn.ReLU(), 
            nn.Linear(dim, 3), 
            nn.Softmax(dim=1)
        )
        
        self.fc_fused = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, t_seq, i_seq, tab, t_pad_mask=None):
        t_out, i_out = t_seq, i_seq
        
        final_t_attn_weights = None
        final_t_attended = None
        
        # 설정한 층(layer) 수만큼 Cross-Attention 반복
        for layer in self.layers:
            t_out, i_out, t_attn_weights, t_attended = layer(t_out, i_out, t_pad_mask)
            final_t_attn_weights = t_attn_weights
            final_t_attended = t_attended
            
        # 시퀀스 차원 압축 (Pooling)
        if t_pad_mask is not None:
            mask = (~t_pad_mask).unsqueeze(-1).float()
            t_pooled = (t_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            t_attended_pooled = (final_t_attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            avg_t_attn_weights = (final_t_attn_weights * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            t_pooled = t_out.mean(dim=1)
            t_attended_pooled = final_t_attended.mean(dim=1)
            avg_t_attn_weights = final_t_attn_weights.mean(dim=1)
            
        i_pooled = i_out.mean(dim=1)
        
        concat_feat = torch.cat([t_pooled, i_pooled, tab], dim=1)
        weights = self.gate(concat_feat)
        
        weighted_sum = weights[:, 0].unsqueeze(1)*t_pooled + weights[:, 1].unsqueeze(1)*i_pooled + weights[:, 2].unsqueeze(1)*tab
        fused = self.fc_fused(torch.cat([weighted_sum, t_pooled, i_pooled, tab], dim=1))
        
        return fused, t_pooled, i_pooled, weights, avg_t_attn_weights, t_attended_pooled

class MultitaskFashionModelV8(nn.Module):
    def __init__(self, num_cat, hidden_dim=256):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_proj = nn.Linear(768, hidden_dim)
        
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.image_encoder = mobilenet.features
        self.image_proj = nn.Linear(960, hidden_dim) 
        
        self.cat_emb = nn.Embedding(num_cat, 32)
        self.tab_fc = nn.Sequential(nn.Linear(1 + 1 + 32, hidden_dim), nn.ReLU())
        
        self.text_self_attn = IntraModalitySelfAttention(hidden_dim)
        self.image_self_attn = IntraModalitySelfAttention(hidden_dim)
        
        # V8: Multi-layer Cross Attention 적용 (기본 2층)
        self.fusion = InterModalityCrossAttentionV8(hidden_dim, num_layers=2)
        
        self.fused_regressor = nn.Linear(hidden_dim, 1)
        self.text_regressor = nn.Linear(hidden_dim, 1)
        self.image_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, ids, mask, pixels, price, miss, cat):
        t_outputs = self.text_encoder(ids, attention_mask=mask)
        t_seq = self.text_proj(t_outputs.last_hidden_state)
        t_pad_mask = (mask == 0)
        
        img_feat = self.image_encoder(pixels)
        B, C, H, W = img_feat.shape
        i_seq = img_feat.view(B, C, H * W).transpose(1, 2)
        i_seq = self.image_proj(i_seq)
        
        tab_feat = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))
        
        t_seq_self = self.text_self_attn(t_seq, padding_mask=t_pad_mask)
        i_seq_self = self.image_self_attn(i_seq)
        
        fused, t_pooled, i_pooled, gates, avg_t_attn_weights, t_attended_pooled = self.fusion(
            t_seq_self, i_seq_self, tab_feat, t_pad_mask
        )
        
        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze() * 4 + 1
        out_text = torch.sigmoid(self.text_regressor(t_pooled)).squeeze() * 4 + 1
        out_image = torch.sigmoid(self.image_regressor(i_pooled)).squeeze() * 4 + 1
        
        return out_fused, out_text, out_image, gates, t_pooled, i_seq_self, avg_t_attn_weights, t_attended_pooled

# ==========================================
# 6. 유틸리티 (학습 및 평가)
# ==========================================
def weighted_mse_loss(pred, target):
    weight_map = {1.0: 4.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0}
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor([weight_map[t.item()] for t in target_rounded], device=target.device)
    return (weights * (pred - target)**2).mean()

def train_epoch(model, loader, optimizer, scheduler, device, acc_steps, epoch):
    model.train()
    contrastive_criterion = ContrastiveAttentionLoss().to(device)
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    for i, batch in enumerate(pbar):
        target = batch["target"].to(device)
        
        out_fused, out_text, out_image, _, t_pooled, i_seq, avg_t_attn_w, t_attended_pooled = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device), 
            batch["pixel_values"].to(device), batch["price"].to(device), 
            batch["price_missing"].to(device), batch["category_id"].to(device)
        )
        
        loss_fused = weighted_mse_loss(out_fused, target)
        loss_text = weighted_mse_loss(out_text, target)
        loss_image = weighted_mse_loss(out_image, target)
        loss_task = loss_fused + 0.4 * loss_text + 0.4 * loss_image
        
        loss_ccr = contrastive_criterion.compute_ccr(t_pooled, i_seq, avg_t_attn_w)
        # V8: Hard Negative Mining 적용
        loss_ccs = contrastive_criterion.compute_ccs_hard_negative(t_pooled, t_attended_pooled)
        
        loss = (loss_task + 0.1 * loss_ccr + 0.1 * loss_ccs) / acc_steps
        loss.backward()
        
        if (i+1) % acc_steps == 0 or (i+1) == len(loader):
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * acc_steps
        pbar.set_postfix({'loss': loss.item() * acc_steps, 'ccr': loss_ccr.item(), 'ccs': loss_ccs.item()})
    return total_loss / len(loader)

def evaluate(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    preds, targets, gates = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} Eval"):
            target = batch["target"].to(device)
            out_fused, out_text, out_image, gate, _, _, _, _ = model(
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
# 7. 메인 실행부
# ==========================================
def main():
    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(AmazonFashionV8Dataset(train_df, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AmazonFashionV8Dataset(val_df, transform=val_transform), batch_size=BATCH_SIZE)
    
    model = MultitaskFashionModelV8(num_cat=df["category_id"].nunique()).to(DEVICE)
    best_mae = float('inf')

    print("\n=======================================================")
    print(" 🚀 V8 Training: Multi-Layer Cross-Attention + Hard Negative CCS")
    print("=======================================================")

    print("\n--- [Phase 1] Text Encoder Frozen ---")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
    optimizer_p1 = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": 1e-4}
    ])
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=PHASE_1_EPOCHS * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(1, PHASE_1_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p1, scheduler_p1, DEVICE, ACCUMULATION_STEPS, epoch)
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch)
        
        print(f"\n[Phase 1 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"Fusion Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")

    print("\n--- [Phase 2] Text Encoder Unfrozen ---")
    for param in model.text_encoder.parameters():
        param.requires_grad = True

    optimizer_p2 = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": 5e-6},
        {"params": model.image_encoder.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n], "lr": 1e-4}
    ])
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=(EPOCHS - PHASE_1_EPOCHS) * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(PHASE_1_EPOCHS + 1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p2, scheduler_p2, DEVICE, ACCUMULATION_STEPS, epoch)
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch)
        
        print(f"\n[Phase 2 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"Fusion Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_mobile_version_v8_model.pth")
            print(f"🌟 New Best Mobile Version V8 Model Saved (MAE: {best_mae:.4f})\n")

if __name__ == "__main__":
    main()
