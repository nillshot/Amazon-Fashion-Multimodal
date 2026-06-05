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
CSV_FILE = 'fashion_train_subset_3_with_meta_text_v10.csv'  # V10 변경: seller_text, review_text가 분리된 CSV 사용 fashion_train_subset_3_with_meta_text_v10
BATCH_SIZE = 2  # V10 변경: RoBERTa를 seller/review에 2번 사용하므로 VRAM 절약
ACCUMULATION_STEPS = 4
EPOCHS = 10
PHASE_1_EPOCHS = 2 
REVIEW_DROPOUT_P = 0.30  # V12 변경: 리뷰 의존도 완화를 위한 Review Modality Dropout 확률
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
class AmazonFashionV9Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else val_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seller_text = str(row["seller_text"])  # V10 변경: 판매자 텍스트 분리 입력
        review_text = str(row["review_text"])  # V10 변경: 리뷰 텍스트 분리 입력

        seller_enc = tokenizer(  # V10 변경: seller_text 별도 토큰화
            seller_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        review_enc = tokenizer(  # V10 변경: review_text 별도 토큰화
            review_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
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
            "seller_input_ids": seller_enc["input_ids"].squeeze(0),  # V10 변경
            "seller_attention_mask": seller_enc["attention_mask"].squeeze(0),  # V10 변경
            "review_input_ids": review_enc["input_ids"].squeeze(0),  # V10 변경
            "review_attention_mask": review_enc["attention_mask"].squeeze(0),  # V10 변경
            "pixel_values": pixel_values,
            "price": price,
            "price_missing": price_missing,
            "category_id": category,
            "target": target
        }

# ==========================================
# 4. CCR & CCS Loss 클래스 (V9: Hard Negative)
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
        """ V9: 배치 내에서 가장 헷갈리는(유사도가 높은) 오답을 찾아 Negative로 활용 """
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
# 5. 모델 아키텍처 (V9: Multi-Layer Cross-Attention)
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
    """ V9: 반복 가능한 단일 Cross-Attention 계층 """
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

class InterModalityCrossAttentionV9(nn.Module):
    """ V12 변경: Concatenate-Attend-Split 기반 Multi-Layer Cross-Attention """
    def __init__(self, dim, num_heads=4, num_layers=2, review_dropout_p=0.30):  # V12 변경
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])

        self.review_dropout_p = review_dropout_p  # V12 변경: 리뷰 모달리티 드롭아웃 확률

        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),  # V10 유지: seller/review/image/tabular 4개 모달
            nn.ReLU(), 
            nn.Linear(dim, 4),  # V10 유지: 4-way gate
            nn.Softmax(dim=1)
        )

        self.fc_fused = nn.Sequential(
            nn.Linear(dim * 5, dim),  # V10 유지: weighted_sum + seller + review + image + tabular
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, seller_seq, review_seq, i_seq, tab, seller_pad_mask=None, review_pad_mask=None):  # V10 유지
        # V12 변경: seller_seq와 review_seq를 먼저 결합하여 이미지와 함께 Cross-Attention 수행
        # 기존 V10은 seller_seq만 이미지와 Cross-Attention을 수행하고 review_seq는 pooling만 수행했음
        seller_len = seller_seq.size(1)  # V12 변경: Split을 위해 seller 길이 저장
        review_len = review_seq.size(1)  # V12 변경: Split을 위해 review 길이 저장

        text_seq = torch.cat([seller_seq, review_seq], dim=1)  # V12 변경: Concatenate
        if seller_pad_mask is not None and review_pad_mask is not None:
            text_pad_mask = torch.cat([seller_pad_mask, review_pad_mask], dim=1)  # V12 변경
        else:
            text_pad_mask = None

        text_out, i_out = text_seq, i_seq

        final_t_attn_weights = None
        final_t_attended = None

        # V12 변경: 결합된 텍스트(seller+review)가 이미지와 Cross-Attention 수행
        for layer in self.layers:
            text_out, i_out, t_attn_weights, t_attended = layer(text_out, i_out, text_pad_mask)
            final_t_attn_weights = t_attn_weights
            final_t_attended = t_attended

        # V12 변경: Attend 이후 다시 seller/review로 분리
        seller_out = text_out[:, :seller_len, :]
        review_out = text_out[:, seller_len:seller_len + review_len, :]
        seller_attended = final_t_attended[:, :seller_len, :]

        # Seller Pooling
        if seller_pad_mask is not None:
            s_mask = (~seller_pad_mask).unsqueeze(-1).float()
            seller_pooled = (seller_out * s_mask).sum(dim=1) / s_mask.sum(dim=1).clamp(min=1e-9)
            t_attended_pooled = (seller_attended * s_mask).sum(dim=1) / s_mask.sum(dim=1).clamp(min=1e-9)
        else:
            seller_pooled = seller_out.mean(dim=1)
            t_attended_pooled = seller_attended.mean(dim=1)

        # Review Pooling
        if review_pad_mask is not None:
            r_mask = (~review_pad_mask).unsqueeze(-1).float()
            review_pooled = (review_out * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1e-9)
        else:
            review_pooled = review_out.mean(dim=1)

        # V12 변경: CCR 계산용 attention weight는 결합 텍스트 전체 기준으로 평균
        if text_pad_mask is not None:
            t_mask = (~text_pad_mask).unsqueeze(-1).float()
            avg_t_attn_weights = (final_t_attn_weights * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(min=1e-9)
        else:
            avg_t_attn_weights = final_t_attn_weights.mean(dim=1)

        i_pooled = i_out.mean(dim=1)

        # V12 변경: 학습 중 일부 샘플의 review_pooled를 0으로 만들어 리뷰 의존도 완화
        if self.training and self.review_dropout_p > 0:
            keep_mask = (torch.rand(review_pooled.size(0), 1, device=review_pooled.device) > self.review_dropout_p).float()
            review_pooled = review_pooled * keep_mask

        concat_feat = torch.cat([seller_pooled, review_pooled, i_pooled, tab], dim=1)
        weights = self.gate(concat_feat)

        weighted_sum = (
            weights[:, 0].unsqueeze(1) * seller_pooled +
            weights[:, 1].unsqueeze(1) * review_pooled +
            weights[:, 2].unsqueeze(1) * i_pooled +
            weights[:, 3].unsqueeze(1) * tab
        )
        fused = self.fc_fused(torch.cat([weighted_sum, seller_pooled, review_pooled, i_pooled, tab], dim=1))

        return fused, seller_pooled, review_pooled, i_pooled, weights, avg_t_attn_weights, t_attended_pooled  # V10 반환 구조 유지

class MultitaskFashionModelV9(nn.Module):
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
        self.review_self_attn = IntraModalitySelfAttention(hidden_dim)  # V10 변경: review_text 전용 self-attention 추가
        self.image_self_attn = IntraModalitySelfAttention(hidden_dim)
        
        # V9: Multi-layer Cross Attention 적용 (기본 2층)
        self.fusion = InterModalityCrossAttentionV9(hidden_dim, num_layers=2, review_dropout_p=REVIEW_DROPOUT_P)  # V12 변경
        
        self.fused_regressor = nn.Linear(hidden_dim, 1)
        self.seller_regressor = nn.Linear(hidden_dim, 1)  # V10 변경: 판매자 텍스트 보조 예측기
        self.review_regressor = nn.Linear(hidden_dim, 1)  # V10 변경: 리뷰 텍스트 보조 예측기
        self.image_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, seller_ids, seller_mask, review_ids, review_mask, pixels, price, miss, cat):  # V10 변경
        seller_outputs = self.text_encoder(seller_ids, attention_mask=seller_mask)  # V10 변경
        seller_seq = self.text_proj(seller_outputs.last_hidden_state)  # V10 변경
        seller_pad_mask = (seller_mask == 0)  # V10 변경

        review_outputs = self.text_encoder(review_ids, attention_mask=review_mask)  # V10 변경: 같은 RoBERTa 공유
        review_seq = self.text_proj(review_outputs.last_hidden_state)  # V10 변경
        review_pad_mask = (review_mask == 0)  # V10 변경
        
        img_feat = self.image_encoder(pixels)
        B, C, H, W = img_feat.shape
        i_seq = img_feat.view(B, C, H * W).transpose(1, 2)
        i_seq = self.image_proj(i_seq)
        
        tab_feat = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))
        
        seller_seq_self = self.text_self_attn(seller_seq, padding_mask=seller_pad_mask)  # V10 변경
        review_seq_self = self.review_self_attn(review_seq, padding_mask=review_pad_mask)  # V10 변경
        i_seq_self = self.image_self_attn(i_seq)
        
        fused, seller_pooled, review_pooled, i_pooled, gates, avg_t_attn_weights, t_attended_pooled = self.fusion(  # V10 변경
            seller_seq_self, review_seq_self, i_seq_self, tab_feat, seller_pad_mask, review_pad_mask
        )
        
        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze(-1) * 4 + 1  # V10 변경: squeeze(-1)로 안정화
        out_seller = torch.sigmoid(self.seller_regressor(seller_pooled)).squeeze(-1) * 4 + 1  # V10 변경
        out_review = torch.sigmoid(self.review_regressor(review_pooled)).squeeze(-1) * 4 + 1  # V10 변경
        out_image = torch.sigmoid(self.image_regressor(i_pooled)).squeeze(-1) * 4 + 1  # V10 변경
        
        return out_fused, out_seller, out_review, out_image, gates, seller_pooled, i_seq_self, avg_t_attn_weights, t_attended_pooled  # V10 변경

# ==========================================
# 6. 유틸리티 (학습 및 평가)
# ==========================================
def weighted_mse_loss(pred, target):
    weight_map = {1.0: 4.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0}
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor([weight_map[t.item()] for t in target_rounded], device=target.device)
    return (weights * (pred - target)**2).mean()

# V12 추가: Gate Entropy Loss
# 특정 모달리티 하나로 gate가 쏠리는 것을 완화
def gate_entropy_loss(weights):
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
    return -entropy

def train_epoch(model, loader, optimizer, scheduler, device, acc_steps, epoch):
    model.train()
    contrastive_criterion = ContrastiveAttentionLoss().to(device)
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    for i, batch in enumerate(pbar):
        target = batch["target"].to(device)
        
        out_fused, out_seller, out_review, out_image, gate_weights, seller_pooled, i_seq, avg_t_attn_w, t_attended_pooled = model(  # V10 변경
            batch["seller_input_ids"].to(device), batch["seller_attention_mask"].to(device),  # V10 변경
            batch["review_input_ids"].to(device), batch["review_attention_mask"].to(device),  # V10 변경
            batch["pixel_values"].to(device), batch["price"].to(device), 
            batch["price_missing"].to(device), batch["category_id"].to(device)
        )
        
        loss_fused = weighted_mse_loss(out_fused, target)
        loss_seller = weighted_mse_loss(out_seller, target)  # V10 변경
        loss_review = weighted_mse_loss(out_review, target)  # V10 변경
        loss_image = weighted_mse_loss(out_image, target)
        loss_task = loss_fused + 0.3 * loss_seller + 0.3 * loss_review + 0.3 * loss_image  # V12 변경: 보조 loss 균형화
        
        loss_ccr = contrastive_criterion.compute_ccr(seller_pooled, i_seq, avg_t_attn_w)  # V10 변경: seller_text와 image 정렬
        # V9: Hard Negative Mining 적용
        loss_ccs = contrastive_criterion.compute_ccs_hard_negative(seller_pooled, t_attended_pooled)  # V10 변경
        
        # loss = (loss_task + 0.1 * loss_ccr + 0.1 * loss_ccs) / acc_steps
        loss_gate_ent = gate_entropy_loss(gate_weights)

        loss = (
            loss_task
            + 0.1 * loss_ccr
            + 0.1 * loss_ccs
            + 0.01 * loss_gate_ent   # V12 추가: Gate Entropy Loss (λ=0.01, 0.02→0.01로 유연화)
        ) / acc_steps
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
            out_fused, out_seller, out_review, out_image, gate, _, _, _, _ = model(  # V10 변경
                batch["seller_input_ids"].to(device), batch["seller_attention_mask"].to(device),  # V10 변경
                batch["review_input_ids"].to(device), batch["review_attention_mask"].to(device),  # V10 변경
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
    os.makedirs("pth", exist_ok=True)  # V10 변경: pth 폴더 없을 때 자동 생성

    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE).fillna({"seller_text": "None", "review_text": "None"})  # V10 변경
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(AmazonFashionV9Dataset(train_df, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AmazonFashionV9Dataset(val_df, transform=val_transform), batch_size=BATCH_SIZE)
    
    model = MultitaskFashionModelV9(num_cat=df["category_id"].nunique()).to(DEVICE)
    best_mae = float('inf')

    print("\n=======================================================")
    print(" 🚀 V12 Training: Concatenate-Attend-Split + Review Dropout")  # V10 변경
    print("=======================================================")
    print(f"V12 Review Modality Dropout P = {REVIEW_DROPOUT_P}")  # V12 변경

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
        print(f"Fusion Gate -> Seller: {avg_gate[0]:.2f}, Review: {avg_gate[1]:.2f}, Image: {avg_gate[2]:.2f}, Tabular: {avg_gate[3]:.2f}\n")  # V10 변경

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
        print(f"Fusion Gate -> Seller: {avg_gate[0]:.2f}, Review: {avg_gate[1]:.2f}, Image: {avg_gate[2]:.2f}, Tabular: {avg_gate[3]:.2f}\n")  # V10 변경
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "pth/best_mobile_version_v12_model_v3_0.005.pth")  # V10 변경
            print(f"🌟 New Best Mobile Version V12 Model Saved (MAE: {best_mae:.4f})\n")  # V10 변경

# ==========================================
# 8. Gate 분포 확인용 함수
# ==========================================
def check_gate_distribution(model, loader, device, save_path=None):
    model.eval()
    all_gates = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Gate Distribution Check"):
            out_fused, out_seller, out_review, out_image, gate, _, _, _, _ = model(
                batch["seller_input_ids"].to(device),
                batch["seller_attention_mask"].to(device),
                batch["review_input_ids"].to(device),
                batch["review_attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["price"].to(device),
                batch["price_missing"].to(device),
                batch["category_id"].to(device)
            )

            all_gates.append(gate.cpu().numpy())

    gates = np.concatenate(all_gates, axis=0)

    gate_df = pd.DataFrame(
        gates,
        columns=["Seller", "Review", "Image", "Tabular"]
    )

    print("\n========== Gate Mean ==========")
    print(gate_df.mean())

    print("\n========== Gate Std ==========")
    print(gate_df.std())

    print("\n========== Gate Min ==========")
    print(gate_df.min())

    print("\n========== Gate Max ==========")
    print(gate_df.max())

    print("\n========== Sample Gate 20개 ==========")
    print(gate_df.head(20))

    if save_path is not None:
        gate_df.to_csv(save_path, index=False)
        print(f"\nGate 분포 저장 완료: {save_path}")

    return gate_df

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # =========================
    # Gate 분포 확인 모드
    # =========================
    df = pd.read_csv(CSV_FILE).fillna({"seller_text": "None", "review_text": "None"})

    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    val_loader = DataLoader(
        AmazonFashionV9Dataset(val_df, transform=val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MultitaskFashionModelV9(
        num_cat=df["category_id"].nunique()
    ).to(DEVICE)

    # 본인 pth 경로로 수정
    MODEL_PATH = "pth/best_mobile_version_v12_model.pth"

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    print(f"Loaded model: {MODEL_PATH}")

    gate_df = check_gate_distribution(
        model,
        val_loader,
        DEVICE,
        save_path="gate_distribution_v12_0.02.csv"
    )