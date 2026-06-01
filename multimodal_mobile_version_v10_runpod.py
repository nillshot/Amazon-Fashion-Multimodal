"""
Multimodal Fashion Model V10 - RunPod Optimized Version
=======================================================
V9 RunPod 전용 코드의 실행 안정화 기능을 V10 구조에 이식한 버전입니다.

V10 유지 사항:
  - seller_text / review_text 분리 입력
  - Seller + Review + Image + Tabular 4-way Gate
  - MobileNetV3 Large + RoBERTa + Cross-Attention 기반 구조

RunPod 추가 사항:
  - argparse 기반 /workspace 경로 실행
  - 체크포인트 자동 저장 및 --resume 재개
  - --smoke-test 파이프라인 검증
  - AMP 혼합정밀도, Gradient Clipping
  - DataLoader 최적화(num_workers, pin_memory, drop_last)
  - max-hours 기반 안전 종료
  - train_v10.log 파일 로깅

사용 예:
  python multimodal_mobile_version_v10_runpod.py \
      --data-csv /workspace/fashion_train_subset_3_with_meta_text_v10.csv \
      --output-dir /workspace/outputs_v10 \
      --epochs 10 \
      --phase1-epochs 2 \
      --batch-size 2 \
      --accum-steps 4 \
      --max-hours 11.5

  python multimodal_mobile_version_v10_runpod.py --smoke-test

  python multimodal_mobile_version_v10_runpod.py \
      --data-csv /workspace/fashion_train_subset_3_with_meta_text_v10.csv \
      --output-dir /workspace/outputs_v10 \
      --resume
"""

import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path

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
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")


# ==========================================
# 0. RunPod 유틸 (V10 변경: V9_runpod 기능 이식)
# ==========================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v10_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "train_v10.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str, default="/workspace/fashion_train_subset_3_with_meta_text_v10.csv")  # V10 변경
    p.add_argument("--output-dir", type=str, default="/workspace/outputs_v10")  # V10 변경
    p.add_argument("--batch-size", type=int, default=2)  # V10 변경: seller/review를 각각 RoBERTa에 넣으므로 기본 2
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--phase1-epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hours", type=float, default=11.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


# ==========================================
# 1. Tokenizer & Image Transform
# ==========================================
def build_transforms():
    v3_weights = models.MobileNet_V3_Large_Weights.DEFAULT
    base_transform = v3_weights.transforms()
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        base_transform,
    ])
    return train_transform, base_transform


# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class AmazonFashionV10Dataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer  # V10 변경: RunPod num_workers 안정성을 위해 tokenizer를 명시적으로 주입
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        seller_text = str(row.get("seller_text", "None"))  # V10 변경: 판매자 텍스트 분리 입력
        review_text = str(row.get("review_text", "None"))  # V10 변경: 리뷰 텍스트 분리 입력

        seller_enc = self.tokenizer(  # V10 변경: seller_text 별도 토큰화
            seller_text, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
        )
        review_enc = self.tokenizer(  # V10 변경: review_text 별도 토큰화
            review_text, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
        )
        
        img_path = str(row["image_path"]).replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(img)
        except Exception:
            pixel_values = torch.zeros(3, 224, 224)

        price = torch.tensor([row.get("price_clean", 0.0)], dtype=torch.float32)
        price_missing = torch.tensor([row.get("price_missing", 1.0)], dtype=torch.float32)
        category = torch.tensor(row.get("category_id", 0), dtype=torch.long)
        target = torch.tensor(row["target"], dtype=torch.float32)

        return {
            "seller_input_ids": seller_enc["input_ids"].squeeze(0),
            "seller_attention_mask": seller_enc["attention_mask"].squeeze(0),
            "review_input_ids": review_enc["input_ids"].squeeze(0),
            "review_attention_mask": review_enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "price": price,
            "price_missing": price_missing,
            "category_id": category,
            "target": target,
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
    """ V9: Multi-Layer 구조 도입 """
    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),  # V10 변경: seller/review/image/tabular 4개 모달
            nn.ReLU(), 
            nn.Linear(dim, 4),  # V10 변경: gate 출력 3개 -> 4개
            nn.Softmax(dim=1)
        )
        
        self.fc_fused = nn.Sequential(
            nn.Linear(dim * 5, dim),  # V10 변경: weighted_sum + seller + review + image + tabular
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, seller_seq, review_seq, i_seq, tab, seller_pad_mask=None, review_pad_mask=None):  # V10 변경
        t_out, i_out = seller_seq, i_seq  # V10 변경: 기존 text 역할을 seller_text가 담당
        
        final_t_attn_weights = None
        final_t_attended = None
        
        # 설정한 층(layer) 수만큼 Cross-Attention 반복
        for layer in self.layers:
            t_out, i_out, t_attn_weights, t_attended = layer(t_out, i_out, seller_pad_mask)  # V10 변경
            final_t_attn_weights = t_attn_weights
            final_t_attended = t_attended
            
        # 시퀀스 차원 압축 (Pooling)
        if seller_pad_mask is not None:  # V10 변경
            mask = (~seller_pad_mask).unsqueeze(-1).float()  # V10 변경
            seller_pooled = (t_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # V10 변경
            t_attended_pooled = (final_t_attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            avg_t_attn_weights = (final_t_attn_weights * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            seller_pooled = t_out.mean(dim=1)  # V10 변경
            t_attended_pooled = final_t_attended.mean(dim=1)
            avg_t_attn_weights = final_t_attn_weights.mean(dim=1)

        if review_pad_mask is not None:  # V10 변경: review_text 별도 pooling
            r_mask = (~review_pad_mask).unsqueeze(-1).float()
            review_pooled = (review_seq * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1e-9)
        else:
            review_pooled = review_seq.mean(dim=1)
            
        i_pooled = i_out.mean(dim=1)
        
        concat_feat = torch.cat([seller_pooled, review_pooled, i_pooled, tab], dim=1)  # V10 변경
        weights = self.gate(concat_feat)
        
        weighted_sum = (
            weights[:, 0].unsqueeze(1) * seller_pooled +  # V10 변경
            weights[:, 1].unsqueeze(1) * review_pooled +  # V10 변경
            weights[:, 2].unsqueeze(1) * i_pooled +       # V10 변경
            weights[:, 3].unsqueeze(1) * tab              # V10 변경
        )
        fused = self.fc_fused(torch.cat([weighted_sum, seller_pooled, review_pooled, i_pooled, tab], dim=1))  # V10 변경
        
        return fused, seller_pooled, review_pooled, i_pooled, weights, avg_t_attn_weights, t_attended_pooled  # V10 변경

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
        self.fusion = InterModalityCrossAttentionV9(hidden_dim, num_layers=2)
        
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
# 6. 유틸리티 (학습 및 평가) - V10 RunPod 최적화
# ==========================================
def weighted_mse_loss(pred, target):
    """V10 변경: V9_runpod처럼 벡터화하여 속도 개선"""
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weight_lookup = torch.tensor([0.0, 4.0, 4.0, 3.0, 2.0, 1.0], device=target.device)
    weights = weight_lookup[target_rounded.long()]
    return (weights * (pred - target) ** 2).mean()


def train_epoch(model, loader, optimizer, scheduler, scaler, device, acc_steps, epoch, logger, deadline_ts, use_amp):
    model.train()
    contrastive_criterion = ContrastiveAttentionLoss().to(device)
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")

    for i, batch in enumerate(pbar):
        # V10 변경: RunPod 시간 제한 전 안전 종료
        if time.time() > deadline_ts:
            logger.warning("⏰ 시간 제한 임박 — 현재 상태 저장 후 안전 종료합니다.")
            return total_loss / max(i, 1), True

        target = batch["target"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            out_fused, out_seller, out_review, out_image, _, seller_pooled, i_seq, avg_t_attn_w, t_attended_pooled = model(
                batch["seller_input_ids"].to(device, non_blocking=True),
                batch["seller_attention_mask"].to(device, non_blocking=True),
                batch["review_input_ids"].to(device, non_blocking=True),
                batch["review_attention_mask"].to(device, non_blocking=True),
                batch["pixel_values"].to(device, non_blocking=True),
                batch["price"].to(device, non_blocking=True),
                batch["price_missing"].to(device, non_blocking=True),
                batch["category_id"].to(device, non_blocking=True),
            )

            loss_fused = weighted_mse_loss(out_fused, target)
            loss_seller = weighted_mse_loss(out_seller, target)
            loss_review = weighted_mse_loss(out_review, target)
            loss_image = weighted_mse_loss(out_image, target)
            loss_task = loss_fused + 0.3 * loss_seller + 0.4 * loss_review + 0.3 * loss_image

            loss_ccr = contrastive_criterion.compute_ccr(seller_pooled, i_seq, avg_t_attn_w)
            loss_ccs = contrastive_criterion.compute_ccs_hard_negative(seller_pooled, t_attended_pooled)
            loss = (loss_task + 0.1 * loss_ccr + 0.1 * loss_ccs) / acc_steps

        scaler.scale(loss).backward()

        if (i + 1) % acc_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # V10 변경: NaN/Inf 방지
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * acc_steps
        pbar.set_postfix({
            "loss": f"{loss.item() * acc_steps:.3f}",
            "ccr": f"{loss_ccr.item():.3f}",
            "ccs": f"{loss_ccs.item():.3f}",
        })

    return total_loss / len(loader), False


@torch.no_grad()
def evaluate(model, loader, device, epoch, use_amp):
    model.eval()
    total_loss = 0.0
    preds, targets, gates = [], [], []

    for batch in tqdm(loader, desc=f"Epoch {epoch} Eval"):
        target = batch["target"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            out_fused, out_seller, out_review, out_image, gate, _, _, _, _ = model(
                batch["seller_input_ids"].to(device, non_blocking=True),
                batch["seller_attention_mask"].to(device, non_blocking=True),
                batch["review_input_ids"].to(device, non_blocking=True),
                batch["review_attention_mask"].to(device, non_blocking=True),
                batch["pixel_values"].to(device, non_blocking=True),
                batch["price"].to(device, non_blocking=True),
                batch["price_missing"].to(device, non_blocking=True),
                batch["category_id"].to(device, non_blocking=True),
            )
            loss = weighted_mse_loss(out_fused, target)

        total_loss += loss.item()
        preds.extend(out_fused.float().cpu().numpy().reshape(-1).tolist())
        targets.extend(batch["target"].numpy().reshape(-1).tolist())
        gates.extend(gate.float().cpu().numpy())

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    avg_gate = np.mean(gates, axis=0)
    return total_loss / len(loader), mse, mae, avg_gate


# ==========================================
# 7. 체크포인트 함수 - V10 변경: RunPod 재개 지원
# ==========================================
def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, phase, best_mae):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "phase": phase,
        "best_mae": best_mae,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    logger.info(f"V10 RunPod Args: {vars(args)}")

    if args.smoke_test:
        logger.info("🧪 SMOKE TEST 모드 — 소량 데이터로 파이프라인만 빠르게 검증합니다.")
        args.epochs = 1
        args.phase1_epochs = 1
        args.batch_size = 2
        args.accum_steps = 1
        args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info(f"Device: {device} | AMP Enabled: {use_amp}")

    logger.info(f"Data loading from: {args.data_csv}")
    if not Path(args.data_csv).exists():
        logger.error(f"CSV not found: {args.data_csv}")
        sys.exit(1)

    df = pd.read_csv(args.data_csv).fillna({"seller_text": "None", "review_text": "None"})
    if "seller_text" not in df.columns or "review_text" not in df.columns:
        raise ValueError("V10 CSV에는 seller_text, review_text 컬럼이 필요합니다.")

    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    num_cat = len(le.classes_)

    if args.smoke_test:
        df = df.sample(min(32, len(df)), random_state=args.seed).reset_index(drop=True)
        logger.info(f"Smoke test sample size: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed)
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Total Categories: {num_cat}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_transform, val_transform = build_transforms()

    train_ds = AmazonFashionV10Dataset(train_df, tokenizer, train_transform)
    val_ds = AmazonFashionV10Dataset(val_df, tokenizer, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MultitaskFashionModelV9(num_cat=num_cat).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model V10 params: {n_params:.1f}M")

    scaler = GradScaler(enabled=use_amp)
    last_ckpt_path = output_dir / "last_ckpt_v10.pth"
    best_ckpt_path = output_dir / "best_mobile_version_v10_model.pth"

    start_epoch, start_phase, best_mae = 1, 1, float("inf")
    resume_optim_state = None

    if args.resume and last_ckpt_path.exists():
        logger.info(f"📂 중단지점부터 학습 재개: {last_ckpt_path}")
        ck = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        start_phase = ck["phase"]
        best_mae = ck.get("best_mae", float("inf"))
        resume_optim_state = (ck.get("optimizer"), ck.get("scheduler"))
        logger.info(f"재개 지점 -> epoch={start_epoch}, phase={start_phase}, best_mae={best_mae:.4f}")
    elif args.resume:
        logger.warning("--resume이 지정됐지만 last_ckpt_v10.pth가 없어 처음부터 시작합니다.")

    start_ts = time.time()
    deadline_ts = start_ts + args.max_hours * 3600
    logger.info(f"안전 제한 종료 시각: {time.strftime('%H:%M:%S', time.localtime(deadline_ts))}")

    interrupted = False

    try:
        # ============ Phase 1: Text Encoder Frozen ============
        if start_phase <= 1:
            logger.info("--- [Phase 1] Text Encoder Frozen ---")
            for p in model.text_encoder.parameters():
                p.requires_grad = False

            optimizer_p1 = torch.optim.AdamW([
                {"params": model.image_encoder.parameters(), "lr": 2e-5},
                {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": 1e-4},
            ])
            steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
            scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.phase1_epochs * steps_per_epoch)

            if start_phase == 1 and resume_optim_state and resume_optim_state[0]:
                optimizer_p1.load_state_dict(resume_optim_state[0])
                if resume_optim_state[1]:
                    scheduler_p1.load_state_dict(resume_optim_state[1])

            phase1_start = start_epoch if start_phase == 1 else 1
            for epoch in range(phase1_start, args.phase1_epochs + 1):
                train_loss, interrupted = train_epoch(
                    model, train_loader, optimizer_p1, scheduler_p1, scaler, device,
                    args.accum_steps, epoch, logger, deadline_ts, use_amp,
                )
                if not interrupted:
                    val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, device, epoch, use_amp)
                    logger.info(
                        f"[P1-E{epoch}] train={train_loss:.4f} val_mae={val_mae:.4f} "
                        f"gate=Seller{avg_gate[0]:.2f}/Review{avg_gate[1]:.2f}/Image{avg_gate[2]:.2f}/Tab{avg_gate[3]:.2f}"
                    )

                save_ckpt(last_ckpt_path, model, optimizer_p1, scheduler_p1, scaler, epoch, phase=1, best_mae=best_mae)
                if interrupted:
                    raise TimeoutError("Phase 1에서 시간 제한 도달")

        # ============ Phase 2: Text Encoder Unfrozen ============
        logger.info("--- [Phase 2] Text Encoder Unfrozen ---")
        for p in model.text_encoder.parameters():
            p.requires_grad = True

        optimizer_p2 = torch.optim.AdamW([
            {"params": model.text_encoder.parameters(), "lr": 5e-6},
            {"params": model.image_encoder.parameters(), "lr": 1e-5},
            {"params": [p for n, p in model.named_parameters() if "encoder" not in n], "lr": 1e-4},
        ])
        steps_per_epoch_p2 = max(1, len(train_loader) // args.accum_steps)
        scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=max(1, (args.epochs - args.phase1_epochs) * steps_per_epoch_p2))

        if start_phase == 2 and resume_optim_state and resume_optim_state[0]:
            optimizer_p2.load_state_dict(resume_optim_state[0])
            if resume_optim_state[1]:
                scheduler_p2.load_state_dict(resume_optim_state[1])

        phase2_start = start_epoch if start_phase == 2 else args.phase1_epochs + 1
        for epoch in range(phase2_start, args.epochs + 1):
            train_loss, interrupted = train_epoch(
                model, train_loader, optimizer_p2, scheduler_p2, scaler, device,
                args.accum_steps, epoch, logger, deadline_ts, use_amp,
            )
            if not interrupted:
                val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, device, epoch, use_amp)
                logger.info(
                    f"[P2-E{epoch}] train={train_loss:.4f} val_mae={val_mae:.4f} "
                    f"gate=Seller{avg_gate[0]:.2f}/Review{avg_gate[1]:.2f}/Image{avg_gate[2]:.2f}/Tab{avg_gate[3]:.2f}"
                )

                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(model.state_dict(), best_ckpt_path)
                    logger.info(f"🌟 최적 V10 가중치 저장: MAE={best_mae:.4f} -> {best_ckpt_path}")

            save_ckpt(last_ckpt_path, model, optimizer_p2, scheduler_p2, scaler, epoch, phase=2, best_mae=best_mae)
            if interrupted:
                raise TimeoutError("Phase 2에서 시간 제한 도달")

        logger.info(f"✅ 학습 완료! Best MAE: {best_mae:.4f}")

    except TimeoutError as e:
        logger.warning(f"⏰ {e} — last_ckpt_v10.pth 저장 완료. 재실행 시 --resume을 사용하세요.")
    except KeyboardInterrupt:
        logger.warning("사용자 중단 감지 — 현재까지의 체크포인트는 last_ckpt_v10.pth에 저장됩니다.")
    except Exception as e:
        logger.exception(f"예상치 못한 에러 발생: {e}")
        raise
    finally:
        elapsed = (time.time() - start_ts) / 3600
        logger.info(f"총 소요 시간: {elapsed:.2f}시간 | 최종 Best MAE = {best_mae:.4f}")


if __name__ == "__main__":
    main()
