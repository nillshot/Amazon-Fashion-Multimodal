"""
Multimodal Fashion Model V11 - RunPod Optimized Version
=======================================================
V10 대비 핵심 변경사항 (Image gate 0.00 → ≥30% 목표):

[문제 진단]
  V10에서 Image gate가 E5 이후 완전히 0.00으로 붕괴됨.
  원인: Review 텍스트 분리로 4-way gate 경쟁 심화 + Image의 유일한
  gradient 경로인 Seller↔Image CrossAttention에서 Seller도 0.00으로 죽어
  Image gradient 경로 완전 차단.

[V11 수정 사항]
  1. Dual CrossAttention: Seller↔Image (Block1) + Review↔Image (Block2) 병렬 추가
     → Image가 Seller에 의존하지 않고 Review에서도 독립적 gradient 수신
  2. Constrained Gate: 각 모달리티 최솟값 보장
     → Image 최소 30%, Seller 5%, Review 10%, Tab 5%
  3. Image auxiliary loss 0.3 → 0.5 (V9 수준 이상으로 복구)
  4. Gate Entropy Loss 추가 (미래 gate collapse 방지)
  5. Early Stopping (patience=3, V10에서 E4 최솟값 0.3249 방치 방지)
  6. Phase2 Text LR 5e-6 → 2e-6 (RoBERTa 과적합 억제)

사용 예:
  python multimodal_mobile_version_v11_runpod.py \
      --data-csv /workspace/fashion_train_subset_3_with_meta_text_v10.csv \
      --output-dir /workspace/outputs_v11 \
      --epochs 10 \
      --phase1-epochs 2 \
      --batch-size 2 \
      --accum-steps 4 \
      --max-hours 11.5

  python multimodal_mobile_version_v11_runpod.py --smoke-test

  python multimodal_mobile_version_v11_runpod.py \
      --data-csv /workspace/fashion_train_subset_3_with_meta_text_v10.csv \
      --output-dir /workspace/outputs_v11 \
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
# 0. 유틸 (시드 / 로깅 / 인자)
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
    logger = logging.getLogger("v11_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "train_v11.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str, default="/workspace/fashion_train_subset_3_with_meta_text_v10.csv")
    p.add_argument("--output-dir", type=str, default="/workspace/outputs_v11")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--phase1-epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hours", type=float, default=11.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    # V11 신규: Early Stopping patience
    p.add_argument("--es-patience", type=int, default=3, help="Early Stopping patience (epochs)")
    return p.parse_args()


# ==========================================
# 1. Image Transform
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
# 2. 데이터셋 클래스 (V10과 동일)
# ==========================================
class AmazonFashionV11Dataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seller_text = str(row.get("seller_text", "None"))
        review_text = str(row.get("review_text", "None"))

        seller_enc = self.tokenizer(
            seller_text, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
        )
        review_enc = self.tokenizer(
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
# 3. CCR & CCS Loss (V10과 동일)
# ==========================================
class ContrastiveAttentionLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin
        )

    def compute_ccr(self, query, keys, attn_weights, k=5):
        dim = keys.size(-1)
        _, indices = torch.sort(attn_weights, dim=-1, descending=True)
        pos_idx, neg_idx = indices[:, :k], indices[:, -k:]
        pos_content = torch.gather(keys, 1, pos_idx.unsqueeze(-1).expand(-1, -1, dim)).mean(1)
        neg_content = torch.gather(keys, 1, neg_idx.unsqueeze(-1).expand(-1, -1, dim)).mean(1)
        return self.triplet_loss(query, pos_content, neg_content)

    def compute_ccs_hard_negative(self, query, attended_info):
        B = query.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=query.device, requires_grad=True)
        sim_matrix = F.cosine_similarity(query.unsqueeze(1), attended_info.unsqueeze(0), dim=-1)
        mask = torch.eye(B, device=query.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)
        hard_negative_indices = sim_matrix.argmax(dim=-1)
        hard_negatives = attended_info[hard_negative_indices]
        return self.triplet_loss(query, attended_info, hard_negatives)


# ==========================================
# 4. 모델 아키텍처
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
    """양방향 CrossAttention 단일 블록 (V10과 동일 구조)"""
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
        t_attended, t_attn_weights = self.text_img_attn(
            query=t_seq, key=i_seq, value=i_seq, need_weights=True
        )
        t_seq = self.norm1_t(t_seq + t_attended)
        t_seq = self.norm2_t(t_seq + self.ffn_t(t_seq))

        i_attended, _ = self.img_text_attn(
            query=i_seq, key=t_seq, value=t_seq, key_padding_mask=t_pad_mask
        )
        i_seq = self.norm1_i(i_seq + i_attended)
        i_seq = self.norm2_i(i_seq + self.ffn_i(i_seq))

        return t_seq, i_seq, t_attn_weights, t_attended


# V11 핵심: Constrained Gate
class ConstrainedModalityGate(nn.Module):
    """
    [V11 신규] 각 모달리티에 최솟값을 보장하는 Constrained Softmax Gate.
    Image 최소 30% 보장으로 gate collapse 구조적 방지.
    """
    def __init__(self, input_dim, num_modalities=4,
                 min_weights=(0.05, 0.10, 0.30, 0.05)):
        super().__init__()
        assert len(min_weights) == num_modalities
        assert sum(min_weights) < 1.0, "min_weights 합이 1.0 미만이어야 함"
        self.gate_linear = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_modalities),
        )
        # 각 모달리티 최솟값: [Seller, Review, Image, Tab]
        self.register_buffer("min_w", torch.tensor(min_weights, dtype=torch.float32))
        self.flex_budget = 1.0 - sum(min_weights)  # softmax가 배분할 수 있는 나머지 비율

    def forward(self, x):
        logits = self.gate_linear(x)
        soft = F.softmax(logits, dim=1)
        # 최솟값 하한 보장: weights = min_w + soft * flex_budget
        weights = self.min_w.unsqueeze(0) + soft * self.flex_budget
        return weights


class DualCrossAttentionFusion(nn.Module):
    """
    [V11 핵심 변경] Dual CrossAttention Fusion
    - Block1: Seller ↔ Image  (V10과 동일)
    - Block2: Review ↔ Image  (V11 신규: Image gradient 경로 복구)
    - 두 결과를 평균하여 최종 i_pooled 생성
    → Image가 Seller gate 0.00에 의존하지 않고
      Review에서도 독립적 gradient를 받음
    """
    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()
        # Block1: Seller ↔ Image (V10과 동일)
        self.seller_img_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])
        # Block2: Review ↔ Image (V11 신규)
        self.review_img_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # [V11] Constrained Gate: Image 최소 30% 보장
        # min_weights 순서: [Seller, Review, Image, Tab]
        self.gate = ConstrainedModalityGate(
            input_dim=dim * 4,
            num_modalities=4,
            min_weights=(0.05, 0.10, 0.30, 0.05),
        )

        self.fc_fused = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, seller_seq, review_seq, i_seq, tab,
                seller_pad_mask=None, review_pad_mask=None):
        # --- Block1: Seller ↔ Image ---
        s_out, i_out_s = seller_seq, i_seq
        s_attn_weights, s_attended = None, None
        for layer in self.seller_img_layers:
            s_out, i_out_s, s_attn_weights, s_attended = layer(s_out, i_out_s, seller_pad_mask)

        # --- Block2: Review ↔ Image (V11 신규) ---
        r_out, i_out_r = review_seq, i_seq
        r_attn_weights, r_attended = None, None
        for layer in self.review_img_layers:
            r_out, i_out_r, r_attn_weights, r_attended = layer(r_out, i_out_r, review_pad_mask)

        # Image: 두 CrossAttention 결과 평균 → 두 텍스트 소스 모두에서 gradient 수신
        i_out = (i_out_s + i_out_r) * 0.5

        # Seller pooling
        if seller_pad_mask is not None:
            s_mask = (~seller_pad_mask).unsqueeze(-1).float()
            seller_pooled = (s_out * s_mask).sum(1) / s_mask.sum(1).clamp(min=1e-9)
            s_attended_pooled = (s_attended * s_mask).sum(1) / s_mask.sum(1).clamp(min=1e-9)
            avg_s_attn = (s_attn_weights * s_mask).sum(1) / s_mask.sum(1).clamp(min=1e-9)
        else:
            seller_pooled = s_out.mean(1)
            s_attended_pooled = s_attended.mean(1)
            avg_s_attn = s_attn_weights.mean(1)

        # Review pooling
        if review_pad_mask is not None:
            r_mask = (~review_pad_mask).unsqueeze(-1).float()
            review_pooled = (r_out * r_mask).sum(1) / r_mask.sum(1).clamp(min=1e-9)
        else:
            review_pooled = r_out.mean(1)

        i_pooled = i_out.mean(1)

        # Constrained Gate
        concat_feat = torch.cat([seller_pooled, review_pooled, i_pooled, tab], dim=1)
        weights = self.gate(concat_feat)

        weighted_sum = (
            weights[:, 0:1] * seller_pooled +
            weights[:, 1:2] * review_pooled +
            weights[:, 2:3] * i_pooled +
            weights[:, 3:4] * tab
        )
        fused = self.fc_fused(
            torch.cat([weighted_sum, seller_pooled, review_pooled, i_pooled, tab], dim=1)
        )

        return (fused, seller_pooled, review_pooled, i_pooled, weights,
                avg_s_attn, s_attended_pooled, i_seq)


class MultitaskFashionModelV11(nn.Module):
    def __init__(self, num_cat, hidden_dim=256):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_proj = nn.Linear(768, hidden_dim)

        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.image_encoder = mobilenet.features
        self.image_proj = nn.Linear(960, hidden_dim)

        self.cat_emb = nn.Embedding(num_cat, 32)
        self.tab_fc = nn.Sequential(nn.Linear(1 + 1 + 32, hidden_dim), nn.ReLU())

        self.seller_self_attn = IntraModalitySelfAttention(hidden_dim)
        self.review_self_attn = IntraModalitySelfAttention(hidden_dim)
        self.image_self_attn = IntraModalitySelfAttention(hidden_dim)

        # V11: Dual CrossAttention Fusion
        self.fusion = DualCrossAttentionFusion(hidden_dim, num_layers=2)

        self.fused_regressor = nn.Linear(hidden_dim, 1)
        self.seller_regressor = nn.Linear(hidden_dim, 1)
        self.review_regressor = nn.Linear(hidden_dim, 1)
        self.image_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, seller_ids, seller_mask, review_ids, review_mask,
                pixels, price, miss, cat):
        seller_outputs = self.text_encoder(seller_ids, attention_mask=seller_mask)
        seller_seq = self.text_proj(seller_outputs.last_hidden_state)
        seller_pad_mask = (seller_mask == 0)

        review_outputs = self.text_encoder(review_ids, attention_mask=review_mask)
        review_seq = self.text_proj(review_outputs.last_hidden_state)
        review_pad_mask = (review_mask == 0)

        img_feat = self.image_encoder(pixels)
        B, C, H, W = img_feat.shape
        i_seq = img_feat.view(B, C, H * W).transpose(1, 2)
        i_seq = self.image_proj(i_seq)

        tab_feat = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))

        seller_seq = self.seller_self_attn(seller_seq, padding_mask=seller_pad_mask)
        review_seq = self.review_self_attn(review_seq, padding_mask=review_pad_mask)
        i_seq = self.image_self_attn(i_seq)

        (fused, seller_pooled, review_pooled, i_pooled, gates,
         avg_s_attn, s_attended_pooled, i_seq_raw) = self.fusion(
            seller_seq, review_seq, i_seq, tab_feat, seller_pad_mask, review_pad_mask
        )

        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze(-1) * 4 + 1
        out_seller = torch.sigmoid(self.seller_regressor(seller_pooled)).squeeze(-1) * 4 + 1
        out_review = torch.sigmoid(self.review_regressor(review_pooled)).squeeze(-1) * 4 + 1
        out_image = torch.sigmoid(self.image_regressor(i_pooled)).squeeze(-1) * 4 + 1

        return (out_fused, out_seller, out_review, out_image,
                gates, seller_pooled, i_seq_raw, avg_s_attn, s_attended_pooled)


# ==========================================
# 5. Loss 함수
# ==========================================
def weighted_mse_loss(pred, target):
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weight_lookup = torch.tensor([0.0, 4.0, 4.0, 3.0, 2.0, 1.0], device=target.device)
    weights = weight_lookup[target_rounded.long()]
    return (weights * (pred - target) ** 2).mean()


def gate_entropy_loss(weights):
    """
    [V11 신규] Gate Entropy 정규화 손실.
    엔트로피를 최대화하는 방향으로 패널티를 부여해 특정 모달리티로의
    추가적 쏠림을 억제함. Constrained Gate의 보조 역할.
    """
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
    return -entropy  # 엔트로피 최소화 = collapse 패널티


# ==========================================
# 6. Early Stopping
# ==========================================
class EarlyStopping:
    """[V11 신규] patience epochs 동안 val MAE 개선 없으면 학습 중단."""
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_mae = float("inf")

    def step(self, val_mae) -> bool:
        """True 반환 시 학습 중단."""
        if val_mae < self.best_mae - self.min_delta:
            self.best_mae = val_mae
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def state_dict(self):
        return {"counter": self.counter, "best_mae": self.best_mae}

    def load_state_dict(self, d):
        self.counter = d["counter"]
        self.best_mae = d["best_mae"]


# ==========================================
# 7. 학습 / 평가 루프
# ==========================================
def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                acc_steps, epoch, logger, deadline_ts, use_amp):
    model.train()
    contrastive_criterion = ContrastiveAttentionLoss().to(device)
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")

    for i, batch in enumerate(pbar):
        if time.time() > deadline_ts:
            logger.warning("⏰ 시간 제한 임박 — 현재 상태 저장 후 안전 종료합니다.")
            return total_loss / max(i, 1), True

        target = batch["target"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            (out_fused, out_seller, out_review, out_image, gate_weights,
             seller_pooled, i_seq, avg_s_attn, s_attended_pooled) = model(
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
            # [V11] Image auxiliary loss 0.3 → 0.5
            loss_task = loss_fused + 0.3 * loss_seller + 0.4 * loss_review + 0.5 * loss_image

            loss_ccr = contrastive_criterion.compute_ccr(seller_pooled, i_seq, avg_s_attn)
            loss_ccs = contrastive_criterion.compute_ccs_hard_negative(seller_pooled, s_attended_pooled)

            # [V11] Gate Entropy Loss 추가 (λ=0.05)
            loss_gate_ent = gate_entropy_loss(gate_weights)

            loss = (loss_task + 0.1 * loss_ccr + 0.1 * loss_ccs + 0.05 * loss_gate_ent) / acc_steps

        scaler.scale(loss).backward()

        if (i + 1) % acc_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * acc_steps
        pbar.set_postfix({
            "loss": f"{loss.item() * acc_steps:.3f}",
            "ccr": f"{loss_ccr.item():.3f}",
            "img_w": f"{gate_weights[:, 2].mean().item():.2f}",  # 이미지 gate 실시간 확인
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
            out_fused, _, _, _, gate, _, _, _, _ = model(
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
# 8. 체크포인트
# ==========================================
def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, phase, best_mae, es_state=None):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "phase": phase,
        "best_mae": best_mae,
        "early_stopping": es_state,
    }, path)


# ==========================================
# 9. 메인
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    logger.info(f"V11 RunPod Args: {vars(args)}")

    if args.smoke_test:
        logger.info("🧪 SMOKE TEST 모드")
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
        raise ValueError("V11 CSV에는 seller_text, review_text 컬럼이 필요합니다.")

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

    train_ds = AmazonFashionV11Dataset(train_df, tokenizer, train_transform)
    val_ds = AmazonFashionV11Dataset(val_df, tokenizer, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = MultitaskFashionModelV11(num_cat=num_cat).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model V11 params: {n_params:.1f}M")
    logger.info("[V11] Constrained Gate: Seller≥5% / Review≥10% / Image≥30% / Tab≥5%")
    logger.info("[V11] Dual CrossAttention: Seller↔Image + Review↔Image")
    logger.info("[V11] Image loss weight: 0.3 → 0.5")
    logger.info(f"[V11] Early Stopping patience: {args.es_patience}")

    scaler = GradScaler(enabled=use_amp)
    last_ckpt_path = output_dir / "last_ckpt_v11.pth"
    best_ckpt_path = output_dir / "best_mobile_version_v11_model.pth"

    start_epoch, start_phase, best_mae = 1, 1, float("inf")
    resume_optim_state = None
    early_stopper = EarlyStopping(patience=args.es_patience)

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
        if ck.get("early_stopping"):
            early_stopper.load_state_dict(ck["early_stopping"])
        logger.info(f"재개 지점 -> epoch={start_epoch}, phase={start_phase}, best_mae={best_mae:.4f}")
    elif args.resume:
        logger.warning("--resume이 지정됐지만 last_ckpt_v11.pth가 없어 처음부터 시작합니다.")

    start_ts = time.time()
    deadline_ts = start_ts + args.max_hours * 3600
    logger.info(f"안전 제한 종료 시각: {time.strftime('%H:%M:%S', time.localtime(deadline_ts))}")

    interrupted = False
    should_stop = False  # Early Stopping 플래그

    try:
        # ============ Phase 1: Text Encoder Frozen ============
        if start_phase <= 1:
            logger.info("--- [Phase 1] Text Encoder Frozen ---")
            for p in model.text_encoder.parameters():
                p.requires_grad = False

            optimizer_p1 = torch.optim.AdamW([
                {"params": model.image_encoder.parameters(), "lr": 2e-5},
                {"params": [p for n, p in model.named_parameters()
                            if "encoder" not in n and p.requires_grad], "lr": 1e-4},
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
                    _, _, val_mae, avg_gate = evaluate(model, val_loader, device, epoch, use_amp)
                    logger.info(
                        f"[P1-E{epoch}] train={train_loss:.4f} val_mae={val_mae:.4f} "
                        f"gate=Seller{avg_gate[0]:.2f}/Review{avg_gate[1]:.2f}"
                        f"/Image{avg_gate[2]:.2f}/Tab{avg_gate[3]:.2f}"
                    )

                save_ckpt(last_ckpt_path, model, optimizer_p1, scheduler_p1, scaler,
                          epoch, phase=1, best_mae=best_mae, es_state=early_stopper.state_dict())
                if interrupted:
                    raise TimeoutError("Phase 1에서 시간 제한 도달")

        # ============ Phase 2: Text Encoder Unfrozen ============
        logger.info("--- [Phase 2] Text Encoder Unfrozen ---")
        for p in model.text_encoder.parameters():
            p.requires_grad = True

        optimizer_p2 = torch.optim.AdamW([
            {"params": model.text_encoder.parameters(), "lr": 2e-6},   # [V11] 5e-6 → 2e-6
            {"params": model.image_encoder.parameters(), "lr": 1e-5},
            {"params": model.fusion.gate.parameters(), "lr": 3e-5},    # [V11] Gate 독립 lr 그룹
            {"params": [p for n, p in model.named_parameters()
                        if "encoder" not in n and "fusion.gate" not in n], "lr": 8e-5},  # [V11] 1e-4 → 8e-5
        ], weight_decay=0.02)
        steps_per_epoch_p2 = max(1, len(train_loader) // args.accum_steps)
        scheduler_p2 = CosineAnnealingLR(
            optimizer_p2,
            T_max=max(1, (args.epochs - args.phase1_epochs) * steps_per_epoch_p2)
        )

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
                _, _, val_mae, avg_gate = evaluate(model, val_loader, device, epoch, use_amp)
                logger.info(
                    f"[P2-E{epoch}] train={train_loss:.4f} val_mae={val_mae:.4f} "
                    f"gate=Seller{avg_gate[0]:.2f}/Review{avg_gate[1]:.2f}"
                    f"/Image{avg_gate[2]:.2f}/Tab{avg_gate[3]:.2f}"
                )

                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(model.state_dict(), best_ckpt_path)
                    logger.info(f"🌟 최적 V11 가중치 저장: MAE={best_mae:.4f} -> {best_ckpt_path}")

                # [V11] Early Stopping 체크
                should_stop = early_stopper.step(val_mae)
                if should_stop:
                    logger.info(f"⏹ Early Stopping: {args.es_patience} epochs 연속 개선 없음 → 학습 종료")

            save_ckpt(last_ckpt_path, model, optimizer_p2, scheduler_p2, scaler,
                      epoch, phase=2, best_mae=best_mae, es_state=early_stopper.state_dict())

            if interrupted:
                raise TimeoutError("Phase 2에서 시간 제한 도달")
            if should_stop:
                break

        logger.info(f"✅ 학습 완료! Best MAE: {best_mae:.4f}")

    except TimeoutError as e:
        logger.warning(f"⏰ {e} — last_ckpt_v11.pth 저장 완료. --resume으로 이어서 실행하세요.")
    except KeyboardInterrupt:
        logger.warning("사용자 중단 감지 — last_ckpt_v11.pth에 저장됩니다.")
    except Exception as e:
        logger.exception(f"예상치 못한 에러 발생: {e}")
        raise
    finally:
        elapsed = (time.time() - start_ts) / 3600
        logger.info(f"총 소요 시간: {elapsed:.2f}시간 | 최종 Best MAE = {best_mae:.4f}")


if __name__ == "__main__":
    main()
