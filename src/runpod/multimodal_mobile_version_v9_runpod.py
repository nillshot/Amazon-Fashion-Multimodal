"""
Multimodal Fashion Model V9 - RunPod Optimized Version
======================================================
주요 추가 사항:
  - 12시간 안전 타이머: 남은 시간 부족 시 자동 안전 종료 및 체크포인트 세이빙 (12시간 제한 대응)
  - 체크포인트 자동 저장/재개 (--resume 인자로 중단점부터 즉각 재개)
  - --smoke-test 모드 (GPU 할당 전 CPU 환경에서도 빠르게 파이프라인 검증 가능, 룰 2번 대응)
  - AMP (혼합정밀도, FP16) → VRAM 절약 + 학습 속도 극대화
  - DataLoader 최적화 (num_workers, pin_memory, drop_last)
  - Gradient Clipping (NaN/Inf 방지), 시드 고정(42), 로그 파일 로깅 구현
  - Fashion CLIP 사전학습 가중치 로드 지원 (--text-pretrained, --image-pretrained)

사용 예:
  # 1) 로컬에서 파이프라인 검증 (GPU 없이도 작동 확인 가능)
  python multimodal_mobile_version_v9_runpod.py --smoke-test

  # 2) RunPod에서 CLIP 사전학습 가중치를 연동하여 본격 학습 (10에폭)
  python multimodal_mobile_version_v9_runpod.py \
      --data-csv /workspace/fashion_train_subset_2_with_images.csv \
      --text-pretrained /workspace/pretrained_fashion_roberta.pth \
      --image-pretrained /workspace/pretrained_fashion_mobilenet.pth \
      --output-dir /workspace/outputs \
      --epochs 10 \
      --phase1-epochs 2 \
      --max-hours 11.5

  # 3) 시간 초과로 중단된 학습을 체크포인트부터 다시 이어가기
  python multimodal_mobile_version_v9_runpod.py \
      --data-csv /workspace/fashion_train_subset_2_with_images.csv \
      --output-dir /workspace/outputs \
      --resume
"""

import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
from transformers import AutoTokenizer, RobertaModel
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ==========================================
# 0. 유틸 (시드 / 로깅 / 인자)
# ==========================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 속도를 위해 False
    torch.backends.cudnn.benchmark = True


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v9_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "train_v9.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str,
                   default="/workspace/fashion_train_subset_2_with_images.csv")
    p.add_argument("--text-pretrained", type=str,
                   default="/workspace/pretrained_fashion_roberta.pth",
                   help="CLIP 사전학습된 RoBERTa 가중치 경로 (있을 때만 로드)")
    p.add_argument("--image-pretrained", type=str,
                   default="/workspace/pretrained_fashion_mobilenet.pth",
                   help="CLIP 사전학습된 MobileNet 가중치 경로 (있을 때만 로드)")
    p.add_argument("--output-dir", type=str, default="/workspace/outputs")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--phase1-epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hours", type=float, default=11.5,
                   help="이 시간이 지나면 다음 배치 직전에 안전 종료 (12시간 룰 대응)")
    p.add_argument("--resume", action="store_true",
                   help="output_dir/last_ckpt_v9.pth 에서 학습 재개")
    p.add_argument("--smoke-test", action="store_true",
                   help="소량 데이터로 전체 파이프라인 검증 모드 (CPU도 가능)")
    p.add_argument("--no-amp", action="store_true", help="AMP 비활성화")
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
class AmazonFashionV9Dataset(Dataset):
    def __init__(self, df, tokenizer, transform):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["input_text"])
        enc = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt")

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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "price": price,
            "price_missing": price_missing,
            "category_id": category,
            "target": target
        }


# ==========================================
# 3. CCR & CCS Loss 클래스 (V9: Hard Negative)
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
# 4. 모델 아키텍처 (V9: Multi-Layer Cross-Attention)
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
        
        for layer in self.layers:
            t_out, i_out, t_attn_weights, t_attended = layer(t_out, i_out, t_pad_mask)
            final_t_attn_weights = t_attn_weights
            final_t_attended = t_attended
            
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
        
        weighted_sum = weights[:, 0:1]*t_pooled + weights[:, 1:2]*i_pooled + weights[:, 2:3]*tab
        fused = self.fc_fused(torch.cat([weighted_sum, t_pooled, i_pooled, tab], dim=1))
        
        return fused, t_pooled, i_pooled, weights, avg_t_attn_weights, t_attended_pooled


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
        self.image_self_attn = IntraModalitySelfAttention(hidden_dim)
        
        self.fusion = InterModalityCrossAttentionV9(hidden_dim, num_layers=2)
        
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
        
        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze(-1) * 4 + 1
        out_text = torch.sigmoid(self.text_regressor(t_pooled)).squeeze(-1) * 4 + 1
        out_image = torch.sigmoid(self.image_regressor(i_pooled)).squeeze(-1) * 4 + 1
        
        return out_fused, out_text, out_image, gates, t_pooled, i_seq_self, avg_t_attn_weights, t_attended_pooled


# ==========================================
# 5. Loss / Train / Eval
# ==========================================
def weighted_mse_loss(pred, target):
    """ 벡터화: dict-comprehension 제거 및 속도 극대화 """
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weight_lookup = torch.tensor([0.0, 4.0, 4.0, 3.0, 2.0, 1.0], device=target.device)
    weights = weight_lookup[target_rounded.long()]
    return (weights * (pred - target)**2).mean()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    acc_steps, epoch, logger, deadline_ts, use_amp):
    model.train()
    contrastive_criterion = ContrastiveAttentionLoss().to(device)
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    
    for i, batch in enumerate(pbar):
        # 12시간 임박 시 조기 안전 저장 체크
        if time.time() > deadline_ts:
            logger.warning("⏰ 시간 제한 임박 — epoch 중간 저장 후 안전 종료합니다.")
            return total_loss / max(i, 1), True  # interrupted=True

        target = batch["target"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        price = batch["price"].to(device, non_blocking=True)
        miss = batch["price_missing"].to(device, non_blocking=True)
        cat = batch["category_id"].to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            out_fused, out_text, out_image, _, t_pooled, i_seq, avg_t_attn_w, t_attended_pooled = model(
                ids, mask, pixels, price, miss, cat
            )
            
            loss_fused = weighted_mse_loss(out_fused, target)
            loss_text = weighted_mse_loss(out_text, target)
            loss_image = weighted_mse_loss(out_image, target)
            loss_task = loss_fused + 0.4 * loss_text + 0.4 * loss_image
            
            loss_ccr = contrastive_criterion.compute_ccr(t_pooled, i_seq, avg_t_attn_w)
            loss_ccs = contrastive_criterion.compute_ccs_hard_negative(t_pooled, t_attended_pooled)
            
            loss = (loss_task + 0.1 * loss_ccr + 0.1 * loss_ccs) / acc_steps
            
        scaler.scale(loss).backward()
        
        if (i+1) % acc_steps == 0 or (i+1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * acc_steps
        pbar.set_postfix({
            "loss": f"{loss.item() * acc_steps:.3f}",
            "ccr": f"{loss_ccr.item():.3f}",
            "ccs": f"{loss_ccs.item():.3f}"
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
            out_fused, _, _, gate, _, _, _, _ = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["pixel_values"].to(device, non_blocking=True),
                batch["price"].to(device, non_blocking=True),
                batch["price_missing"].to(device, non_blocking=True),
                batch["category_id"].to(device, non_blocking=True)
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
# 6. 체크포인트 함수
# ==========================================
def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, phase, best_mae):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "phase": phase,
        "best_mae": best_mae
    }, path)


# ==========================================
# 7. 메인 실행부
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    logger.info(f"V9 RunPod Args: {vars(args)}")

    # ---- Smoke Test 모드 (룰 2 대응) ----
    if args.smoke_test:
        logger.info("🧪 SMOKE TEST 모드 — 소량 데이터로 파이프라인 검증 작동")
        args.epochs = 1
        args.phase1_epochs = 1
        args.batch_size = 2
        args.accum_steps = 1
        args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info(f"Device: {device} | AMP Enabled: {use_amp}")

    # ---- 데이터 로드 ----
    logger.info(f"Data loading from: {args.data_csv}")
    if not Path(args.data_csv).exists():
        logger.error(f"CSV not found: {args.data_csv}")
        sys.exit(1)
    df = pd.read_csv(args.data_csv).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    num_cat = len(le.classes_)

    if args.smoke_test:
        df = df.sample(min(32, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"Smoke test sample size: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Total Categories: {num_cat}")

    # ---- Tokenizer / Dataloader ----
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_transform, val_transform = build_transforms()

    train_ds = AmazonFashionV9Dataset(train_df, tokenizer, train_transform)
    val_ds = AmazonFashionV9Dataset(val_df, tokenizer, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # ---- 모델 생성 ----
    model = MultitaskFashionModelV9(num_cat=num_cat).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model V9 params: {n_params:.1f}M")

    # ---- Fashion CLIP 사전학습 가중치 로드 ----
    # (새로 학습을 누른 상황이고 중단 복구가 아닐 때만 사전학습 로드)
    if not args.resume:
        loaded_pretrained = False
        
        # RoBERTa 사전학습 로드
        if args.text_pretrained and Path(args.text_pretrained).exists():
            logger.info(f"🔥 loading CLIP Pretrained RoBERTa weights from {args.text_pretrained}")
            model.text_encoder.load_state_dict(torch.load(args.text_pretrained, map_location=device))
            loaded_pretrained = True
        else:
            logger.warning("⚠️ No CLIP RoBERTa pretrained weight loaded. Using default HuggingFace.")
            
        # MobileNet 사전학습 로드
        if args.image_pretrained and Path(args.image_pretrained).exists():
            logger.info(f"🔥 loading CLIP Pretrained MobileNet weights from {args.image_pretrained}")
            model.image_encoder.load_state_dict(torch.load(args.image_pretrained, map_location=device))
            loaded_pretrained = True
        else:
            logger.warning("⚠️ No CLIP MobileNet pretrained weight loaded. Using default ImageNet.")
            
        if loaded_pretrained:
            logger.info("✅ Fashion CLIP 사전학습 가중치 이식 성공!")

    scaler = GradScaler(enabled=use_amp)

    # ---- 체크포인트 재개 처리 ----
    last_ckpt_path = output_dir / "last_ckpt_v9.pth"
    best_ckpt_path = output_dir / "best_mobile_version_v9_model.pth"

    start_epoch, start_phase, best_mae = 1, 1, float("inf")
    resume_optim_state = None
    
    if args.resume and last_ckpt_path.exists():
        logger.info(f"📂 중단지점으로부터 학습을 재개합니다: {last_ckpt_path}")
        ck = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        start_phase = ck["phase"]
        best_mae = ck["best_mae"]
        resume_optim_state = (ck.get("optimizer"), ck.get("scheduler"))
        logger.info(f"재개 지점 정보 -> epoch={start_epoch}, phase={start_phase}, best_mae={best_mae:.4f}")

    # ---- 12시간 안전 한계 시각 ----
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

            optim_p1 = torch.optim.AdamW([
                {"params": model.image_encoder.parameters(), "lr": 2e-5},
                {"params": [p for n, p in model.named_parameters()
                            if "encoder" not in n and p.requires_grad], "lr": 1e-4},
            ])
            steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
            sched_p1 = CosineAnnealingLR(optim_p1, T_max=args.phase1_epochs * steps_per_epoch)

            if start_phase == 1 and resume_optim_state and resume_optim_state[0]:
                optim_p1.load_state_dict(resume_optim_state[0])
                sched_p1.load_state_dict(resume_optim_state[1])

            phase1_start = start_epoch if start_phase == 1 else 1
            for epoch in range(phase1_start, args.phase1_epochs + 1):
                train_loss, interrupted = train_one_epoch(
                    model, train_loader, optim_p1, sched_p1, scaler, device,
                    args.accum_steps, epoch, logger, deadline_ts, use_amp
                )
                if not interrupted:
                    val_loss, val_mse, val_mae, avg_gate = evaluate(
                        model, val_loader, device, epoch, use_amp
                    )
                    logger.info(f"[P1-E{epoch}] train={train_loss:.4f} val_mae={val_mae:.4f} "
                                f"gate=T{avg_gate[0]:.2f}/I{avg_gate[1]:.2f}/Tab{avg_gate[2]:.2f}")

                # 매 에폭 세이브
                save_ckpt(last_ckpt_path, model, optim_p1, sched_p1, scaler,
                          epoch, phase=1, best_mae=best_mae)

                if interrupted:
                    raise TimeoutError("Phase 1에서 12시간 시간 제한 도달")

        # ============ Phase 2: Text Encoder Unfrozen ============
        logger.info("--- [Phase 2] Text Encoder Unfrozen ---")
        for p in model.text_encoder.parameters():
            p.requires_grad = True

        optim_p2 = torch.optim.AdamW([
            {"params": model.text_encoder.parameters(), "lr": 5e-6},
            {"params": model.image_encoder.parameters(), "lr": 1e-5},
            {"params": [p for n, p in model.named_parameters()
                        if "encoder" not in n], "lr": 1e-4},
        ])
        steps_per_epoch_p2 = max(1, len(train_loader) // args.accum_steps)
        sched_p2 = CosineAnnealingLR(
            optim_p2, T_max=(args.epochs - args.phase1_epochs) * steps_per_epoch_p2
        )

        if start_phase == 2 and resume_optim_state and resume_optim_state[0]:
            optim_p2.load_state_dict(resume_optim_state[0])
            sched_p2.load_state_dict(resume_optim_state[1])

        phase2_start = start_epoch if start_phase == 2 else args.phase1_epochs + 1
        for epoch in range(phase2_start, args.epochs + 1):
            train_loss, interrupted = train_one_epoch(
                model, train_loader, optim_p2, sched_p2, scaler, device,
                args.accum_steps, epoch, logger, deadline_ts, use_amp
            )
            if not interrupted:
                val_loss, val_mse, val_mae, avg_gate = evaluate(
                    model, val_loader, device, epoch, use_amp
                )
                logger.info(f"[P2-E{epoch}] train={train_loss:.4f} val_mae={val_mae:.4f} "
                            f"gate=T{avg_gate[0]:.2f}/I{avg_gate[1]:.2f}/Tab{avg_gate[2]:.2f}")

                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(model.state_dict(), best_ckpt_path)
                    logger.info(f"🌟 최적 성능 가중치 갱신: MAE={best_mae:.4f} -> {best_ckpt_path}")

            save_ckpt(last_ckpt_path, model, optim_p2, sched_p2, scaler,
                      epoch, phase=2, best_mae=best_mae)

            if interrupted:
                raise TimeoutError("Phase 2에서 12시간 시간 제한 도달")

        logger.info(f"✅ 학습 성공적으로 완료! 최우수 MAE: {best_mae:.4f}")

    except TimeoutError as e:
        logger.warning(f"⏰ {e} — last_ckpt_v9.pth 안전하게 저장 완료. RunPod 인스턴스 종료 전 재실행 시 --resume 으로 이어서 가동하세요.")
    except KeyboardInterrupt:
        logger.warning("사용자 중단(Ctrl+C) 감지 — 최종 학습 상태 체크포인트 저장 시도")
    except Exception as e:
        logger.exception(f"예상치 못한 에러가 발생했습니다: {e}")
        raise
    finally:
        elapsed = (time.time() - start_ts) / 3600
        logger.info(f"총 소요 시간: {elapsed:.2f}시간 | 최종 Best MAE = {best_mae:.4f}")


if __name__ == "__main__":
    main()
