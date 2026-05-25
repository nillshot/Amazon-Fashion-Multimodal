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
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 하이퍼파라미터 및 설정 (CLIP 사전학습 전용)
# ==========================================
CSV_FILE = 'fashion_train_subset_2_with_images.csv'
BATCH_SIZE = 16  # 대조 학습은 배치가 클수록 좋으나 VRAM 상황을 고려하여 설정
EPOCHS = 5       # 패션 텍스트-이미지 도메인 정렬을 위한 기본 에폭
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device for Pre-training: {DEVICE}")

# ==========================================
# 2. Tokenizer & Image Transform
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
v3_weights = models.MobileNet_V3_Large_Weights.DEFAULT
base_transform = v3_weights.transforms()

# 사전학습용 강인한 데이터 증강 (배경 마스킹 및 플립 등)
pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    base_transform
])

# ==========================================
# 3. 사전학습용 데이터셋 (라벨 없이 이미지-텍스트 쌍만 반환)
# ==========================================
class FashionCLIPDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else base_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 텍스트 로드 및 토큰화
        text = str(row["input_text"])
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # 이미지 로드
        img_path = str(row["image_path"]).replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(img)
        except Exception:
            # 로딩 실패 시 Zero 텐서로 대체
            pixel_values = torch.zeros(3, 224, 224)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values
        }

# ==========================================
# 4. Fashion CLIP 모델 아키텍처 (MobileNet + RoBERTa 기반)
# ==========================================
class FashionCLIPModel(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128):
        super().__init__()
        # V9과 동일한 RoBERTa 및 선형 투영 레이어 선언
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_proj = nn.Linear(768, hidden_dim)
        
        # V9과 동일한 MobileNet-V3 Large 및 선형 투영 레이어 선언
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.image_encoder = mobilenet.features
        self.image_proj = nn.Linear(960, hidden_dim)
        
        # 대조 학습 공간(shared embedding space)으로 정사하기 위한 2층 MLP 투영 헤드 (Projection Head)
        self.text_clip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.image_clip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 학습 가능한 온도 파라미터 (learnable temperature)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ids, mask, pixels):
        # 1. 텍스트 인코딩 및 풀링
        t_outputs = self.text_encoder(ids, attention_mask=mask)
        t_seq = self.text_proj(t_outputs.last_hidden_state)  # [B, 128, 256]
        
        # 패딩을 제외한 토큰들의 평균 풀링 (Active Token Mean Pooling)
        active_mask = mask.unsqueeze(-1).float()
        t_pooled = (t_seq * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1e-9)  # [B, 256]
        t_emb = self.text_clip_head(t_pooled)  # [B, 128]
        
        # 2. 이미지 인코딩 및 풀링
        img_feat = self.image_encoder(pixels)  # [B, 960, 7, 7]
        B, C, H, W = img_feat.shape
        i_seq = img_feat.view(B, C, H * W).transpose(1, 2)
        i_seq = self.image_proj(i_seq)  # [B, 49, 256]
        
        # 전체 이미지 패치에 대한 평균 풀링
        i_pooled = i_seq.mean(dim=1)  # [B, 256]
        i_emb = self.image_clip_head(i_pooled)  # [B, 128]
        
        # 3. L2 정규화 (유사도 계산을 위한 단위 벡터화)
        t_emb = F.normalize(t_emb, p=2, dim=-1)
        i_emb = F.normalize(i_emb, p=2, dim=-1)
        
        return t_emb, i_emb

# ==========================================
# 5. 대칭 대조 손실 함수 (InfoNCE Loss)
# ==========================================
def compute_contrastive_loss(t_emb, i_emb, temp_param):
    B = t_emb.size(0)
    # 이미지와 텍스트 임베딩 간 코사인 유사도 연산 및 온도 파라미터 적용
    temp = torch.exp(temp_param)
    logits = torch.matmul(t_emb, i_emb.t()) * temp  # [B, B]
    
    # 자기 자신(정답 매칭)이 대각선에 위치하므로 Target은 0, 1, 2, ..., B-1
    labels = torch.arange(B, device=t_emb.device)
    
    # 텍스트 기준 이미지 맞추기 + 이미지 기준 텍스트 맞추기 (Symmetric Loss)
    loss_t = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.t(), labels)
    
    return (loss_t + loss_i) / 2

# ==========================================
# 6. 사전학습 기동 메인 루프
# ==========================================
def main():
    if not os.path.exists(CSV_FILE):
        print(f"CSV 파일 {CSV_FILE}을 찾을 수 없습니다. 경로를 확인해 주세요.")
        return
        
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    dataset = FashionCLIPDataset(df, transform=pretrain_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = FashionCLIPModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    print("\n--- Fashion CLIP-style Pre-training Started ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"CLIP Epoch {epoch}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            pixels = batch["pixel_values"].to(DEVICE)
            
            t_emb, i_emb = model(ids, mask, pixels)
            
            loss = compute_contrastive_loss(t_emb, i_emb, model.temperature)
            loss.backward()
            
            # 그레디언트 클리핑으로 안정성 도모
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Temp": f"{torch.exp(model.temperature).item():.2f}"})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} 완성 | 평균 대조 Loss: {avg_loss:.4f}")
        
    # 가중치 파일로 저장
    print("\n--- Saving Pretrained Encoders Weight ---")
    torch.save(model.text_encoder.state_dict(), "pretrained_fashion_roberta.pth")
    torch.save(model.image_encoder.state_dict(), "pretrained_fashion_mobilenet.pth")
    print("성공적으로 저장되었습니다!")
    print("1. pretrained_fashion_roberta.pth (RoBERTa 백본)")
    print("2. pretrained_fashion_mobilenet.pth (MobileNet-V3 백본)")
    print("\n[사용 방법]")
    print("V9 모델 코드인 `MultitaskFashionModelV9`의 __init__ 시점에 아래와 같이 로드해 사용해 보세요:")
    print(">>> model.text_encoder.load_state_dict(torch.load('pretrained_fashion_roberta.pth'))")
    print(">>> model.image_encoder.load_state_dict(torch.load('pretrained_fashion_mobilenet.pth'))")

if __name__ == "__main__":
    main()
