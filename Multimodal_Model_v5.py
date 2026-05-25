import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel

"""
Multimodal Model v5 — CCR & CCS Implementation
================================================
Paper: "More Than Just Attention: Improving Cross-Modal Attentions with 
        Contrastive Constraints for Image-Text Matching" (arXiv:2105.09597)

Key Features:
1. Backbones: MobileNet-V2 (Image) & RoBERTa (Text)
2. CCR (Contrastive Content Re-sourcing): Guides attention to focus on relevant regions.
3. CCS (Contrastive Content Swapping): Ensures semantic alignment by swapping query tokens.
4. Plug-in Loss: Designed to be added to existing MSE/MAE training loops.
"""

# =====================================================================
# 1. Contrastive Constraints Loss 클래스
# =====================================================================

class ContrastiveAttentionLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveAttentionLoss, self).__init__()
        self.margin = margin
        # Triplet Loss를 사용하여 유사도 기반 제약 조건 구현
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), 
            margin=margin
        )

    def compute_ccr_loss(self, query, keys, attn_weights, k=5):
        """
        CCR: Query가 어텐션 상위 영역(Positive)과 더 가깝고, 하위 영역(Negative)과는 멀도록 학습.
        """
        batch_size, seq_len, dim = keys.size()
        
        # 가중치 순으로 정렬하여 상위/하위 k개 인덱스 확보
        sorted_weights, indices = torch.sort(attn_weights, dim=-1, descending=True)
        
        pos_indices = indices[:, :k]
        neg_indices = indices[:, -k:]
        
        pos_indices_exp = pos_indices.unsqueeze(-1).expand(-1, -1, dim)
        neg_indices_exp = neg_indices.unsqueeze(-1).expand(-1, -1, dim)
        
        # 상위/하위 k개의 특징값 추출
        pos_keys = torch.gather(keys, 1, pos_indices_exp)
        neg_keys = torch.gather(keys, 1, neg_indices_exp)
        
        pos_weights = torch.gather(attn_weights, 1, pos_indices)
        neg_weights = torch.gather(attn_weights, 1, neg_indices)
        
        # 가중합을 통한 Positive/Negative Content 생성
        pos_content = torch.sum(pos_keys * pos_weights.unsqueeze(-1), dim=1) / (torch.sum(pos_weights, dim=1, keepdim=True) + 1e-8)
        neg_content = torch.sum(neg_keys * neg_weights.unsqueeze(-1), dim=1) / (torch.sum(neg_weights, dim=1, keepdim=True) + 1e-8)
        
        # CCR Loss: query가 pos_content와 가깝도록
        return self.triplet_loss(query, pos_content, neg_content)

    def compute_ccs_loss(self, query, attended_info):
        """
        CCS: 추출된 이미지 정보(attended_info)가 가짜 텍스트(Swapped Query)보다 진짜 텍스트와 가깝도록 학습.
        """
        # Batch 내에서 데이터를 한 칸 밀어서 가짜(Negative) 샘플 생성
        swapped_query = torch.roll(query, shifts=1, dims=0)
        
        # CCS Loss: attended_info가 진짜 query와 더 가깝도록
        return self.triplet_loss(attended_info, query, swapped_query)


# =====================================================================
# 2. Cross-Attention v5 모듈
# =====================================================================

class CrossAttentionV5(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionV5, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, text_query, img_keys):
        Q = self.query_proj(text_query).unsqueeze(1)
        K = self.key_proj(img_keys)
        V = self.value_proj(img_keys)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        attended_info = torch.bmm(attn_weights, V).squeeze(1)
        
        return attended_info, attn_weights.squeeze(1)


# =====================================================================
# 3. Main Multimodal Model v5 (MobileNet-V2 + RoBERTa)
# =====================================================================

class MultimodalFoodModelV5(nn.Module):
    def __init__(self, embed_dim=512, text_model_name="roberta-base"):
        super(MultimodalFoodModelV5, self).__init__()
        
        # [Image] MobileNet-V2 Backbone
        mobilenet = models.mobilenet_v2(pretrained=True).features
        self.img_backbone = mobilenet 
        self.img_proj = nn.Linear(1280, embed_dim)
        
        # [Text] RoBERTa Backbone
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(768, embed_dim)
        
        # [Fusion] Cross-Attention
        self.cross_attn = CrossAttentionV5(embed_dim)
        
        # [Output] Rating Regressor
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, text_ids, attention_mask, images):
        # 1. Text Feature Extraction (RoBERTa CLS token)
        text_outputs = self.text_backbone(text_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_features)
        
        # 2. Image Feature Extraction (MobileNet Patch-level)
        img_features = self.img_backbone(images) # [B, 1280, 7, 7]
        img_features = img_features.flatten(2).transpose(1, 2) # [B, 49, 1280]
        img_emb = self.img_proj(img_features)
        
        # 3. Cross-modal Fusion with Attention
        attended_img_info, attn_weights = self.cross_attn(text_emb, img_emb)
        
        # 4. Final Rating Prediction
        fused_features = torch.cat([text_emb, attended_img_info], dim=-1)
        rating_pred = self.regressor(fused_features).squeeze(-1)
        
        return {
            "rating_pred": rating_pred,
            "text_emb": text_emb,
            "img_emb": img_emb,
            "attended_info": attended_img_info,
            "attn_weights": attn_weights
        }


# =====================================================================
# 4. Training Helper: Loss Integration Example
# =====================================================================

def compute_v5_loss(outputs, true_ratings, lambda_ccr=0.1, lambda_ccs=0.1):
    # Base Task Loss: MAE
    loss_task = F.l1_loss(outputs["rating_pred"], true_ratings)
    
    # Contrastive Constraints
    contrastive_criterion = ContrastiveAttentionLoss(margin=0.2)
    
    loss_ccr = contrastive_criterion.compute_ccr_loss(
        query=outputs["text_emb"], 
        keys=outputs["img_emb"], 
        attn_weights=outputs["attn_weights"]
    )
    
    loss_ccs = contrastive_criterion.compute_ccs_loss(
        query=outputs["text_emb"], 
        attended_info=outputs["attended_info"]
    )
    
    # Combined Loss
    total_loss = loss_task + (lambda_ccr * loss_ccr) + (lambda_ccs * loss_ccs)
    
    return total_loss, loss_task, loss_ccr, loss_ccs

if __name__ == "__main__":
    print("Multimodal Model v5 Module Loaded.")
    print("Architecture: RoBERTa-base + MobileNet-V2 + CCR/CCS Attention Constraints.")
