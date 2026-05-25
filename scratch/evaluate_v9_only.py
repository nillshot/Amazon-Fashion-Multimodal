import os
import sys
# Add project root directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from multimodal_mobile_version_v7 import val_transform, CSV_FILE
from multimodal_mobile_version_v9 import MultitaskFashionModelV9, AmazonFashionV9Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_BATCH_SIZE = 64

def get_metrics(weights_filename, model_name, val_loader, df):
    weights_path = os.path.join(project_root, weights_filename)
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found!")
        return None
    
    model = MultitaskFashionModelV9(num_cat=df["category_id"].nunique()).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    true_labels, pred_fused = [], []
    with torch.no_grad():
        for batch in val_loader:
            target = batch["target"]
            outputs = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), 
                batch["pixel_values"].to(DEVICE), batch["price"].to(DEVICE), 
                batch["price_missing"].to(DEVICE), batch["category_id"].to(DEVICE)
            )
            out_fused = outputs[0]
            true_labels.extend(target.numpy())
            pred_fused.extend(out_fused.cpu().numpy())
            
    mae = mean_absolute_error(true_labels, pred_fused)
    mse = mean_squared_error(true_labels, pred_fused)
    r2 = r2_score(true_labels, pred_fused)
    return {"MAE": mae, "MSE": mse, "R2": r2}

def main():
    csv_path = os.path.join(project_root, CSV_FILE)
    df = pd.read_csv(csv_path).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    val_loader = DataLoader(
        AmazonFashionV9Dataset(val_df, transform=val_transform), 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False
    )
    
    print("Evaluating Model V9 v1...")
    v1_metrics = get_metrics("best_mobile_version_v9_model_v1.pth", "Model V9 v1", val_loader, df)
    
    print("Evaluating Model V9 v2...")
    v2_metrics = get_metrics("best_mobile_version_v9_model_v2.pth", "Model V9 v2", val_loader, df)
    
    print("\n================ V9 COMPARISON ================")
    if v1_metrics:
        print(f"Model V9 v1 (Existing):")
        print(f"  - MAE: {v1_metrics['MAE']:.4f}")
        print(f"  - MSE: {v1_metrics['MSE']:.4f}")
        print(f"  - R2 Score: {v1_metrics['R2']:.4f}")
    if v2_metrics:
        print(f"Model V9 v2 (With Pretrained weights):")
        print(f"  - MAE: {v2_metrics['MAE']:.4f}")
        print(f"  - MSE: {v2_metrics['MSE']:.4f}")
        print(f"  - R2 Score: {v2_metrics['R2']:.4f}")
    print("===============================================")

if __name__ == "__main__":
    main()
