import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from multimodal_mobile_version_v7 import MultitaskFashionModelV7, AmazonFashionV7Dataset, val_transform, CSV_FILE
from multimodal_mobile_version_v8 import MultitaskFashionModelV8, AmazonFashionV8Dataset
from multimodal_mobile_version_v9 import MultitaskFashionModelV9, AmazonFashionV9Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load and prepare data
df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
le = LabelEncoder()
df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
_, val_df = train_test_split(df, test_size=0.1, random_state=42)

def evaluate_no_tqdm(model, val_loader, weights_path, model_name):
    print(f"Evaluating {model_name}...")
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
    print(f"[{model_name} Results] MAE: {mae:.4f} | MSE: {mse:.4f} | R2 Score: {r2:.4f}")

if __name__ == "__main__":
    BATCH_SIZE_EVAL = 32
    
    # Eval V7
    val_loader_v7 = DataLoader(AmazonFashionV7Dataset(val_df, transform=val_transform), batch_size=BATCH_SIZE_EVAL, shuffle=False)
    model_v7 = MultitaskFashionModelV7(num_cat=df["category_id"].nunique()).to(DEVICE)
    evaluate_no_tqdm(model_v7, val_loader_v7, "best_mobile_version_v7_model.pth", "Model V7")
    
    # Eval V8
    val_loader_v8 = DataLoader(AmazonFashionV8Dataset(val_df, transform=val_transform), batch_size=BATCH_SIZE_EVAL, shuffle=False)
    model_v8 = MultitaskFashionModelV8(num_cat=df["category_id"].nunique()).to(DEVICE)
    evaluate_no_tqdm(model_v8, val_loader_v8, "best_mobile_version_v8_model.pth", "Model V8")
    
    # Eval V9
    val_loader_v9 = DataLoader(AmazonFashionV9Dataset(val_df, transform=val_transform), batch_size=BATCH_SIZE_EVAL, shuffle=False)
    model_v9 = MultitaskFashionModelV9(num_cat=df["category_id"].nunique()).to(DEVICE)
    evaluate_no_tqdm(model_v9, val_loader_v9, "best_mobile_version_v9_model.pth", "Model V9")
