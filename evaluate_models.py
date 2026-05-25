import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# V7 파일에서 필요한 요소 불러오기
from multimodal_mobile_version_v7 import (
    MultitaskFashionModelV7, 
    AmazonFashionV7Dataset, 
    val_transform, 
    CSV_FILE, 
    BATCH_SIZE
)

# V8 파일에서 필요한 요소 불러오기
from multimodal_mobile_version_v8 import (
    MultitaskFashionModelV8, 
    AmazonFashionV8Dataset
)

# V9 파일에서 필요한 요소 불러오기
from multimodal_mobile_version_v9 import (
    MultitaskFashionModelV9, 
    AmazonFashionV9Dataset
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 그림 생성 함수
def create_evaluation_plots(y_true, y_pred, prefix):
    out_dir = "model_evaluate" if os.path.exists("model_evaluate") else "."
    
    # 1. 실제값 vs 예측값 상관관계
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f'{prefix} - 실제값 vs 예측값 상관관계')
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_상관관계그래프.png'), dpi=300)
    plt.close()
    
    # 2. 오차 누적 분포 (Error CDF)
    abs_errors = np.abs(y_true - y_pred)
    sorted_errors = np.sort(abs_errors)
    p = 1. * np.arange(len(abs_errors)) / (len(abs_errors) - 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, p, color='orange', linewidth=2)
    plt.title(f'{prefix} - 오차 누적 분포 (Error CDF)')
    plt.xlabel('Absolute Error')
    plt.ylabel('Proportion')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(sorted_errors))
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_오차누적분포.png'), dpi=300)
    plt.close()
    
    # 3. Model Performance Summary (성능 요약 그림)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    max_err = max_error(y_true, y_pred)
    
    plt.figure(figsize=(8, 4))
    plt.axis('off')
    textstr = f"--- {prefix} Performance Summary ---\n\n"
    textstr += f"R-Squared: {r2:.4f}\n"
    textstr += f"MAE: {mae:.2f}\n"
    textstr += f"RMSE: {rmse:.2f}\n"
    textstr += f"MAPE: {mape:.2f}%\n"
    textstr += f"Max Error: {max_err:.2f}"
    
    plt.text(0.5, 0.5, textstr, fontsize=15, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_성능요약.png'), dpi=300)
    plt.close()

# 모델 평가 함수
def evaluate_model(model, val_loader, weights_path, model_name):
    print(f"\n--- {model_name} 평가 시작 ---")
    if not os.path.exists(weights_path):
        print(f"경고: {weights_path} 파일이 없습니다. 건너뜁니다.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    true_labels, pred_fused = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"{model_name} 예측 중"):
            target = batch["target"].to(DEVICE)
            # V7은 8개의 값을 반환하므로 첫 번째 값(out_fused)만 사용
            outputs = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), 
                batch["pixel_values"].to(DEVICE), batch["price"].to(DEVICE), 
                batch["price_missing"].to(DEVICE), batch["category_id"].to(DEVICE)
            )
            out_fused = outputs[0]
            true_labels.extend(target.cpu().numpy())
            pred_fused.extend(out_fused.cpu().numpy())
            
    create_evaluation_plots(np.array(true_labels), np.array(pred_fused), model_name)
    
    # 지표 계산 및 출력
    mae = mean_absolute_error(true_labels, pred_fused)
    mse = mean_squared_error(true_labels, pred_fused)
    r2 = r2_score(true_labels, pred_fused)
    print(f"\n[{model_name} Metrics]")
    print(f"- MAE: {mae:.4f}")
    print(f"- MSE: {mse:.4f}")
    print(f"- R2 Score: {r2:.4f}")
    print(f"{model_name} 평가 및 세 장의 그림 생성 완료!")

# 메인 실행부
def main():
    print("1. V7 평가 데이터 준비 중...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    EVAL_BATCH_SIZE = 32
    print(f"Evaluation Batch Size: {EVAL_BATCH_SIZE}")
    
    # V7 가중치 파일 평가
    # print("--- V7 모델 준비 중 ---")
    # val_loader_v7 = DataLoader(
    #     AmazonFashionV7Dataset(val_df, transform=val_transform), 
    #     batch_size=EVAL_BATCH_SIZE, 
    #     shuffle=False
    # )
    # model_v7 = MultitaskFashionModelV7(num_cat=df["category_id"].nunique()).to(DEVICE)
    # evaluate_model(model_v7, val_loader_v7, "best_mobile_version_v7_model.pth", "Model_V7")
    
    # V8 가중치 파일 평가
    # print("\n--- V8 모델 준비 중 ---")
    # val_loader_v8 = DataLoader(
    #     AmazonFashionV8Dataset(val_df, transform=val_transform), 
    #     batch_size=EVAL_BATCH_SIZE, 
    #     shuffle=False
    # )
    # model_v8 = MultitaskFashionModelV8(num_cat=df["category_id"].nunique()).to(DEVICE)
    # evaluate_model(model_v8, val_loader_v8, "best_mobile_version_v8_model.pth", "Model_V8")
    
    # V9 가중치 파일 평가
    print("\n--- V9 모델 준비 중 ---")
    val_loader_v9 = DataLoader(
        AmazonFashionV9Dataset(val_df, transform=val_transform), 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False
    )
    
    # V9 v1 가중치 파일 평가
    print("\n--- V9 v1 모델 평가 중 ---")
    model_v9_v1 = MultitaskFashionModelV9(num_cat=df["category_id"].nunique()).to(DEVICE)
    evaluate_model(model_v9_v1, val_loader_v9, "best_mobile_version_v9_model_v1.pth", "Model_V9_v1")
    
    # V9 v2 가중치 파일 평가
    print("\n--- V9 v2 모델 평가 중 ---")
    model_v9_v2 = MultitaskFashionModelV9(num_cat=df["category_id"].nunique()).to(DEVICE)
    evaluate_model(model_v9_v2, val_loader_v9, "best_mobile_version_v9_model_v2.pth", "Model_V9_v2")
    
    print("\n모든 모델 평가가 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    main()
