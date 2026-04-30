# ==========================================
# Google Colab 전용: 타겟 드롭아웃 멀티모달 학습 스크립트 (TensorFlow 버젼)
# ==========================================
# 실행 전 필수 작업 (Colab 환경):
# 1. Colab에서 새 노트를 열고 런타임 유형을 'GPU'로 설정하세요.
# 2. 첫 번째 셀에 아래 명령어를 넣고 실행하여 라이브러리를 설치하세요:
#    !pip install transformers scikit-learn pandas pillow tqdm tensorflow
# 3. 구글 드라이브의 '내 드라이브/BigData' 폴더 안에 아래 파일/폴더가 있어야 합니다:
#    - fashion_train_subset_2_with_images.csv
#    - images/ 폴더 (또는 그 안의 이미지들)
# ==========================================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # 최신 TF에서 Keras 2 방식 호환성 확보

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image

# 최신 코랩 환경의 ImportError 해결을 위한 강제 임포트 로직
from transformers import AutoTokenizer
try:
    # 1. 일반적인 방식 시도
    from transformers import TFAutoModel
except ImportError:
    try:
        # 2. 직접 경로 시도 (최신 코랩 환경 대응)
        import transformers.models.auto.modeling_tf_auto as tf_modeling
        TFAutoModel = tf_modeling.TFAutoModel
    except Exception as e:
        # 3. 실패 시 상세 에러 출력
        print(f"오류: 텐서플로우 모델을 불러올 수 없습니다. ({e})")
        print("코랩 런타임을 다시 시작하거나, PyTorch 버전을 사용하세요.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import random

# ==========================================
# 1. 환경 설정 (구글 드라이브 마운트)
# ==========================================
print("구글 드라이브 마운트를 시작합니다...")
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/BigData'
CSV_FILE = os.path.join(BASE_DIR, 'fashion_train_subset_2_with_images.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

BATCH_SIZE = 16 
ACCUMULATION_STEPS = 1

EPOCHS_PHASE1 = 2
EPOCHS_PHASE2 = 5
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5

print(f"\n사용 가능한 GPU 개수: {len(tf.config.list_physical_devices('GPU'))}")

# ==========================================
# 2. 데이터 제너레이터 (Sequence)
# ==========================================
class AmazonFashionFullSequence(Sequence):
    def __init__(self, df, tokenizer, batch_size=16, is_training=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.is_training = is_training
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        input_ids, attention_masks, pixel_values = [], [], []
        prices, price_missings, categories, targets = [], [], [], []
        
        for _, row in batch_df.iterrows():
            text = str(row["input_text"])
            enc = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
            input_ids.append(enc["input_ids"][0])
            attention_masks.append(enc["attention_mask"][0])
            
            original_img_path = str(row["image_path"]).replace("\\", "/")
            img_filename = os.path.basename(original_img_path)
            img_path = os.path.join(IMAGE_DIR, img_filename)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                img_arr = np.array(img, dtype=np.float32)
                # Keras EfficientNetB0 내부의 전처리 함수 사용
                img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
            except:
                # 에러시 zero 이미지
                img_arr = np.zeros((224, 224, 3), dtype=np.float32)

            pixel_values.append(img_arr)
            prices.append([row["price_clean"]])
            price_missings.append([row["price_missing"]])
            categories.append(row["category_id"])
            targets.append(row["target"])

        return (
            tf.convert_to_tensor(input_ids, dtype=tf.int32),
            tf.convert_to_tensor(attention_masks, dtype=tf.int32),
            tf.convert_to_tensor(pixel_values, dtype=tf.float32),
            tf.convert_to_tensor(prices, dtype=tf.float32),
            tf.convert_to_tensor(price_missings, dtype=tf.float32),
            tf.convert_to_tensor(categories, dtype=tf.int32)
        ), tf.convert_to_tensor(targets, dtype=tf.float32)

    def on_epoch_end(self):
        if self.is_training:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

# ==========================================
# 3. 모델 아키텍처 (Targeted Modality Dropout 적용)
# ==========================================
class TargetedModalityDropout(tf.keras.layers.Layer):
    def __init__(self, text_drop_p=0.8, general_drop_p=0.2, **kwargs):
        super().__init__(**kwargs)
        self.text_drop_p = text_drop_p
        self.general_drop_p = general_drop_p

    def call(self, inputs, training=None):
        t, i, tab = inputs
        if not training:
            return t, i, tab
            
        batch_size = tf.shape(t)[0]
        
        # 1. 텍스트 강제 마스킹 (80% 확률)
        rand_text = tf.random.uniform((batch_size,))
        text_mask = tf.cast(rand_text >= self.text_drop_p, dtype=tf.float32)
        
        # 2. 이미지/정형 데이터 일반화 (20% 확률로 둘 중 하나 드롭아웃)
        rand_gen = tf.random.uniform((batch_size,))
        apply_gen = rand_gen < self.general_drop_p
        
        rand_choice = tf.random.uniform((batch_size,)) < 0.5
        img_mask = tf.cast(tf.logical_not(tf.logical_and(apply_gen, rand_choice)), dtype=tf.float32)
        tab_mask = tf.cast(tf.logical_not(tf.logical_and(apply_gen, tf.logical_not(rand_choice))), dtype=tf.float32)
        
        return t * tf.expand_dims(text_mask, 1), i * tf.expand_dims(img_mask, 1), tab * tf.expand_dims(tab_mask, 1)

class ThreeWayGMU(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        t, i, tab = inputs
        concat_feat = tf.concat([t, i, tab], axis=1)
        h = self.fc1(concat_feat)
        weights = self.fc2(h)
        
        w_t = tf.expand_dims(weights[:, 0], 1)
        w_i = tf.expand_dims(weights[:, 1], 1)
        w_tab = tf.expand_dims(weights[:, 2], 1)
        
        fused = w_t * t + w_i * i + w_tab * tab
        return fused, weights

class MultitaskFashionModelTF(tf.keras.Model):
    def __init__(self, num_cat, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        # TFAutoModel을 사용하여 호환성 문제 해결
        self.text_encoder = TFAutoModel.from_pretrained("roberta-base")
        self.image_encoder = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
        
        self.text_fc = tf.keras.layers.Dense(hidden_dim)
        self.image_fc = tf.keras.layers.Dense(hidden_dim)
        
        self.cat_emb = tf.keras.layers.Embedding(num_cat, 32)
        self.tab_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        
        self.modality_dropout = TargetedModalityDropout(0.8, 0.2)
        self.gmu = ThreeWayGMU(hidden_dim)
        
        self.text_regressor = tf.keras.layers.Dense(1)
        self.image_regressor = tf.keras.layers.Dense(1)
        self.fused_regressor = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        ids, mask, pixels, price, miss, cat = inputs
        
        # Text feature
        t_out = self.text_encoder(input_ids=ids, attention_mask=mask, training=training)
        t_feat_raw = self.text_fc(t_out.pooler_output)
        
        # Image feature
        i_out = self.image_encoder(pixels, training=training)
        i_feat_raw = self.image_fc(i_out)
        
        # Tabular feature
        cat_feat = self.cat_emb(cat)
        tab_input = tf.concat([price, miss, cat_feat], axis=1)
        tab_feat_raw = self.tab_fc(tab_input)
        
        # Sub-task outputs
        out_text = tf.squeeze(tf.sigmoid(self.text_regressor(t_feat_raw))) * 4.0 + 1.0
        out_image = tf.squeeze(tf.sigmoid(self.image_regressor(i_feat_raw))) * 4.0 + 1.0
        
        # Fused output
        t_feat, i_feat, tab_feat = self.modality_dropout((t_feat_raw, i_feat_raw, tab_feat_raw), training=training)
        fused, gates = self.gmu((t_feat, i_feat, tab_feat))
        out_fused = tf.squeeze(tf.sigmoid(self.fused_regressor(fused))) * 4.0 + 1.0
        
        return out_fused, out_text, out_image, gates

    def freeze_backbones(self):
        self.text_encoder.trainable = False
        self.image_encoder.trainable = False

    def unfreeze_backbones(self):
        self.text_encoder.trainable = True
        self.image_encoder.trainable = True

# ==========================================
# 4. 학습/평가 유틸리티
# ==========================================
@tf.function
def weighted_mse_loss(pred, target):
    target_rounded = tf.clip_by_value(tf.round(target), 1.0, 5.0)
    
    weights = tf.ones_like(target)
    weights = tf.where(target_rounded <= 2.0, 4.0, weights)
    weights = tf.where(target_rounded == 3.0, 3.0, weights)
    weights = tf.where(target_rounded == 4.0, 2.0, weights)
    weights = tf.where(target_rounded == 5.0, 1.0, weights)
    
    return tf.reduce_mean(weights * tf.square(pred - target))

@tf.function
def train_step(model, inputs, target, optimizer, acc_steps):
    with tf.GradientTape() as tape:
        out_fused, out_text, out_image, _ = model(inputs, training=True)
        
        loss_fused = weighted_mse_loss(out_fused, target)
        loss_text = weighted_mse_loss(out_text, target)
        loss_image = weighted_mse_loss(out_image, target)
        
        loss = (loss_fused + 0.5 * loss_text + 0.5 * loss_image) / tf.cast(acc_steps, tf.float32)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss * tf.cast(acc_steps, tf.float32)

def train_epoch(model, generator, optimizer, acc_steps, epoch, phase_name):
    total_loss = 0.0
    pbar = tqdm(range(len(generator)), desc=f"[{phase_name}] Epoch {epoch} Train")
    for i in pbar:
        inputs, target = generator[i]
        loss = train_step(model, inputs, target, optimizer, acc_steps)
        total_loss += loss.numpy()
        pbar.set_postfix({'loss': loss.numpy()})
    generator.on_epoch_end()
    return total_loss / len(generator)

def evaluate(model, generator, epoch, phase_name):
    total_loss = 0.0
    preds, targets_list, gates_list = [], [], []
    
    for i in tqdm(range(len(generator)), desc=f"[{phase_name}] Epoch {epoch} Eval"):
        inputs, target = generator[i]
        out_fused, out_text, out_image, gates = model(inputs, training=False)
        
        loss = weighted_mse_loss(out_fused, target)
        total_loss += loss.numpy()
        
        preds.extend(out_fused.numpy())
        targets_list.extend(target.numpy())
        gates_list.extend(gates.numpy())
        
    mse = mean_squared_error(targets_list, preds)
    mae = mean_absolute_error(targets_list, preds)
    avg_gate = np.mean(gates_list, axis=0)
    return total_loss / len(generator), mse, mae, avg_gate

# ==========================================
# 5. 메인 실행부
# ==========================================
def main():
    if not os.path.exists(CSV_FILE):
        print(f"오류: {CSV_FILE} 파일을 찾을 수 없습니다. 경로 설정을 다시 확인해주세요.")
        return

    print("1. 데이터 로딩 중...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    train_generator = AmazonFashionFullSequence(train_df, tokenizer, batch_size=BATCH_SIZE, is_training=True)
    val_generator = AmazonFashionFullSequence(val_df, tokenizer, batch_size=BATCH_SIZE, is_training=False)
    
    model = MultitaskFashionModelTF(num_cat=df["category_id"].nunique())
    
    # 모델 빌드(Build)를 위해 dummy call 실행
    inputs_dummy, _ = train_generator[0]
    _ = model(inputs_dummy, training=False)
    
    best_mae = float('inf')

    print("\nPHASE 1: Feature Extraction (백본 모델 동결)")
    model.freeze_backbones()
    
    # TensorFlow 2.10 이상부터 tf.keras.optimizers에 AdamW가 포함되어 있습니다.
    optimizer_p1 = tf.keras.optimizers.AdamW(learning_rate=LR_PHASE1)
    
    for epoch in range(1, EPOCHS_PHASE1 + 1):
        train_loss = train_epoch(model, train_generator, optimizer_p1, ACCUMULATION_STEPS, epoch, "Phase 1")
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_generator, epoch, "Phase 1")
        print(f"[Phase 1 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f} | Val MSE: {val_mse:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")

    print("\nPHASE 2: Full Fine-tuning (전체 모델 미세조정)")
    model.unfreeze_backbones()
    
    # Cosine Annealing Learning Rate 적용
    total_steps = EPOCHS_PHASE2 * len(train_generator)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=LR_PHASE2, decay_steps=total_steps)
    optimizer_p2 = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    
    for epoch in range(1, EPOCHS_PHASE2 + 1):
        train_loss = train_epoch(model, train_generator, optimizer_p2, ACCUMULATION_STEPS, epoch, "Phase 2")
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_generator, epoch, "Phase 2")
        print(f"[Phase 2 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f} | Val MSE: {val_mse:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            model_save_path = os.path.join(BASE_DIR, "best_colab_targeted_dropout_model_tf")
            model.save_weights(model_save_path)
            print(f"🌟 새로운 최고 성능 모델 저장 완료! (MAE: {best_mae:.4f}) -> {model_save_path}\n")
        else:
            print()

if __name__ == "__main__":
    main()
