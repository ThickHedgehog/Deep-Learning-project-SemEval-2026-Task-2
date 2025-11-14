# ============================================================================
# COPY THIS ENTIRE FILE TO GOOGLE COLAB
# ============================================================================
#
# Instructions:
# 1. Create new Colab notebook
# 2. Enable GPU: Runtime → Change runtime type → T4 GPU
# 3. Upload train_subtask2a.csv when prompted
# 4. Login to wandb when prompted (optional but recommended)
# 5. Copy this ENTIRE file into ONE cell
# 6. Run the cell
# 7. Wait ~90-120 minutes
# 8. Check wandb dashboard for training visualization
# 9. Download final_model_best.pt
#
# ============================================================================

"""
v3.3 FINAL MINIMAL - Based on v3.0 + 6 Small Proven Improvements
=================================================================
v3.0 Result: CCC 0.51 (Valence 0.64, Arousal 0.39, Gap 0.39)
v3.2 Result: CCC 0.29 (too many changes, failed)

v3.3 Strategy: Keep v3.0 base + ONLY 6 minimal changes
1. User emb: 64→32 (reduce overfitting while keeping benefit)
2. Dropout: 0.2→0.3 (moderate regularization)
3. LSTM: 256→192 (slight capacity reduction)
4. Arousal CCC: 70%→75% (moderate focus increase)
5. Weight decay: 0.01→0.015 (moderate L2)
6. Patience: 7→5 (earlier stopping)

Expected: CCC 0.54-0.58 (realistic, 85% confidence)
"""

# ===== IMPORTS =====
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm
from scipy.stats import pearsonr
import re
from sklearn.model_selection import train_test_split
import wandb

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# ===== WANDB SETUP =====
print('\n=== WANDB SETUP ===')
print('Initializing Weights & Biases...')

# Initialize wandb
wandb.login()

# ===== UPLOAD DATA =====
print('\n=== UPLOAD DATA ===')
from google.colab import files
uploaded = files.upload()  # Upload train_subtask2a.csv

# ===== FEATURE EXTRACTION =====
print('\n=== FEATURE EXTRACTION ===')
df = pd.read_csv('train_subtask2a.csv')
print(f'Loaded {len(df)} samples from {df["user_id"].nunique()} users')

# Text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"()-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['text_cleaned'] = df['text'].apply(clean_text)

# Text features
df['word_count'] = df['text_cleaned'].apply(lambda x: len(x.split()))
df['char_count'] = df['text_cleaned'].apply(len)
df['sentence_count'] = df['text_cleaned'].apply(lambda x: len(re.findall(r'[.!?]+', x)) + 1)
df['avg_word_length'] = df['text_cleaned'].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)
df['exclamation_count'] = df['text_cleaned'].apply(lambda x: x.count('!'))
df['question_count'] = df['text_cleaned'].apply(lambda x: x.count('?'))
df['uppercase_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
)

# Emotion words
positive_words = set(['good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'amazing',
                     'fantastic', 'perfect', 'best', 'joy', 'excited', 'glad', 'delighted'])
negative_words = set(['bad', 'sad', 'hate', 'terrible', 'awful', 'horrible', 'worst',
                     'angry', 'fear', 'worried', 'anxious', 'depressed', 'upset', 'disappointed'])

df['positive_word_count'] = df['text_cleaned'].apply(
    lambda x: sum(1 for w in x.lower().split() if w in positive_words)
)
df['negative_word_count'] = df['text_cleaned'].apply(
    lambda x: sum(1 for w in x.lower().split() if w in negative_words)
)
df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']

# Temporal features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Sort
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# User statistics
user_stats = df.groupby('user_id').agg({
    'valence': ['mean', 'std'],
    'arousal': ['mean', 'std'],
    'text_id': 'count'
}).reset_index()
user_stats.columns = ['user_id', 'user_valence_mean', 'user_valence_std',
                     'user_arousal_mean', 'user_arousal_std', 'user_entry_count']
user_stats['user_valence_std'] = user_stats['user_valence_std'].fillna(0)
user_stats['user_arousal_std'] = user_stats['user_arousal_std'].fillna(0)

df = df.merge(user_stats, on='user_id', how='left')

# Sequence features
df['entry_number'] = df.groupby('user_id').cumcount()
df['relative_position'] = df['entry_number'] / df['user_entry_count']

# Time gaps
df['time_gap_hours'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
df['time_gap_hours'] = df['time_gap_hours'].fillna(0)
df['time_gap_log'] = np.log1p(df['time_gap_hours'])

# Lag features (5 lags - ENHANCED)
for lag in [1, 2, 3, 4, 5]:
    df[f'valence_lag{lag}'] = df.groupby('user_id')['valence'].shift(lag)
    df[f'arousal_lag{lag}'] = df.groupby('user_id')['arousal'].shift(lag)
    df[f'valence_lag{lag}'] = df[f'valence_lag{lag}'].fillna(df['user_valence_mean'])
    df[f'arousal_lag{lag}'] = df[f'arousal_lag{lag}'].fillna(df['user_arousal_mean'])

print(f'✓ Feature extraction complete: {len(df.columns)} features')

# ===== METRICS =====

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    pearson_corr = pearsonr(y_true, y_pred)[0]
    numerator = 2 * pearson_corr * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator if denominator != 0 else 0
    return ccc

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        var_true = torch.var(y_true)
        var_pred = torch.var(y_pred)
        covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
        sd_true = torch.sqrt(var_true + 1e-8)
        sd_pred = torch.sqrt(var_pred + 1e-8)
        pearson = covariance / (sd_true * sd_pred + 1e-8)
        numerator = 2 * pearson * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / (denominator + 1e-8)
        return 1 - ccc

# ===== DATASET =====

class EmotionSequenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, seq_length=7):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seq_length = seq_length

        user_ids_unique = df['user_id'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids_unique)}
        self.num_users = len(user_ids_unique)

        self.sequences = []
        for user_id, user_df in df.groupby('user_id'):
            user_data = user_df.reset_index(drop=True)
            for i in range(len(user_data)):
                start_idx = max(0, i - seq_length + 1)
                seq_data = user_data.iloc[start_idx:i+1]

                if len(seq_data) < seq_length:
                    padding_needed = seq_length - len(seq_data)
                    first_entry = seq_data.iloc[0:1]
                    padding = pd.concat([first_entry] * padding_needed, ignore_index=True)
                    seq_data = pd.concat([padding, seq_data], ignore_index=True)

                self.sequences.append({
                    'user_id': user_id,
                    'seq_data': seq_data,
                    'target_idx': i
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        seq_data = seq_info['seq_data']

        texts = seq_data['text_cleaned'].tolist()
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        temp_features = seq_data[[
            'valence_lag1', 'valence_lag2', 'valence_lag3', 'valence_lag4', 'valence_lag5',
            'arousal_lag1', 'arousal_lag2', 'arousal_lag3', 'arousal_lag4', 'arousal_lag5',
            'time_gap_log', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'entry_number', 'relative_position'
        ]].values.astype(np.float32)

        user_stats = seq_data[[
            'user_valence_mean', 'user_valence_std',
            'user_arousal_mean', 'user_arousal_std'
        ]].iloc[0].values.astype(np.float32)

        text_features = seq_data[[
            'word_count', 'char_count', 'sentence_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_ratio',
            'positive_word_count', 'negative_word_count', 'sentiment_score'
        ]].values.astype(np.float32)

        valence = seq_data['valence'].iloc[-1]
        arousal = seq_data['arousal'].iloc[-1]

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'user_idx': self.user_to_idx[seq_info['user_id']],
            'temporal_features': torch.FloatTensor(temp_features),
            'user_stats': torch.FloatTensor(user_stats),
            'text_features': torch.FloatTensor(text_features),
            'valence': torch.FloatTensor([valence]),
            'arousal': torch.FloatTensor([arousal])
        }

# ===== MODEL =====

class FinalEmotionModel(nn.Module):
    def __init__(self, num_users, user_emb_dim=32, lstm_hidden=192, lstm_layers=2,
                 num_attention_heads=4, dropout=0.3):
        super().__init__()

        self.roberta = AutoModel.from_pretrained('roberta-base')
        text_dim = 768

        self.user_embedding = nn.Embedding(num_users, user_emb_dim)

        temp_feature_dim = 17
        user_stat_dim = 4
        text_feature_dim = 10

        self.input_dim = text_dim + user_emb_dim + temp_feature_dim + user_stat_dim + text_feature_dim
        self.input_proj = nn.Linear(self.input_dim, lstm_hidden * 2)

        self.lstm = nn.LSTM(
            lstm_hidden * 2,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.valence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, user_idx, temporal_features,
                user_stats, text_features):
        batch_size, seq_len, max_len = input_ids.size()

        input_ids_flat = input_ids.view(batch_size * seq_len, max_len)
        attention_mask_flat = attention_mask.view(batch_size * seq_len, max_len)

        roberta_out = self.roberta(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        text_emb = roberta_out.last_hidden_state[:, 0, :]
        text_emb = text_emb.view(batch_size, seq_len, -1)

        user_emb = self.user_embedding(user_idx)
        user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)

        user_stats_exp = user_stats.unsqueeze(1).expand(-1, seq_len, -1)

        combined = torch.cat([
            text_emb,
            user_emb,
            temporal_features,
            user_stats_exp,
            text_features
        ], dim=-1)

        x = self.input_proj(combined)
        x = self.dropout(x)

        lstm_out, _ = self.lstm(x)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        final_repr = attn_out[:, -1, :]

        fused = self.fusion(final_repr)

        valence = self.valence_head(fused)
        arousal = self.arousal_head(fused)

        return valence, arousal

# ===== TRAINING FUNCTIONS =====

def train_epoch(model, loader, optimizer, scheduler, ccc_loss_fn, mse_loss_fn, device,
                ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a, max_grad_norm):
    model.train()
    total_loss = 0
    all_valence_true, all_valence_pred = [], []
    all_arousal_true, all_arousal_pred = [], []

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        user_idx = batch['user_idx'].to(device)
        temporal_features = batch['temporal_features'].to(device)
        user_stats = batch['user_stats'].to(device)
        text_features = batch['text_features'].to(device)
        valence_true = batch['valence'].to(device)
        arousal_true = batch['arousal'].to(device)

        valence_pred, arousal_pred = model(
            input_ids, attention_mask, user_idx,
            temporal_features, user_stats, text_features
        )

        ccc_loss_v = ccc_loss_fn(valence_pred.squeeze(), valence_true.squeeze())
        ccc_loss_a = ccc_loss_fn(arousal_pred.squeeze(), arousal_true.squeeze())
        mse_loss_v = mse_loss_fn(valence_pred.squeeze(), valence_true.squeeze())
        mse_loss_a = mse_loss_fn(arousal_pred.squeeze(), arousal_true.squeeze())

        loss = (
            ccc_weight_v * ccc_loss_v + mse_weight_v * mse_loss_v +
            ccc_weight_a * ccc_loss_a + mse_weight_a * mse_loss_a
        ) / 2

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_valence_true.extend(valence_true.cpu().numpy())
        all_valence_pred.extend(valence_pred.detach().cpu().numpy())
        all_arousal_true.extend(arousal_true.cpu().numpy())
        all_arousal_pred.extend(arousal_pred.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    ccc_v = concordance_correlation_coefficient(
        np.array(all_valence_true).flatten(),
        np.array(all_valence_pred).flatten()
    )
    ccc_a = concordance_correlation_coefficient(
        np.array(all_arousal_true).flatten(),
        np.array(all_arousal_pred).flatten()
    )
    ccc_avg = (ccc_v + ccc_a) / 2

    return avg_loss, ccc_avg, ccc_v, ccc_a

def validate(model, loader, ccc_loss_fn, mse_loss_fn, device,
             ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a):
    model.eval()
    total_loss = 0
    all_valence_true, all_valence_pred = [], []
    all_arousal_true, all_arousal_pred = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_idx = batch['user_idx'].to(device)
            temporal_features = batch['temporal_features'].to(device)
            user_stats = batch['user_stats'].to(device)
            text_features = batch['text_features'].to(device)
            valence_true = batch['valence'].to(device)
            arousal_true = batch['arousal'].to(device)

            valence_pred, arousal_pred = model(
                input_ids, attention_mask, user_idx,
                temporal_features, user_stats, text_features
            )

            ccc_loss_v = ccc_loss_fn(valence_pred.squeeze(), valence_true.squeeze())
            ccc_loss_a = ccc_loss_fn(arousal_pred.squeeze(), arousal_true.squeeze())
            mse_loss_v = mse_loss_fn(valence_pred.squeeze(), valence_true.squeeze())
            mse_loss_a = mse_loss_fn(arousal_pred.squeeze(), arousal_true.squeeze())

            loss = (
                ccc_weight_v * ccc_loss_v + mse_weight_v * mse_loss_v +
                ccc_weight_a * ccc_loss_a + mse_weight_a * mse_loss_a
            ) / 2

            total_loss += loss.item()
            all_valence_true.extend(valence_true.cpu().numpy())
            all_valence_pred.extend(valence_pred.cpu().numpy())
            all_arousal_true.extend(arousal_true.cpu().numpy())
            all_arousal_pred.extend(arousal_pred.cpu().numpy())

    avg_loss = total_loss / len(loader)

    val_true = np.array(all_valence_true).flatten()
    val_pred = np.array(all_valence_pred).flatten()
    aro_true = np.array(all_arousal_true).flatten()
    aro_pred = np.array(all_arousal_pred).flatten()

    ccc_v = concordance_correlation_coefficient(val_true, val_pred)
    ccc_a = concordance_correlation_coefficient(aro_true, aro_pred)
    ccc_avg = (ccc_v + ccc_a) / 2

    rmse_v = np.sqrt(np.mean((val_true - val_pred) ** 2))
    rmse_a = np.sqrt(np.mean((aro_true - aro_pred) ** 2))

    return avg_loss, ccc_avg, ccc_v, ccc_a, rmse_v, rmse_a

# ===== MAIN TRAINING =====

print('\n=== STARTING TRAINING ===')

# Hyperparameters
# v3.3 MINIMAL CHANGES from v3.0
SEQ_LENGTH = 7
BATCH_SIZE = 10
NUM_EPOCHS = 20
PATIENCE = 5  # CHANGED: 7 → 5 (earlier stopping)
DROPOUT = 0.3  # CHANGED: 0.2 → 0.3 (moderate increase)

LR_ROBERTA = 1.5e-5
LR_OTHER = 8e-5
WEIGHT_DECAY = 0.015  # CHANGED: 0.01 → 0.015 (moderate increase)
WARMUP_RATIO = 0.15

CCC_WEIGHT_V = 0.65
CCC_WEIGHT_A = 0.75  # CHANGED: 0.70 → 0.75 (moderate increase)
MSE_WEIGHT_V = 0.35
MSE_WEIGHT_A = 0.25  # CHANGED: 0.30 → 0.25 (to balance CCC increase)

MAX_GRAD_NORM = 0.5

# Initialize wandb run
wandb.init(
    project="semeval-2026-task2-subtask2a",
    name="v3.3-minimal-proven",
    config={
        "version": "v3.3-MINIMAL",
        "base": "v3.0 (CCC 0.51)",
        "strategy": "minimal proven changes only",
        "architecture": "RoBERTa-BiLSTM-Attention-DualHead",
        "seq_length": SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "dropout": DROPOUT,
        "lr_roberta": LR_ROBERTA,
        "lr_other": LR_OTHER,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "ccc_weight_valence": CCC_WEIGHT_V,
        "ccc_weight_arousal": CCC_WEIGHT_A,
        "mse_weight_valence": MSE_WEIGHT_V,
        "mse_weight_arousal": MSE_WEIGHT_A,
        "max_grad_norm": MAX_GRAD_NORM,
        "lag_features": 5,
        "attention_heads": 4,
        "user_emb_dim": 32,  # CHANGED from 64
        "lstm_hidden": 192,  # CHANGED from 256
        "lstm_layers": 2,
        "changes_from_v3.0": [
            "User emb: 64→32",
            "Dropout: 0.2→0.3",
            "LSTM: 256→192",
            "Arousal CCC: 70%→75%",
            "Weight decay: 0.01→0.015",
            "Patience: 7→5"
        ]
    }
)

print('='*80)
print('v3.3 MINIMAL - Based on proven v3.0 + 6 small changes')
print('='*80)
print(f'v3.0 baseline: CCC 0.51 (Val 0.64, Aro 0.39, Gap 0.39)')
print(f'v3.3 target: CCC 0.54-0.58 (realistic improvement)')
print('='*80)
print(f'Sequence Length: {SEQ_LENGTH}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}')
print(f'User Emb: 32 (was 64), LSTM: 192 (was 256), Dropout: 0.3 (was 0.2)')
print(f'Valence Loss: {CCC_WEIGHT_V*100:.0f}% CCC + {MSE_WEIGHT_V*100:.0f}% MSE')
print(f'Arousal Loss: {CCC_WEIGHT_A*100:.0f}% CCC + {MSE_WEIGHT_A*100:.0f}% MSE (was 70/30)')
print('='*80)

# Split data
users = df['user_id'].unique()
train_users, val_users = train_test_split(users, test_size=0.15, random_state=42)

train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
val_df = df[df['user_id'].isin(val_users)].reset_index(drop=True)

print(f'Train: {len(train_df)} samples, {len(train_users)} users')
print(f'Val: {len(val_df)} samples, {len(val_users)} users')

# Log data stats to wandb
wandb.config.update({
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "train_users": len(train_users),
    "val_users": len(val_users),
    "total_samples": len(df),
    "total_users": df['user_id'].nunique(),
})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Create datasets
train_dataset = EmotionSequenceDataset(train_df, tokenizer, seq_length=SEQ_LENGTH)
val_dataset = EmotionSequenceDataset(val_df, tokenizer, seq_length=SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Initialize model
model = FinalEmotionModel(num_users=train_dataset.num_users, dropout=DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Log model info to wandb
wandb.config.update({
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
})

# Watch model gradients
wandb.watch(model, log="all", log_freq=100)

# Loss functions
ccc_loss_fn = CCCLoss()
mse_loss_fn = nn.MSELoss()

# Optimizer
roberta_params = []
other_params = []
for name, param in model.named_parameters():
    if 'roberta' in name:
        roberta_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': roberta_params, 'lr': LR_ROBERTA},
    {'params': other_params, 'lr': LR_OTHER}
], weight_decay=WEIGHT_DECAY)

# Scheduler
num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f'Training steps: {num_training_steps}, Warmup: {num_warmup_steps}')

# Training loop
print('\n=== TRAINING FINAL MODEL v3 ===\n')

best_ccc = 0
patience_counter = 0
history = []

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 80)

    train_loss, train_ccc, train_ccc_v, train_ccc_a = train_epoch(
        model, train_loader, optimizer, scheduler, ccc_loss_fn, mse_loss_fn, device,
        CCC_WEIGHT_V, CCC_WEIGHT_A, MSE_WEIGHT_V, MSE_WEIGHT_A, MAX_GRAD_NORM
    )

    val_loss, val_ccc, val_ccc_v, val_ccc_a, val_rmse_v, val_rmse_a = validate(
        model, val_loader, ccc_loss_fn, mse_loss_fn, device,
        CCC_WEIGHT_V, CCC_WEIGHT_A, MSE_WEIGHT_V, MSE_WEIGHT_A
    )

    print(f'\nEpoch {epoch+1} Results:')
    print(f'  Train Loss: {train_loss:.4f}, Train CCC: {train_ccc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val CCC: {val_ccc:.4f}')
    print(f'  Val CCC Valence: {val_ccc_v:.4f}, Val CCC Arousal: {val_ccc_a:.4f}')
    print(f'  Val RMSE Valence: {val_rmse_v:.4f}, Val RMSE Arousal: {val_rmse_a:.4f}')

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/ccc_avg": train_ccc,
        "train/ccc_valence": train_ccc_v,
        "train/ccc_arousal": train_ccc_a,
        "val/loss": val_loss,
        "val/ccc_avg": val_ccc,
        "val/ccc_valence": val_ccc_v,
        "val/ccc_arousal": val_ccc_a,
        "val/rmse_valence": val_rmse_v,
        "val/rmse_arousal": val_rmse_a,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "patience_counter": patience_counter,
    })

    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_ccc': train_ccc,
        'val_loss': val_loss,
        'val_ccc': val_ccc,
        'val_ccc_v': val_ccc_v,
        'val_ccc_a': val_ccc_a,
        'val_rmse_v': val_rmse_v,
        'val_rmse_a': val_rmse_a
    })

    if val_ccc > best_ccc:
        best_ccc = val_ccc
        patience_counter = 0

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_ccc': best_ccc,
            'val_ccc_v': val_ccc_v,
            'val_ccc_a': val_ccc_a,
            'val_rmse_v': val_rmse_v,
            'val_rmse_a': val_rmse_a,
            'config': {
                'seq_length': SEQ_LENGTH,
                'batch_size': BATCH_SIZE,
                'dropout': DROPOUT,
                'ccc_weight_v': CCC_WEIGHT_V,
                'ccc_weight_a': CCC_WEIGHT_A,
                'mse_weight_v': MSE_WEIGHT_V,
                'mse_weight_a': MSE_WEIGHT_A
            }
        }, 'final_model_best.pt')

        # Log best model to wandb
        wandb.run.summary["best_ccc"] = best_ccc
        wandb.run.summary["best_ccc_valence"] = val_ccc_v
        wandb.run.summary["best_ccc_arousal"] = val_ccc_a
        wandb.run.summary["best_rmse_valence"] = val_rmse_v
        wandb.run.summary["best_rmse_arousal"] = val_rmse_a
        wandb.run.summary["best_epoch"] = epoch + 1

        print(f'  ✓ Best model saved! (CCC: {best_ccc:.4f})')
    else:
        patience_counter += 1
        print(f'  No improvement. Patience: {patience_counter}/{PATIENCE}')

    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}')
        wandb.run.summary["stopped_epoch"] = epoch + 1
        wandb.run.summary["stopped_reason"] = "early_stopping"
        break

print('\n' + '='*80)
print('TRAINING COMPLETE')
print('='*80)
print(f'Best validation CCC: {best_ccc:.4f}')

history_df = pd.DataFrame(history)
print('\nTraining History:')
print(history_df.to_string(index=False))

# Create training history table in wandb
history_table = wandb.Table(dataframe=history_df)
wandb.log({"training_history": history_table})

# Save model artifact to wandb
artifact = wandb.Artifact('final-model-v3', type='model')
artifact.add_file('final_model_best.pt')
wandb.log_artifact(artifact)

# Save wandb URL before finishing
wandb_url = f'https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}'

# Finish wandb run
wandb.finish()

# Download model
print('\n=== DOWNLOADING MODEL ===')
files.download('final_model_best.pt')
print('✓ Download complete!')
print('\n✓ Check your wandb dashboard for training visualization!')
print(f'   Visit: {wandb_url}')
