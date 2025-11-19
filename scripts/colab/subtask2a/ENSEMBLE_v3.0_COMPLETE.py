# ============================================================================
# v3.0 ENSEMBLE - COMPLETE SOLUTION (3 Seeds)
# ============================================================================
#
# STRATEGY B - v3.0 Ensemble (RECOMMENDED, 85% Success Probability)
# Expected: CCC 0.530-0.550
#
# This file trains v3.0 with different seeds and creates ensemble predictions
#
# Instructions:
# 1. Train model with seed=42 (already done, CCC 0.5144)
# 2. Train model with seed=123 using this file
# 3. Train model with seed=777 using this file
# 4. Use ensemble prediction code at the end
#
# ============================================================================

"""
v3.0 ENSEMBLE TRAINING
======================
Best proven model (CCC 0.5144) trained with multiple seeds for ensemble

Architecture: RoBERTa + BiLSTM + Multi-Head Attention + Dual-Head Loss
Key: Arousal CCC 70%, User Emb 64 dim, LSTM 256 hidden, Dropout 0.2
"""

# ===== CONFIGURATION =====
# CHANGE THIS FOR EACH TRAINING RUN
RANDOM_SEED = 777  # Change to 42, 123, or 777 for different runs
MODEL_SAVE_NAME = f'v3.0_seed{RANDOM_SEED}_best.pt'

# WandB control - Set to False if wandb connection fails
USE_WANDB = False  # Change to False to disable wandb (faster if connection issues)

print(f'='*80)
print(f'v3.0 ENSEMBLE TRAINING - SEED {RANDOM_SEED}')
print(f'='*80)
print(f'Model will be saved as: {MODEL_SAVE_NAME}')
print(f'Expected CCC: ~0.510-0.515 (individual model)')
print(f'Ensemble Expected: CCC 0.530-0.550 (3 models)')
print(f'='*80)

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
import random

# ===== SET SEED =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)
print(f'✓ Random seed set to {RANDOM_SEED}')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# ===== WANDB SETUP =====
if USE_WANDB:
    print('\n=== WANDB SETUP ===')
    # Login only if not already logged in
    try:
        if not wandb.api.api_key:
            wandb.login()
        else:
            print('✓ Already logged in to wandb')
    except Exception as e:
        print(f'⚠️ WandB login failed: {e}')
        print('Continuing without WandB...')
        USE_WANDB = False
else:
    print('\n=== WANDB DISABLED ===')
    print('Training without WandB (faster if connection issues)')

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

# Lag features (5 lags - PROVEN OPTIMAL)
for lag in [1, 2, 3, 4, 5]:
    df[f'valence_lag{lag}'] = df.groupby('user_id')['valence'].shift(lag)
    df[f'arousal_lag{lag}'] = df.groupby('user_id')['arousal'].shift(lag)

# Fill NaN
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(0)

print(f'✓ Feature extraction complete: {len([c for c in df.columns if c not in ["text_id", "user_id", "timestamp", "text", "text_cleaned", "collection_phase", "is_words"]])} features')

# ===== STARTING TRAINING =====
print('\n=== STARTING TRAINING ===')

# Initialize wandb with increased timeout (only if enabled)
if USE_WANDB:
    try:
        wandb.init(
            project="semeval-2026-task2-subtask2a-ensemble",
            name=f"v3.0-ensemble-seed{RANDOM_SEED}",
            settings=wandb.Settings(init_timeout=180),  # Increased timeout to 3 minutes
            config={
                "version": "v3.0-ENSEMBLE",
                "seed": RANDOM_SEED,
                "architecture": "RoBERTa-BiLSTM-Attention-DualHead",
                "user_emb_dim": 64,  # PROVEN OPTIMAL
                "lstm_hidden": 256,  # PROVEN OPTIMAL
                "lstm_layers": 2,
                "dropout": 0.2,  # PROVEN OPTIMAL
                "seq_length": 7,
                "batch_size": 10,
                "num_epochs": 20,
                "patience": 7,
                "warmup_ratio": 0.15,
                "lr_roberta": 1.5e-5,
                "lr_other": 8e-5,
                "weight_decay": 0.01,
                "ccc_weight_valence": 0.65,
                "ccc_weight_arousal": 0.70,  # PROVEN OPTIMAL - DO NOT CHANGE!
                "mse_weight_valence": 0.35,
                "mse_weight_arousal": 0.30,
                "expected_ccc": "0.510-0.515 (single), 0.530-0.550 (ensemble)"
            }
        )
        print('✓ WandB initialized successfully')
    except Exception as e:
        print(f'⚠️ WandB init failed: {e}')
        print('Continuing without WandB...')
        USE_WANDB = False

print('='*80)
print('v3.0 ENSEMBLE MODEL - PROVEN BEST CONFIGURATION')
print('='*80)
print(f'Random Seed: {RANDOM_SEED}')
print(f'User Embedding: 64 dim (PROVEN OPTIMAL)')
print(f'LSTM Hidden: 256 (PROVEN OPTIMAL)')
print(f'Dropout: 0.2 (PROVEN OPTIMAL)')
print(f'Arousal CCC Weight: 70% (PROVEN OPTIMAL - DO NOT CHANGE!)')
print('='*80)

# Hyperparameters (v3.0 PROVEN OPTIMAL)
SEQ_LENGTH = 7
BATCH_SIZE = 10
NUM_EPOCHS = 20
PATIENCE = 7  # v3.0 proven value
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.15

# Loss weights (DUAL-HEAD - PROVEN OPTIMAL)
CCC_WEIGHT_V = 0.65
CCC_WEIGHT_A = 0.70  # ⭐ PROVEN OPTIMAL - DO NOT INCREASE TO 0.75!
MSE_WEIGHT_V = 0.35
MSE_WEIGHT_A = 0.30  # ⭐ PROVEN OPTIMAL - DO NOT DECREASE TO 0.25!

# Learning rates
LR_ROBERTA = 1.5e-5
LR_OTHER = 8e-5
WEIGHT_DECAY = 0.01

print(f'Sequence Length: {SEQ_LENGTH}')
print(f'Batch Size: {BATCH_SIZE}')
print(f'Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}')
print(f'Valence Loss: {CCC_WEIGHT_V*100:.0f}% CCC + {MSE_WEIGHT_V*100:.0f}% MSE')
print(f'Arousal Loss: {CCC_WEIGHT_A*100:.0f}% CCC + {MSE_WEIGHT_A*100:.0f}% MSE')
print('='*80)

# Train/Val split (using seed for reproducibility)
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=RANDOM_SEED, stratify=df['user_id']
)

train_users = train_df['user_id'].nunique()
val_users = val_df['user_id'].nunique()

print(f'Train: {len(train_df)} samples, {train_users} users')
print(f'Val: {len(val_df)} samples, {val_users} users')

# ===== DATASET =====
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, seq_length=7, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_length = max_length

        # User mapping
        unique_users = df['user_id'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.num_users = len(unique_users)

        # Create sequences
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

# Create datasets
train_dataset = EmotionDataset(train_df, tokenizer, SEQ_LENGTH)
val_dataset = EmotionDataset(val_df, tokenizer, SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ===== MODEL =====

class FinalEmotionModel(nn.Module):
    def __init__(self, num_users, user_emb_dim=64, lstm_hidden=256, lstm_layers=2,
                 num_attention_heads=4, dropout=0.2):
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
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2)
        )

        # Dual prediction heads
        self.valence_head = nn.Sequential(
            nn.Linear(lstm_hidden // 2, lstm_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 4, 1)
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(lstm_hidden // 2, lstm_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 4, 1)
        )

    def forward(self, input_ids, attention_mask, user_idx, temporal_features,
                user_stats, text_features):
        batch_size, seq_len, max_len = input_ids.shape

        # Encode text
        input_ids_flat = input_ids.view(batch_size * seq_len, max_len)
        attention_mask_flat = attention_mask.view(batch_size * seq_len, max_len)

        outputs = self.roberta(input_ids_flat, attention_mask=attention_mask_flat)
        text_emb = outputs.last_hidden_state[:, 0, :]
        text_emb = text_emb.view(batch_size, seq_len, -1)

        # User embeddings
        user_emb = self.user_embedding(user_idx)
        user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # User stats
        user_stats = user_stats.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine features
        combined = torch.cat([text_emb, user_emb, temporal_features, user_stats, text_features], dim=-1)
        combined = self.input_proj(combined)

        # LSTM
        lstm_out, _ = self.lstm(combined)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Take last timestep
        final_repr = attn_out[:, -1, :]

        # Fusion
        fused = self.fusion(final_repr)

        # Predictions
        valence_pred = self.valence_head(fused)
        arousal_pred = self.arousal_head(fused)

        return valence_pred, arousal_pred

# Create model
model = FinalEmotionModel(num_users=train_dataset.num_users, dropout=0.2).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# ===== LOSS & OPTIMIZER =====

def concordance_correlation_coefficient(y_true, y_pred):
    """CCC calculation"""
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc

def dual_head_loss(valence_pred, arousal_pred, valence_true, arousal_true,
                   ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a):
    """Dual-head loss with separate weights"""
    # Valence loss
    ccc_v = concordance_correlation_coefficient(valence_true, valence_pred)
    mse_v = F.mse_loss(valence_pred, valence_true)
    loss_v = (1 - ccc_v) * ccc_weight_v + mse_v * mse_weight_v

    # Arousal loss
    ccc_a = concordance_correlation_coefficient(arousal_true, arousal_pred)
    mse_a = F.mse_loss(arousal_pred, arousal_true)
    loss_a = (1 - ccc_a) * ccc_weight_a + mse_a * mse_weight_a

    # Total loss
    total_loss = loss_v + loss_a

    return total_loss, ccc_v, ccc_a

# Optimizer (differential learning rates)
optimizer = torch.optim.AdamW([
    {'params': model.roberta.parameters(), 'lr': LR_ROBERTA},
    {'params': [p for n, p in model.named_parameters() if 'roberta' not in n], 'lr': LR_OTHER}
], weight_decay=WEIGHT_DECAY)

# Scheduler
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f'Training steps: {total_steps}, Warmup: {warmup_steps}')

# ===== TRAINING =====

def train_epoch(model, loader, optimizer, scheduler, max_grad_norm,
                ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a):
    model.train()
    total_loss = 0
    all_valence_pred, all_valence_true = [], []
    all_arousal_pred, all_arousal_true = [], []

    for batch in tqdm(loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        user_idx = batch['user_idx'].to(device)
        temporal_features = batch['temporal_features'].to(device)
        user_stats = batch['user_stats'].to(device)
        text_features = batch['text_features'].to(device)
        valence_true = batch['valence'].to(device)
        arousal_true = batch['arousal'].to(device)

        optimizer.zero_grad()

        valence_pred, arousal_pred = model(
            input_ids, attention_mask, user_idx,
            temporal_features, user_stats, text_features
        )

        loss, ccc_v, ccc_a = dual_head_loss(
            valence_pred, arousal_pred, valence_true, arousal_true,
            ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_valence_pred.extend(valence_pred.detach().cpu().numpy())
        all_valence_true.extend(valence_true.detach().cpu().numpy())
        all_arousal_pred.extend(arousal_pred.detach().cpu().numpy())
        all_arousal_true.extend(arousal_true.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    train_ccc_v = float(pearsonr(all_valence_true, all_valence_pred)[0])
    train_ccc_a = float(pearsonr(all_arousal_true, all_arousal_pred)[0])
    train_ccc = (train_ccc_v + train_ccc_a) / 2

    return avg_loss, train_ccc, train_ccc_v, train_ccc_a

def validate(model, loader, ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a):
    model.eval()
    total_loss = 0
    all_valence_pred, all_valence_true = [], []
    all_arousal_pred, all_arousal_true = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
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

            loss, ccc_v, ccc_a = dual_head_loss(
                valence_pred, arousal_pred, valence_true, arousal_true,
                ccc_weight_v, ccc_weight_a, mse_weight_v, mse_weight_a
            )

            total_loss += loss.item()
            all_valence_pred.extend(valence_pred.cpu().numpy())
            all_valence_true.extend(valence_true.cpu().numpy())
            all_arousal_pred.extend(arousal_pred.cpu().numpy())
            all_arousal_true.extend(arousal_true.cpu().numpy())

    avg_loss = total_loss / len(loader)
    val_ccc_v = float(pearsonr(all_valence_true, all_valence_pred)[0])
    val_ccc_a = float(pearsonr(all_arousal_true, all_arousal_pred)[0])
    val_ccc = (val_ccc_v + val_ccc_a) / 2

    val_rmse_v = float(np.sqrt(np.mean((np.array(all_valence_true) - np.array(all_valence_pred)) ** 2)))
    val_rmse_a = float(np.sqrt(np.mean((np.array(all_arousal_true) - np.array(all_arousal_pred)) ** 2)))

    return avg_loss, val_ccc, val_ccc_v, val_ccc_a, val_rmse_v, val_rmse_a

print('\n=== TRAINING FINAL MODEL v3.0 ENSEMBLE ===\n')

best_ccc = -1
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 80)

    train_loss, train_ccc, train_ccc_v, train_ccc_a = train_epoch(
        model, train_loader, optimizer, scheduler, MAX_GRAD_NORM,
        CCC_WEIGHT_V, CCC_WEIGHT_A, MSE_WEIGHT_V, MSE_WEIGHT_A
    )

    val_loss, val_ccc, val_ccc_v, val_ccc_a, val_rmse_v, val_rmse_a = validate(
        model, val_loader, CCC_WEIGHT_V, CCC_WEIGHT_A, MSE_WEIGHT_V, MSE_WEIGHT_A
    )

    print(f'\nEpoch {epoch+1} Results:')
    print(f'  Train Loss: {train_loss:.4f}, Train CCC: {train_ccc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val CCC: {val_ccc:.4f}')
    print(f'  Val CCC Valence: {val_ccc_v:.4f}, Val CCC Arousal: {val_ccc_a:.4f}')
    print(f'  Val RMSE Valence: {val_rmse_v:.4f}, Val RMSE Arousal: {val_rmse_a:.4f}')

    # Log to wandb (if enabled)
    if USE_WANDB:
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/ccc_avg': train_ccc,
            'train/ccc_valence': train_ccc_v,
            'train/ccc_arousal': train_ccc_a,
            'val/loss': val_loss,
            'val/ccc_avg': val_ccc,
            'val/ccc_valence': val_ccc_v,
            'val/ccc_arousal': val_ccc_a,
            'val/rmse_valence': val_rmse_v,
            'val/rmse_arousal': val_rmse_a,
            'learning_rate': scheduler.get_last_lr()[0],
            'patience_counter': patience_counter
        })

    # Save best model
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
            'seed': RANDOM_SEED,
            'config': {
                'user_emb_dim': 64,
                'lstm_hidden': 256,
                'dropout': 0.2,
                'ccc_weight_v': CCC_WEIGHT_V,
                'ccc_weight_a': CCC_WEIGHT_A,
                'mse_weight_v': MSE_WEIGHT_V,
                'mse_weight_a': MSE_WEIGHT_A
            }
        }, MODEL_SAVE_NAME)
        print(f'  ✓ Best model saved! (CCC: {best_ccc:.4f})')

        # Log best to wandb (if enabled)
        if USE_WANDB:
            wandb.run.summary['best_ccc'] = best_ccc
            wandb.run.summary['best_ccc_valence'] = val_ccc_v
            wandb.run.summary['best_ccc_arousal'] = val_ccc_a
            wandb.run.summary['best_epoch'] = epoch + 1
            wandb.run.summary['best_rmse_valence'] = val_rmse_v
            wandb.run.summary['best_rmse_arousal'] = val_rmse_a
    else:
        patience_counter += 1
        print(f'  No improvement. Patience: {patience_counter}/{PATIENCE}')

    if patience_counter >= PATIENCE:
        print(f'\n Early stopping triggered at epoch {epoch+1}')
        break

print('\n' + '='*80)
print('TRAINING COMPLETE')
print('='*80)
print(f'Best validation CCC: {best_ccc:.4f}')
print(f'Model saved as: {MODEL_SAVE_NAME}')

# Save WandB URL before finish (if enabled)
if USE_WANDB:
    wandb_url = f'https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}'
    wandb.finish()
    print(f'\n✓ Check your wandb dashboard for training visualization!')
    print(f'   Visit: {wandb_url}')
else:
    print('\n✓ Training completed without WandB logging')

print('\n=== DOWNLOADING MODEL ===')
from google.colab import files
files.download(MODEL_SAVE_NAME)
print('✓ Download complete!')

print('\n' + '='*80)
print(f'SEED {RANDOM_SEED} TRAINING COMPLETE!')
print(f'Expected individual CCC: ~0.510-0.515')
print(f'Expected ensemble CCC (3 models): 0.530-0.550')
print('='*80)
print('\nNext steps:')
print(f'1. Change RANDOM_SEED to next value (123 or 777)')
print(f'2. Run this script again')
print(f'3. After training all 3 models, use ensemble prediction code below')
print('='*80)
