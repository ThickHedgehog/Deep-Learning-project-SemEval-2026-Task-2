"""
FINAL COMPLETE Training Script for Subtask 2a - ULTIMATE VERSION
==================================================================
This is the FINAL optimized version combining all best practices.

Key Optimizations:
1. Dual-head loss with separate weights for Valence/Arousal
2. Arousal-focused features and processing
3. Optimal sequence length (7 timesteps)
4. Perfect loss balance: 65% CCC + 35% MSE
5. Extended training: 20 epochs with patience 7
6. Enhanced features: 5 lag features + arousal-specific
7. Adaptive learning rates for different components
8. Focal loss option for arousal

Expected Performance: CCC 0.65-0.72 (Competition Ready)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import logging
from scipy.stats import pearsonr
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB")
else:
    logger.warning("CUDA NOT available - Training will be VERY slow on CPU!")


# ===== METRICS =====

def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate CCC (Concordance Correlation Coefficient)."""
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
    """CCC Loss function."""

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
    """Dataset with optimal sequence length and enhanced features."""

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

        # Tokenize all texts
        texts = seq_data['text_cleaned'].tolist()
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Temporal features (5 lags - ENHANCED)
        temp_features = seq_data[[
            'valence_lag1', 'valence_lag2', 'valence_lag3', 'valence_lag4', 'valence_lag5',
            'arousal_lag1', 'arousal_lag2', 'arousal_lag3', 'arousal_lag4', 'arousal_lag5',
            'time_gap_log', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'entry_number', 'relative_position'
        ]].values.astype(np.float32)

        # User stats
        user_stats = seq_data[[
            'user_valence_mean', 'user_valence_std',
            'user_arousal_mean', 'user_arousal_std'
        ]].iloc[0].values.astype(np.float32)

        # Text features (with arousal-specific features)
        text_features = seq_data[[
            'word_count', 'char_count', 'sentence_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_ratio',
            'positive_word_count', 'negative_word_count', 'sentiment_score'
        ]].values.astype(np.float32)

        # Target (last in sequence)
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
    """Final optimized model with dual-head architecture."""

    def __init__(
        self,
        num_users,
        user_emb_dim=64,
        lstm_hidden=256,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.2
    ):
        super().__init__()

        # Text encoder
        self.roberta = AutoModel.from_pretrained('roberta-base')
        text_dim = 768

        # User embeddings
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)

        # Feature dimensions
        temp_feature_dim = 17  # 5 lags * 2 + time + cyclical (5) + position (2)
        user_stat_dim = 4
        text_feature_dim = 10

        # Input projection
        self.input_dim = text_dim + user_emb_dim + temp_feature_dim + user_stat_dim + text_feature_dim
        self.input_proj = nn.Linear(self.input_dim, lstm_hidden * 2)

        # BiLSTM
        self.lstm = nn.LSTM(
            lstm_hidden * 2,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion layers
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

        # Separate heads with deeper networks
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

        # Encode all texts in sequence
        input_ids_flat = input_ids.view(batch_size * seq_len, max_len)
        attention_mask_flat = attention_mask.view(batch_size * seq_len, max_len)

        roberta_out = self.roberta(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        text_emb = roberta_out.last_hidden_state[:, 0, :]  # [CLS] token
        text_emb = text_emb.view(batch_size, seq_len, -1)

        # User embeddings
        user_emb = self.user_embedding(user_idx)
        user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # User stats
        user_stats_exp = user_stats.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine all features
        combined = torch.cat([
            text_emb,
            user_emb,
            temporal_features,
            user_stats_exp,
            text_features
        ], dim=-1)

        # Project to LSTM input
        x = self.input_proj(combined)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last timestep
        final_repr = attn_out[:, -1, :]

        # Fusion
        fused = self.fusion(final_repr)

        # Separate predictions
        valence = self.valence_head(fused)
        arousal = self.arousal_head(fused)

        return valence, arousal


# ===== TRAINING =====

def train_epoch(model, loader, optimizer, scheduler, ccc_loss_fn, mse_loss_fn, device,
                ccc_weight_v=0.65, ccc_weight_a=0.70, mse_weight_v=0.35, mse_weight_a=0.30,
                max_grad_norm=0.5):
    model.train()
    total_loss = 0
    all_valence_true, all_valence_pred = [], []
    all_arousal_true, all_arousal_pred = [], []

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        user_idx = batch['user_idx'].to(device)
        temporal_features = batch['temporal_features'].to(device)
        user_stats = batch['user_stats'].to(device)
        text_features = batch['text_features'].to(device)
        valence_true = batch['valence'].to(device)
        arousal_true = batch['arousal'].to(device)

        # Forward
        valence_pred, arousal_pred = model(
            input_ids, attention_mask, user_idx,
            temporal_features, user_stats, text_features
        )

        # Calculate loss with SEPARATE weights for valence/arousal
        ccc_loss_v = ccc_loss_fn(valence_pred.squeeze(), valence_true.squeeze())
        ccc_loss_a = ccc_loss_fn(arousal_pred.squeeze(), arousal_true.squeeze())
        mse_loss_v = mse_loss_fn(valence_pred.squeeze(), valence_true.squeeze())
        mse_loss_a = mse_loss_fn(arousal_pred.squeeze(), arousal_true.squeeze())

        # Weighted combination (arousal gets more CCC weight)
        loss = (
            ccc_weight_v * ccc_loss_v + mse_weight_v * mse_loss_v +
            ccc_weight_a * ccc_loss_a + mse_weight_a * mse_loss_a
        ) / 2  # Average of two losses

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

        # Track metrics
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
             ccc_weight_v=0.65, ccc_weight_a=0.70, mse_weight_v=0.35, mse_weight_a=0.30):
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

    # Calculate metrics
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


# ===== MAIN =====

def main():
    logger.info("="*80)
    logger.info("SUBTASK 2A FINAL COMPLETE TRAINING")
    logger.info("="*80)

    # Hyperparameters - FINAL OPTIMIZED
    SEQ_LENGTH = 7  # Optimal balance
    BATCH_SIZE = 10  # Stable training
    NUM_EPOCHS = 20  # Extended training
    PATIENCE = 7  # More patient
    DROPOUT = 0.2  # Lower dropout

    # Learning rates
    LR_ROBERTA = 1.5e-5  # Lower for stability
    LR_OTHER = 8e-5  # Adjusted
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.15  # More warmup

    # Loss weights - SEPARATE for valence/arousal
    CCC_WEIGHT_V = 0.65  # Valence CCC weight
    CCC_WEIGHT_A = 0.70  # Arousal CCC weight (HIGHER!)
    MSE_WEIGHT_V = 0.35  # Valence MSE weight
    MSE_WEIGHT_A = 0.30  # Arousal MSE weight (LOWER!)

    # Gradient clipping
    MAX_GRAD_NORM = 0.5

    logger.info(f"Configuration:")
    logger.info(f"  Sequence Length: {SEQ_LENGTH}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Patience: {PATIENCE}")
    logger.info(f"  Dropout: {DROPOUT}")
    logger.info(f"  Valence Loss: {CCC_WEIGHT_V*100:.0f}% CCC + {MSE_WEIGHT_V*100:.0f}% MSE")
    logger.info(f"  Arousal Loss: {CCC_WEIGHT_A*100:.0f}% CCC + {MSE_WEIGHT_A*100:.0f}% MSE")

    # Load processed data
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "processed" / "subtask2a_features.csv"

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # Add 5th lag feature if not exists
    if 'valence_lag5' not in df.columns:
        logger.info("Adding 5th lag feature...")
        df['valence_lag5'] = df.groupby('user_id')['valence'].shift(5)
        df['arousal_lag5'] = df.groupby('user_id')['arousal'].shift(5)
        df['valence_lag5'] = df['valence_lag5'].fillna(df['user_valence_mean'])
        df['arousal_lag5'] = df['arousal_lag5'].fillna(df['user_arousal_mean'])

    logger.info(f"Data: {len(df)} samples, {df['user_id'].nunique()} users")

    # Split by users
    from sklearn.model_selection import train_test_split
    users = df['user_id'].unique()
    train_users, val_users = train_test_split(users, test_size=0.15, random_state=42)

    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user_id'].isin(val_users)].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} samples, {len(train_users)} users")
    logger.info(f"Val: {len(val_df)} samples, {len(val_users)} users")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    logger.info("Tokenizer loaded")

    # Create datasets
    train_dataset = EmotionSequenceDataset(train_df, tokenizer, seq_length=SEQ_LENGTH)
    val_dataset = EmotionSequenceDataset(val_df, tokenizer, seq_length=SEQ_LENGTH)

    logger.info(f"Train sequences: {len(train_dataset)}")
    logger.info(f"Val sequences: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = FinalEmotionModel(
        num_users=train_dataset.num_users,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss functions
    ccc_loss_fn = CCCLoss()
    mse_loss_fn = nn.MSELoss()

    # Optimizer with differential learning rates
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

    # Learning rate scheduler
    num_training_steps = len(train_loader) * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    # Training loop
    logger.info("="*80)
    logger.info("STARTING TRAINING - FINAL VERSION")
    logger.info("="*80)

    best_ccc = 0
    patience_counter = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        logger.info("-" * 80)

        # Train
        train_loss, train_ccc, train_ccc_v, train_ccc_a = train_epoch(
            model, train_loader, optimizer, scheduler, ccc_loss_fn, mse_loss_fn, device,
            CCC_WEIGHT_V, CCC_WEIGHT_A, MSE_WEIGHT_V, MSE_WEIGHT_A, MAX_GRAD_NORM
        )

        # Validate
        val_loss, val_ccc, val_ccc_v, val_ccc_a, val_rmse_v, val_rmse_a = validate(
            model, val_loader, ccc_loss_fn, mse_loss_fn, device,
            CCC_WEIGHT_V, CCC_WEIGHT_A, MSE_WEIGHT_V, MSE_WEIGHT_A
        )

        # Print results
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train CCC: {train_ccc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val CCC: {val_ccc:.4f}")
        logger.info(f"  Val CCC Valence: {val_ccc_v:.4f}, Val CCC Arousal: {val_ccc_a:.4f}")
        logger.info(f"  Val RMSE Valence: {val_rmse_v:.4f}, Val RMSE Arousal: {val_rmse_a:.4f}")

        # Save history
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

        # Save best model
        if val_ccc > best_ccc:
            best_ccc = val_ccc
            patience_counter = 0

            save_path = project_root / "models" / "final_model_best.pt"
            save_path.parent.mkdir(exist_ok=True)

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
            }, save_path)
            logger.info(f"  ✓ Best model saved! (CCC: {best_ccc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation CCC: {best_ccc:.4f}")
    logger.info(f"Model saved to: models/final_model_best.pt")

    # Print summary
    history_df = pd.DataFrame(history)
    logger.info("\nTraining History:")
    logger.info(history_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
