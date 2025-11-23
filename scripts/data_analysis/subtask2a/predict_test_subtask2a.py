# ============================================================================
# Subtask 2a - Test Data Prediction with Ensemble
# ============================================================================
#
# Generate predictions for test data using trained ensemble models
#
# Requirements:
# - 3 trained models: subtask2a_seed42_best.pt, subtask2a_seed123_best.pt, subtask2a_seed777_best.pt
# - Test data: test_subtask2a.csv (user_id, text_id, text, timestamp, etc.)
# - Ensemble weights from: results/subtask2a/ensemble_results.json
#
# Output:
# - pred_subtask2a.csv (user_id, pred_state_change_valence, pred_state_change_arousal)
#
# ============================================================================

"""
Subtask 2a - Test Prediction Script
====================================
Load 3 trained models, generate predictions using performance-based ensemble weights
"""

print('='*80)
print('Subtask 2a - Test Data Prediction with Ensemble')
print('='*80)

# ===== IMPORTS =====
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import os
import re

# ===== DEVICE SETUP =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# ===== PATHS =====
# Auto-detect if running in Colab or local
import sys
if 'google.colab' in sys.modules or os.path.exists('/content'):
    # Colab environment
    BASE_DIR = '/content/Deep-Learning-project-SemEval-2026-Task-2'
    TEST_DATA_PATH = f'{BASE_DIR}/data/test/test_subtask2a.csv'
    MODEL_DIR = f'{BASE_DIR}/models'
    RESULTS_DIR = f'{BASE_DIR}/results/subtask2a'
    print('🔍 Detected: Google Colab environment')
else:
    # Local environment
    BASE_DIR = 'D:/Study/Github/Deep-Learning-project-SemEval-2026-Task-2'
    TEST_DATA_PATH = f'{BASE_DIR}/data/test/test_subtask2a.csv'
    MODEL_DIR = f'{BASE_DIR}/models'
    RESULTS_DIR = f'{BASE_DIR}/results/subtask2a'
    print('🔍 Detected: Local environment')

print(f'📁 Base directory: {BASE_DIR}')

MODEL_PATHS = {
    'seed42': f'{MODEL_DIR}/subtask2a_seed42_best.pt',
    'seed123': f'{MODEL_DIR}/subtask2a_seed123_best.pt',
    'seed777': f'{MODEL_DIR}/subtask2a_seed777_best.pt'
}

ENSEMBLE_WEIGHTS_PATH = f'{RESULTS_DIR}/ensemble_results.json'

# ===== CONFIGURATION =====
SEQ_LENGTH = 128
BATCH_SIZE = 16

# ===== FEATURE EXTRACTION FUNCTIONS =====

def extract_text_features(text):
    """Extract text-based features from a single text"""
    # Length features
    text_length = len(text)
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)

    # Sentence features
    sentence_count = len([s for s in text.split('.') if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)

    # Punctuation features
    exclamation_count = text.count('!')
    question_count = text.count('?')
    comma_count = text.count(',')
    period_count = text.count('.')

    # Case features
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / max(len(text), 1)

    # Emotion word counts (simple)
    positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'fantastic']
    negative_words = ['bad', 'sad', 'terrible', 'hate', 'awful', 'horrible', 'miserable']

    text_lower = text.lower()
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)

    # Digit features
    digit_count = sum(1 for c in text if c.isdigit())

    # Special char features
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))

    return [
        text_length, word_count, avg_word_length,
        sentence_count, avg_sentence_length,
        exclamation_count, question_count, comma_count, period_count,
        upper_count, upper_ratio,
        positive_count, negative_count,
        digit_count, special_char_count
    ]

def preprocess_test_data(df):
    """
    Preprocess test data and create lag features

    For test data in Subtask 2a, we need to:
    1. Group by user_id
    2. Sort by timestamp within each user
    3. Create lag features from observed history
    """
    print('\n=== Preprocessing Test Data ===')

    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # Initialize feature columns
    df['lag_1_valence'] = 0.0
    df['lag_1_arousal'] = 0.0
    df['lag_2_valence'] = 0.0
    df['lag_2_arousal'] = 0.0
    df['lag_mean_valence'] = 0.0

    # User statistics (if valence/arousal provided in test data for context)
    # If not provided, we'll use zeros
    user_stats_cols = []
    if 'valence' in df.columns and 'arousal' in df.columns:
        user_stats = df.groupby('user_id').agg({
            'valence': ['mean', 'std', 'min', 'max', 'median'],
            'arousal': ['mean', 'std', 'min', 'max', 'median'],
            'text': 'count'
        }).reset_index()

        user_stats.columns = ['user_id',
            'user_valence_mean', 'user_valence_std', 'user_valence_min', 'user_valence_max', 'user_valence_median',
            'user_arousal_mean', 'user_arousal_std', 'user_arousal_min', 'user_arousal_max', 'user_arousal_median',
            'user_text_count'
        ]

        # Fill NaN std with 0
        user_stats['user_valence_std'] = user_stats['user_valence_std'].fillna(0)
        user_stats['user_arousal_std'] = user_stats['user_arousal_std'].fillna(0)

        # Normalize user_text_count
        user_stats['user_text_count_norm'] = user_stats['user_text_count'] / user_stats['user_text_count'].max()

        df = df.merge(user_stats, on='user_id', how='left')

        user_stats_cols = ['user_valence_mean', 'user_valence_std', 'user_valence_min', 'user_valence_max', 'user_valence_median',
                          'user_arousal_mean', 'user_arousal_std', 'user_arousal_min', 'user_arousal_max', 'user_arousal_median',
                          'user_text_count', 'user_text_count_norm']

        # Create lag features within each user
        for user_id, group in df.groupby('user_id'):
            indices = group.index

            for i, idx in enumerate(indices):
                if i >= 1:
                    df.loc[idx, 'lag_1_valence'] = df.loc[indices[i-1], 'valence']
                    df.loc[idx, 'lag_1_arousal'] = df.loc[indices[i-1], 'arousal']

                if i >= 2:
                    df.loc[idx, 'lag_2_valence'] = df.loc[indices[i-2], 'valence']
                    df.loc[idx, 'lag_2_arousal'] = df.loc[indices[i-2], 'arousal']

                if i >= 1:
                    df.loc[idx, 'lag_mean_valence'] = df.loc[indices[:i], 'valence'].mean()
    else:
        # If no valence/arousal in test data, use zeros
        print('⚠️ No valence/arousal in test data, using zero features')
        user_stats_cols = []

        # Create dummy user stats
        for col in ['user_valence_mean', 'user_valence_std', 'user_valence_min', 'user_valence_max', 'user_valence_median',
                    'user_arousal_mean', 'user_arousal_std', 'user_arousal_min', 'user_arousal_max', 'user_arousal_median',
                    'user_text_count', 'user_text_count_norm']:
            df[col] = 0.0
            user_stats_cols.append(col)

    # Extract text features
    print('Extracting text features...')
    text_features_list = []
    for text in tqdm(df['text'], desc='Text features'):
        text_features_list.append(extract_text_features(text))

    text_features = np.array(text_features_list)
    text_feature_cols = [f'text_feat_{i}' for i in range(text_features.shape[1])]

    for i, col in enumerate(text_feature_cols):
        df[col] = text_features[:, i]

    print(f'✓ Created {len(text_feature_cols)} text features')
    print(f'✓ Created 5 lag features')
    print(f'✓ Created {len(user_stats_cols)} user statistics')

    return df, user_stats_cols, text_feature_cols

# ===== DATASET =====

class TestEmotionDataset(Dataset):
    """Dataset for test data prediction"""

    def __init__(self, df, tokenizer, seq_length, user_stats_cols, text_feature_cols):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.user_stats_cols = user_stats_cols
        self.text_feature_cols = text_feature_cols

        # User mapping
        unique_users = df['user_id'].unique()
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.num_users = len(unique_users)

        print(f'Test dataset: {len(self.df)} samples, {self.num_users} users')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize text
        encoding = self.tokenizer(
            row['text'],
            add_special_tokens=True,
            max_length=self.seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # User index
        user_idx = self.user_to_idx[row['user_id']]

        # Temporal features (lag features)
        temporal_features = torch.tensor([
            row['lag_1_valence'], row['lag_1_arousal'],
            row['lag_2_valence'], row['lag_2_arousal'],
            row['lag_mean_valence']
        ], dtype=torch.float32)

        # User statistics
        user_stats = torch.tensor([row[col] for col in self.user_stats_cols], dtype=torch.float32)

        # Text features
        text_features = torch.tensor([row[col] for col in self.text_feature_cols], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'temporal_features': temporal_features,
            'user_stats': user_stats,
            'text_features': text_features,
            'user_id': row['user_id']  # Keep for submission
        }

# ===== MODEL ARCHITECTURE =====

class FinalEmotionModel(nn.Module):
    """
    RoBERTa + BiLSTM + Multi-Head Attention + Dual-Head Loss

    This is the exact architecture used during training
    """
    def __init__(self, num_users, user_emb_dim=64, lstm_hidden=256, lstm_layers=2,
                 attention_heads=8, dropout=0.3):
        super().__init__()

        # RoBERTa encoder
        self.roberta = AutoModel.from_pretrained('roberta-base')
        roberta_dim = 768

        # User embeddings
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)

        # BiLSTM
        self.lstm = nn.LSTM(
            roberta_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feature dimensions
        temporal_dim = 5  # lag features
        user_stats_dim = 12  # user statistics
        text_features_dim = 15  # text features

        combined_dim = lstm_output_dim + user_emb_dim + temporal_dim + user_stats_dim + text_features_dim

        # Dual-head output layers
        self.dropout = nn.Dropout(dropout)

        self.valence_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, user_idx, temporal_features, user_stats, text_features):
        # RoBERTa encoding
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state  # (batch, seq_len, 768)

        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # (batch, seq_len, lstm_hidden*2)

        # Multi-head attention
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  # (batch, seq_len, lstm_hidden*2)

        # Global average pooling
        pooled_output = torch.mean(attn_output, dim=1)  # (batch, lstm_hidden*2)

        # User embeddings
        user_emb = self.user_embedding(user_idx)  # (batch, user_emb_dim)

        # Concatenate all features
        combined = torch.cat([
            pooled_output,
            user_emb,
            temporal_features,
            user_stats,
            text_features
        ], dim=1)

        combined = self.dropout(combined)

        # Dual-head predictions
        valence_pred = self.valence_head(combined).squeeze(-1)
        arousal_pred = self.arousal_head(combined).squeeze(-1)

        return valence_pred, arousal_pred

# ===== LOAD ENSEMBLE WEIGHTS =====

print('\n=== Loading Ensemble Weights ===')
try:
    with open(ENSEMBLE_WEIGHTS_PATH, 'r') as f:
        ensemble_info = json.load(f)

    weights = ensemble_info['ensemble']['weights']
    print(f'✓ Loaded ensemble weights:')
    print(f'  seed42:  {weights["seed42"]:.4f}')
    print(f'  seed123: {weights["seed123"]:.4f}')
    print(f'  seed777: {weights["seed777"]:.4f}')
except FileNotFoundError:
    print('⚠️ Ensemble weights file not found, using equal weights')
    weights = {'seed42': 1/3, 'seed123': 1/3, 'seed777': 1/3}

# ===== LOAD TEST DATA =====

print('\n=== Loading Test Data ===')
if not os.path.exists(TEST_DATA_PATH):
    print(f'❌ Error: Test data not found at {TEST_DATA_PATH}')
    print('Please download test_subtask2a.csv and place it in the current directory')
    print('Or update TEST_DATA_PATH variable')
    exit(1)

test_df = pd.read_csv(TEST_DATA_PATH)
print(f'✓ Loaded test data: {len(test_df)} samples')
print(f'Columns: {list(test_df.columns)}')

# Preprocess test data
test_df, user_stats_cols, text_feature_cols = preprocess_test_data(test_df)

# ===== CREATE DATASET & DATALOADER =====

print('\n=== Creating Dataset ===')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

test_dataset = TestEmotionDataset(
    test_df,
    tokenizer,
    SEQ_LENGTH,
    user_stats_cols,
    text_feature_cols
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ===== GENERATE PREDICTIONS WITH ENSEMBLE =====

print('\n=== Generating Predictions with Ensemble ===')

all_predictions = {}

for seed_name, model_path in MODEL_PATHS.items():
    print(f'\nLoading model: {seed_name}')

    if not os.path.exists(model_path):
        print(f'❌ Error: Model file not found: {model_path}')
        print('Please ensure all 3 models are in the correct location')
        exit(1)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print(f'✓ Loaded checkpoint (CCC: {checkpoint["best_ccc"]:.4f}, Epoch: {checkpoint["epoch"]})')

    # Create model
    model = FinalEmotionModel(num_users=test_dataset.num_users)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Generate predictions
    valence_preds = []
    arousal_preds = []
    user_ids = []

    print(f'Generating predictions with {seed_name}...')
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'{seed_name} prediction'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_idx = batch['user_idx'].to(device)
            temporal_features = batch['temporal_features'].to(device)
            user_stats = batch['user_stats'].to(device)
            text_features = batch['text_features'].to(device)

            valence_pred, arousal_pred = model(
                input_ids, attention_mask, user_idx,
                temporal_features, user_stats, text_features
            )

            valence_preds.extend(valence_pred.cpu().numpy())
            arousal_preds.extend(arousal_pred.cpu().numpy())

            if seed_name == 'seed42':  # Only collect user_ids once
                user_ids.extend(batch['user_id'].numpy())

    all_predictions[seed_name] = {
        'valence': np.array(valence_preds),
        'arousal': np.array(arousal_preds)
    }

    print(f'✓ {seed_name} predictions complete')
    print(f'  Valence range: [{all_predictions[seed_name]["valence"].min():.3f}, {all_predictions[seed_name]["valence"].max():.3f}]')
    print(f'  Arousal range: [{all_predictions[seed_name]["arousal"].min():.3f}, {all_predictions[seed_name]["arousal"].max():.3f}]')

# ===== WEIGHTED ENSEMBLE =====

print('\n=== Creating Weighted Ensemble ===')

ensemble_valence = np.zeros_like(all_predictions['seed42']['valence'])
ensemble_arousal = np.zeros_like(all_predictions['seed42']['arousal'])

for seed_name, preds in all_predictions.items():
    weight = weights[seed_name]
    ensemble_valence += weight * preds['valence']
    ensemble_arousal += weight * preds['arousal']
    print(f'{seed_name}: weight {weight:.4f}')

print(f'\n✓ Ensemble predictions created')
print(f'  Valence range: [{ensemble_valence.min():.3f}, {ensemble_valence.max():.3f}]')
print(f'  Arousal range: [{ensemble_arousal.min():.3f}, {ensemble_arousal.max():.3f}]')

# ===== AGGREGATE BY USER (Subtask 2a specific) =====

print('\n=== Aggregating Predictions by User ===')

# For Subtask 2a, we predict state change PER USER, not per text
# We need to aggregate predictions somehow - typically we predict for the LAST text of each user

# Group by user and get last prediction
results_df = pd.DataFrame({
    'user_id': user_ids,
    'pred_state_change_valence': ensemble_valence,
    'pred_state_change_arousal': ensemble_arousal
})

# Get last prediction per user (most recent timestamp)
test_df_with_pred = test_df.copy()
test_df_with_pred['pred_state_change_valence'] = ensemble_valence
test_df_with_pred['pred_state_change_arousal'] = ensemble_arousal

# Sort by timestamp and get last entry per user
final_predictions = test_df_with_pred.sort_values('timestamp').groupby('user_id').last().reset_index()
final_predictions = final_predictions[['user_id', 'pred_state_change_valence', 'pred_state_change_arousal']]

print(f'✓ Final predictions: {len(final_predictions)} users')

# ===== SAVE SUBMISSION FILE =====

output_path = 'pred_subtask2a.csv'
final_predictions.to_csv(output_path, index=False)

print('\n' + '='*80)
print('PREDICTION COMPLETE')
print('='*80)
print(f'✓ Saved predictions to: {output_path}')
print(f'✓ Number of users: {len(final_predictions)}')
print(f'\nSubmission format:')
print(final_predictions.head(10))
print('\nStatistics:')
print(f'  Valence - Mean: {final_predictions["pred_state_change_valence"].mean():.3f}, '
      f'Std: {final_predictions["pred_state_change_valence"].std():.3f}')
print(f'  Arousal - Mean: {final_predictions["pred_state_change_arousal"].mean():.3f}, '
      f'Std: {final_predictions["pred_state_change_arousal"].std():.3f}')

print('\n' + '='*80)
print('Next Steps:')
print('1. Create submission.zip with this pred_subtask2a.csv file')
print('2. Upload to Codabench: https://www.codabench.org/competitions/9963/')
print('3. Wait for evaluation results')
print('='*80)
