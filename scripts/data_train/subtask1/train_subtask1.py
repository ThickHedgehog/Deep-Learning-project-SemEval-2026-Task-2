"""
Training script for Subtask 1: Longitudinal Affect Assessment
Model: BERT + LSTM + User Embeddings + Attention
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU Device: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Check if this is a very new GPU that might need special handling
    if 'RTX 50' in gpu_name or 'RTX 60' in gpu_name:
        logger.warning(f"Detected very new GPU: {gpu_name}")
        logger.warning("If you encounter CUDA errors, try:")
        logger.warning("  1. Install PyTorch nightly build")
        logger.warning("  2. Set TORCH_CUDA_ARCH_LIST environment variable")
        logger.warning("  3. See GPU_SETUP_INSTRUCTIONS.md for details")
else:
    logger.warning("=" * 70)
    logger.warning("CUDA is NOT available - using CPU!")
    logger.warning("Training will be VERY slow on CPU.")
    logger.warning("To use GPU, install PyTorch with CUDA support:")
    logger.warning("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    logger.warning("See GPU_SETUP_INSTRUCTIONS.md for more details.")
    logger.warning("=" * 70)


# ===== 1. DATA SPLITTING =====

def temporal_split_by_user(df, train_ratio=0.7, val_ratio=0.15):
    """Split data temporally per user to preserve chronological order."""
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].sort_values('timestamp')
        n = len(user_df)
        
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(user_df.iloc[:train_idx])
        val_dfs.append(user_df.iloc[train_idx:val_idx])
        test_dfs.append(user_df.iloc[val_idx:])
    
    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)


# ===== 2. DATASET CLASS =====

class EmotionDataset(Dataset):
    """Dataset for emotion prediction with temporal sequences."""
    
    def __init__(self, df, tokenizer, max_length=128, seq_length=5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seq_length = seq_length
        
        # Create user ID mapping
        self.user_ids = df['user_id'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        
        # Create sequences per user
        self.sequences = []
        for user_id in self.user_ids:
            user_df = df[df['user_id'] == user_id].sort_values('timestamp').reset_index(drop=True)
            if len(user_df) < seq_length:
                continue
            
            # Create overlapping sequences
            for i in range(len(user_df) - seq_length + 1):
                seq_data = user_df.iloc[i:i + seq_length]
                self.sequences.append({
                    'texts': seq_data['text_cleaned'].tolist(),
                    'valences': seq_data['valence'].tolist(),
                    'arousals': seq_data['arousal'].tolist(),
                    'user_id': self.user_to_idx[user_id]
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize all texts in sequence
        input_ids_list = []
        attention_mask_list = []
        
        for text in seq['texts']:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'].squeeze(0))
            attention_mask_list.append(encoding['attention_mask'].squeeze(0))
        
        return {
            'input_ids': torch.stack(input_ids_list),  # [seq_len, max_length]
            'attention_mask': torch.stack(attention_mask_list),  # [seq_len, max_length]
            'valence': torch.tensor(seq['valences'], dtype=torch.float),  # [seq_len]
            'arousal': torch.tensor(seq['arousals'], dtype=torch.float),  # [seq_len]
            'user_id': torch.tensor(seq['user_id'], dtype=torch.long)
        }


# ===== 3. MODEL =====

class EmotionPredictionModel(nn.Module):
    """BERT + LSTM + User Embeddings + Attention for emotion prediction."""
    
    def __init__(self, num_users, bert_model='bert-base-uncased', lstm_hidden=256, user_embed_dim=64):
        super().__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_dim = self.bert.config.hidden_size
        
        # User embeddings
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            bert_dim + user_embed_dim,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Regression heads (applied to each timestep)
        self.valence_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask, user_id):
        batch_size, seq_len, max_len = input_ids.shape
        
        # Encode each text with BERT
        input_ids = input_ids.view(-1, max_len)
        attention_mask = attention_mask.view(-1, max_len)
        
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        text_embeds = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_embeds = text_embeds.view(batch_size, seq_len, -1)
        
        # Add user embeddings
        user_embeds = self.user_embedding(user_id).unsqueeze(1).expand(-1, seq_len, -1)
        combined_embeds = torch.cat([text_embeds, user_embeds], dim=-1)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(combined_embeds)  # [batch, seq_len, lstm_hidden*2]
        
        # Predict emotions for EACH timestep
        valence = self.valence_head(lstm_out).squeeze(-1)  # [batch, seq_len]
        arousal = self.arousal_head(lstm_out).squeeze(-1)  # [batch, seq_len]
        
        return valence, arousal


# ===== 4. TRAINING FUNCTION =====

def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        user_id = batch['user_id'].to(device)
        valence_target = batch['valence'].to(device)  # [batch, seq_len]
        arousal_target = batch['arousal'].to(device)  # [batch, seq_len]
        
        optimizer.zero_grad()
        
        valence_pred, arousal_pred = model(input_ids, attention_mask, user_id)
        
        loss_v = criterion(valence_pred, valence_target)
        loss_a = criterion(arousal_pred, arousal_target)
        loss = loss_v + loss_a
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_valence_pred, all_arousal_pred = [], []
    all_valence_true, all_arousal_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_id = batch['user_id'].to(device)
            valence_target = batch['valence'].to(device)  # [batch, seq_len]
            arousal_target = batch['arousal'].to(device)  # [batch, seq_len]
            
            valence_pred, arousal_pred = model(input_ids, attention_mask, user_id)
            
            loss_v = criterion(valence_pred, valence_target)
            loss_a = criterion(arousal_pred, arousal_target)
            loss = loss_v + loss_a
            
            total_loss += loss.item()
            
            # Flatten sequences for correlation calculation
            all_valence_pred.extend(valence_pred.cpu().numpy().flatten())
            all_arousal_pred.extend(arousal_pred.cpu().numpy().flatten())
            all_valence_true.extend(valence_target.cpu().numpy().flatten())
            all_arousal_true.extend(arousal_target.cpu().numpy().flatten())
    
    # Calculate Pearson correlation
    valence_corr = np.corrcoef(all_valence_true, all_valence_pred)[0, 1]
    arousal_corr = np.corrcoef(all_arousal_true, all_arousal_pred)[0, 1]
    
    return total_loss / len(dataloader), valence_corr, arousal_corr


# ===== 5. MAIN TRAINING LOOP =====

def main():
    # Load processed data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'processed' / 'subtask1_processed.csv'
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {df['user_id'].nunique()} users")
    
    # Split data
    logger.info("Splitting data temporally per user...")
    train_df, val_df, test_df = temporal_split_by_user(df)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = EmotionDataset(train_df, tokenizer, seq_length=5)
    val_dataset = EmotionDataset(val_df, tokenizer, seq_length=5)
    test_dataset = EmotionDataset(test_df, tokenizer, seq_length=5)
    
    logger.info(f"Train sequences: {len(train_dataset)}")
    logger.info(f"Val sequences: {len(val_dataset)}")
    logger.info(f"Test sequences: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    num_users = df['user_id'].nunique()
    logger.info(f"Initializing model with {num_users} users...")
    model = EmotionPredictionModel(num_users=num_users).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 10
    
    best_val_loss = float('inf')
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_corr_v, val_corr_a = evaluate(model, val_loader, criterion)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Valence Corr: {val_corr_v:.4f}, Arousal Corr: {val_corr_a:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = project_root / 'models' / 'subtask1_best_model.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_corr_v, test_corr_a = evaluate(model, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Valence Correlation: {test_corr_v:.4f}")
    logger.info(f"Test Arousal Correlation: {test_corr_a:.4f}")
    
    logger.info("\nTraining completed!")


if __name__ == '__main__':
    main()
