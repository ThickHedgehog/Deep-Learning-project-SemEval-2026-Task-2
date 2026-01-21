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

def temporal_split_by_user(df, train_ratio=0.7, val_ratio=0.15, test_user_ratio=0.2):

    import numpy as np
    
    all_users = df['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_users)
    
    # Split users
    n_users = len(all_users)
    n_train_users = int(n_users * (1 - test_user_ratio))
    
    train_users = all_users[:n_train_users]
    test_only_users = all_users[n_train_users:]
    
    train_dfs, val_dfs, test_dfs = [], [], []
    
    # Process train users (temporal split within each user)
    for user_id in train_users:
        user_df = df[df['user_id'] == user_id].sort_values('timestamp')
        n = len(user_df)
        
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(user_df.iloc[:train_idx])
        val_dfs.append(user_df.iloc[train_idx:val_idx])
        test_dfs.append(user_df.iloc[val_idx:])
    
    # Process test-only users (unseen users - all data goes to test)
    for user_id in test_only_users:
        user_df = df[df['user_id'] == user_id].sort_values('timestamp')
        test_dfs.append(user_df)
    
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
    
    return train_df, val_df, test_df


# ===== 2. DATASET CLASS =====

class EmotionDataset(Dataset):
    
    def __init__(self, df, tokenizer, user_to_idx=None, max_length=128, max_history=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_history = max_history
        
        # Create or use provided user ID mapping
        if user_to_idx is None:
            self.user_ids = df['user_id'].unique()
            self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        else:
            self.user_to_idx = user_to_idx
            self.user_ids = list(user_to_idx.keys())
        
        # Create samples: each text with its history
        self.samples = []
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id].sort_values('timestamp').reset_index(drop=True)
            
            # For each text, create a sample with history
            for i in range(len(user_df)):
                # Get history (up to max_history previous texts)
                start_idx = max(0, i - self.max_history)
                history = user_df.iloc[start_idx:i]['text_cleaned'].tolist()
                current_text = user_df.iloc[i]['text_cleaned']
                
                # Get user index (or -1 for unseen users)
                user_idx = self.user_to_idx.get(user_id, -1)
                
                self.samples.append({
                    'history': history,  # Previous texts
                    'current_text': current_text,  # Text to predict for
                    'valence': user_df.iloc[i]['valence'],
                    'arousal': user_df.iloc[i]['arousal'],
                    'user_id': user_idx,
                    'has_user_embed': user_idx != -1
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize current text
        current_encoding = self.tokenizer(
            sample['current_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize history texts (pad if less than max_history)
        history_input_ids = []
        history_attention_mask = []
        
        for text in sample['history']:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            history_input_ids.append(encoding['input_ids'].squeeze(0))
            history_attention_mask.append(encoding['attention_mask'].squeeze(0))
        
        # Pad history to max_history length
        while len(history_input_ids) < self.max_history:
            history_input_ids.insert(0, torch.zeros(self.max_length, dtype=torch.long))
            history_attention_mask.insert(0, torch.zeros(self.max_length, dtype=torch.long))
        
        # Keep only last max_history items
        history_input_ids = history_input_ids[-self.max_history:]
        history_attention_mask = history_attention_mask[-self.max_history:]
        
        return {
            'current_input_ids': current_encoding['input_ids'].squeeze(0),  # [max_length]
            'current_attention_mask': current_encoding['attention_mask'].squeeze(0),  # [max_length]
            'history_input_ids': torch.stack(history_input_ids),  # [max_history, max_length]
            'history_attention_mask': torch.stack(history_attention_mask),  # [max_history, max_length]
            'valence': torch.tensor(sample['valence'], dtype=torch.float),
            'arousal': torch.tensor(sample['arousal'], dtype=torch.float),
            'user_id': torch.tensor(max(0, sample['user_id']), dtype=torch.long),  # Use 0 for unseen
            'has_user_embed': torch.tensor(sample['has_user_embed'], dtype=torch.bool)
        }


# ===== 3. MODEL =====

class EmotionPredictionModel(nn.Module):
    
    def __init__(self, num_users, bert_model='bert-base-uncased', lstm_hidden=256, user_embed_dim=64):
        super().__init__()
        
        # BERT encoder (shared for current text and history)
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_dim = self.bert.config.hidden_size
        
        # User embeddings (optional, for seen users only)
        self.user_embedding = nn.Embedding(num_users + 1, user_embed_dim)  # +1 for unseen
        self.user_embed_dim = user_embed_dim
        
        # LSTM for processing history
        self.history_lstm = nn.LSTM(
            bert_dim,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3
        )
        
        # Attention mechanism to aggregate history
        self.attention = nn.Sequential(
            nn.Linear(bert_dim + lstm_hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Final prediction layers
        combined_dim = bert_dim + lstm_hidden + user_embed_dim
        
        self.valence_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, current_input_ids, current_attention_mask, 
                history_input_ids, history_attention_mask, 
                user_id, has_user_embed):
        batch_size = current_input_ids.shape[0]
        
        # Encode current text with BERT
        current_bert_out = self.bert(current_input_ids, attention_mask=current_attention_mask)
        current_embed = current_bert_out.last_hidden_state[:, 0, :]  # [batch, bert_dim]
        
        # Encode history texts with BERT
        max_history, max_len = history_input_ids.shape[1], history_input_ids.shape[2]
        history_input_ids = history_input_ids.view(-1, max_len)
        history_attention_mask = history_attention_mask.view(-1, max_len)
        
        # Check which history entries are valid (not padding)
        valid_history = history_attention_mask.sum(dim=1) > 0  # [batch*max_history]
        
        # Encode only valid history
        if valid_history.any():
            history_bert_out = self.bert(history_input_ids, attention_mask=history_attention_mask)
            history_embeds = history_bert_out.last_hidden_state[:, 0, :]  # [batch*max_history, bert_dim]
            history_embeds = history_embeds.view(batch_size, max_history, -1)  # [batch, max_history, bert_dim]
            
            # Process history with LSTM
            lstm_out, (hidden, _) = self.history_lstm(history_embeds)  # lstm_out: [batch, max_history, lstm_hidden]
            history_context = hidden[-1]  # Use last hidden state: [batch, lstm_hidden]
        else:
            # No valid history, use zero context
            history_context = torch.zeros(batch_size, self.history_lstm.hidden_size, device=current_embed.device)
        
        # Get user embeddings
        user_embeds = self.user_embedding(user_id)  # [batch, user_embed_dim]
        
        # For unseen users, zero out user embeddings
        user_embeds = user_embeds * has_user_embed.unsqueeze(1).float()
        
        # Combine all features
        combined = torch.cat([current_embed, history_context, user_embeds], dim=-1)  # [batch, combined_dim]
        
        # Predict valence and arousal
        valence = self.valence_head(combined).squeeze(-1)  # [batch]
        arousal = self.arousal_head(combined).squeeze(-1)  # [batch]
        
        return valence, arousal


# ===== 4. TRAINING FUNCTION =====

def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        current_input_ids = batch['current_input_ids'].to(device)
        current_attention_mask = batch['current_attention_mask'].to(device)
        history_input_ids = batch['history_input_ids'].to(device)
        history_attention_mask = batch['history_attention_mask'].to(device)
        user_id = batch['user_id'].to(device)
        has_user_embed = batch['has_user_embed'].to(device)
        valence_target = batch['valence'].to(device)  # [batch]
        arousal_target = batch['arousal'].to(device)  # [batch]
        
        optimizer.zero_grad()
        
        valence_pred, arousal_pred = model(
            current_input_ids, current_attention_mask,
            history_input_ids, history_attention_mask,
            user_id, has_user_embed
        )
        
        loss_v = criterion(valence_pred, valence_target)
        loss_a = criterion(arousal_pred, arousal_target)
        loss = loss_v + loss_a
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            current_input_ids = batch['current_input_ids'].to(device)
            current_attention_mask = batch['current_attention_mask'].to(device)
            history_input_ids = batch['history_input_ids'].to(device)
            history_attention_mask = batch['history_attention_mask'].to(device)
            user_id = batch['user_id'].to(device)
            has_user_embed = batch['has_user_embed'].to(device)
            valence_target = batch['valence'].to(device)  # [batch]
            arousal_target = batch['arousal'].to(device)  # [batch]
            
            valence_pred, arousal_pred = model(
                current_input_ids, current_attention_mask,
                history_input_ids, history_attention_mask,
                user_id, has_user_embed
            )
            
            loss_v = criterion(valence_pred, valence_target)
            loss_a = criterion(arousal_pred, arousal_target)
            loss = loss_v + loss_a
            
            total_loss += loss.item()
            
            all_valence_pred.extend(valence_pred.cpu().numpy())
            all_arousal_pred.extend(arousal_pred.cpu().numpy())
            all_valence_true.extend(valence_target.cpu().numpy())
            all_arousal_true.extend(arousal_target.cpu().numpy())
    
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
    logger.info("Splitting data temporally per user + creating unseen users...")
    train_df, val_df, test_df = temporal_split_by_user(df, train_ratio=0.7, val_ratio=0.15, test_user_ratio=0.2)
    logger.info(f"Train: {len(train_df)} samples from {train_df['user_id'].nunique()} users")
    logger.info(f"Val: {len(val_df)} samples from {val_df['user_id'].nunique()} users")
    logger.info(f"Test: {len(test_df)} samples from {test_df['user_id'].nunique()} users")
    
    # Count seen/unseen in test
    train_users = set(train_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    seen_users = test_users & train_users
    unseen_users = test_users - train_users
    logger.info(f"Test set: {len(seen_users)} seen users, {len(unseen_users)} unseen users")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create user mapping from training data only
    train_users = train_df['user_id'].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(train_users)}
    logger.info(f"Training with {len(user_to_idx)} users")
    
    # Create datasets (val and test may have unseen users)
    logger.info("Creating datasets...")
    train_dataset = EmotionDataset(train_df, tokenizer, user_to_idx=user_to_idx, max_history=10)
    val_dataset = EmotionDataset(val_df, tokenizer, user_to_idx=user_to_idx, max_history=10)
    test_dataset = EmotionDataset(test_df, tokenizer, user_to_idx=user_to_idx, max_history=10)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    num_users = len(user_to_idx)
    logger.info(f"Initializing model with {num_users} users...")
    model = EmotionPredictionModel(num_users=num_users).to(device)
    
    # Save user mapping for evaluation
    import json
    user_mapping_path = project_root / 'models' / 'user_mapping.json'
    user_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_mapping_path, 'w') as f:
        json.dump(user_to_idx, f)
    logger.info(f"Saved user mapping to {user_mapping_path}")
    
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
