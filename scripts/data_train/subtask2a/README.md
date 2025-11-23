# Subtask 2a - Training Scripts

## 🎯 Overview

This folder contains training scripts for Subtask 2a (Emotion Prediction).

---

## 📁 Files

### `train_ensemble_subtask2a.py`

**Purpose**: Train emotion prediction models with different random seeds for ensemble

**Architecture**:
- RoBERTa-base encoder (125M params)
- BiLSTM (256 hidden, 2 layers)
- Multi-head attention (8 heads)
- User embeddings (64 dim)
- Dual-head output (Valence & Arousal)

**Configuration**:
```python
RANDOM_SEED = 777  # Change to 42, 123, or 777
MODEL_SAVE_NAME = f'subtask2a_seed{RANDOM_SEED}_best.pt'
USE_WANDB = False  # Optional experiment tracking
```

**Training Details**:
- Batch size: 16
- Learning rate: 1e-5 (AdamW)
- Max epochs: 50
- Early stopping: Patience 10
- Dropout: 0.3
- Loss: Dual-head CCC+MSE

**Expected Performance**:
- Individual model: CCC 0.50-0.66
- Training time: ~90-120 minutes on T4 GPU

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

```bash
# 1. Upload script to Google Colab
# 2. Set runtime to T4 GPU
# 3. Configure random seed
RANDOM_SEED = 42  # or 123, 777

# 4. Run training
# Script will automatically:
#   - Download RoBERTa model
#   - Load training data
#   - Train model
#   - Save best checkpoint
```

### Option 2: Local Training

```bash
# Requirements
pip install torch transformers pandas numpy scipy scikit-learn

# Run training
python train_ensemble_subtask2a.py
```

---

## 📊 Training Strategy

### Ensemble Approach

Train 3 models with different seeds:

1. **Seed 42**: Already trained (CCC 0.5053)
2. **Seed 123**: CCC 0.5330
3. **Seed 777**: CCC 0.6554

**Ensemble Result**: CCC 0.5846-0.6046

---

## 💾 Output

**Model File**: `subtask2a_seed{SEED}_best.pt`

Contains:
- Model state dict
- Best validation CCC
- Valence & Arousal CCC
- Training epoch
- RMSE metrics

**Location**: Auto-downloaded from Colab or saved locally

---

## 📈 Key Features

### Dual-Head Loss
- **Valence**: 65% CCC + 35% MSE
- **Arousal**: 70% CCC + 30% MSE

### Feature Engineering (39 features)
- 5 Lag features (temporal context)
- 15 User statistics
- 19 Text features

### Regularization
- Dropout: 0.3
- Weight decay: 0.01
- Early stopping

---

## ⚙️ Hyperparameters

```python
# Architecture
USER_EMB_DIM = 64
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTENTION_HEADS = 8
DROPOUT = 0.3

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
MAX_EPOCHS = 50
PATIENCE = 10
WEIGHT_DECAY = 0.01

# Loss weights
CCC_WEIGHT_VALENCE = 0.65
CCC_WEIGHT_AROUSAL = 0.70
MSE_WEIGHT_VALENCE = 0.35
MSE_WEIGHT_AROUSAL = 0.30
```

---

## 🔍 Monitoring

Training progress includes:
- Epoch-by-epoch CCC (Valence & Arousal)
- Train-validation gap
- RMSE metrics
- Learning rate schedule
- Best model checkpointing

---

## 📖 References

See: [docs/subtask2a/ENSEMBLE_GUIDE.md](../../../docs/subtask2a/ENSEMBLE_GUIDE.md) for complete guide

---

**Last Updated**: 2025-11-14
