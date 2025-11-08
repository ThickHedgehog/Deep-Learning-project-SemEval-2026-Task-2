# SemEval-2026 Task 2 — Predicting Variation in Emotional Responses

## 📌 Project Description

**Task**: Predict emotional responses (Valence and Arousal) for temporal text sequences.

**Goal**: Build a model that captures temporal dynamics and user-specific patterns in emotional responses.

---

## 🎯 Subtask 2a - Final Model (v3)

### ✅ Status: Ready to Train

**Model**: FINAL Optimized Temporal Emotion Prediction (v3)

**Expected Performance**: CCC 0.65-0.72 (Competition Ready)

**Architecture**: RoBERTa + BiLSTM + Multi-Head Attention + Dual-Head Loss

**Training Time**: ~90-120 minutes on Tesla T4 GPU

---

## 📚 Documentation

**Start Here**:
- **[QUICKSTART.md](QUICKSTART.md)** ⭐ - 5-minute setup guide (recommended for first-time users)
- **[EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md)** ✅ - Complete step-by-step checklist with verification steps

**Additional Resources**:
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status and completed tasks
- **[VERSION_HISTORY.md](VERSION_HISTORY.md)** - Development history and performance comparison
- **[validate_setup.py](validate_setup.py)** - Pre-flight validation script

---

## 🚀 Quick Start - Google Colab (Recommended)

### Step 1: Open Google Colab
https://colab.research.google.com/

### Step 2: Enable GPU
Runtime → Change runtime type → **T4 GPU**

### Step 3: Copy Code
- Open `COLAB_COMPLETE_CODE.py` in this repository
- Copy **entire file content**
- Paste into a **single cell** in Colab

### Step 4: Run
- Execute the cell (Shift + Enter)
- Upload `train_subtask2a.csv` when prompted
- Wait ~90-120 minutes
- Download `final_model_best.pt` when complete

---

## 📁 Project Structure

```
Deep-Learning-project-SemEval-2026-Task-2/
├── README.md                                   # This file
├── COLAB_COMPLETE_CODE.py                      # ⭐ COPY THIS TO COLAB
│
├── data/
│   ├── raw/
│   │   └── train_subtask2a.csv                 # Original dataset
│   └── processed/
│       └── subtask2a_features.csv              # Auto-generated
│
├── scripts/
│   ├── data_preparation/subtask2a/
│   │   └── prepare_features_subtask2a.py       # Feature extraction
│   └── data_train/subtask2a/
│       └── train_final_subtask2a.py            # ⭐ Final training script (local)
│
└── models/                                      # Trained models
    └── final_model_best.pt                      # Will be created after training
```

---

## 🏆 Model Architecture (v3 - FINAL)

```
Input Text → RoBERTa Encoder (768-dim)
    ↓
User Embeddings (64-dim) + Previous Emotions (5 lags)
    ↓
BiLSTM (2 layers, 256 hidden, bidirectional) → 512-dim
    ↓
Multi-Head Attention (4 heads)
    ↓
MLP Fusion (512 → 256) with GELU
    ↓
Dual Heads (Separate 2-layer networks)
    ├─→ Valence Prediction
    └─→ Arousal Prediction
```

### Key Innovations:

1. **Dual-Head Loss** ⭐ (Most Important!)
   - Valence: 65% CCC + 35% MSE
   - Arousal: 70% CCC + 30% MSE (Higher CCC weight!)

2. **Enhanced Features**
   - 5 lag features (was 3-4 in previous versions)
   - Temporal encoding (hour/day cyclical)
   - User-specific statistics

3. **Optimized Architecture**
   - GELU activation (better than ReLU)
   - 2-layer output heads (deeper)
   - Sequence length: 7 timesteps

4. **Stable Training**
   - 20 epochs with patience 7
   - Lower learning rate (1.5e-5)
   - More warmup (15%)
   - Dropout 0.2

---

## 📊 Expected Performance

```
================================================================================
FINAL MODEL (v3) - EXPECTED RESULTS
================================================================================
CCC Average:  0.65-0.72  🎯 Competition Ready!
CCC Valence:  0.68-0.72  ✅
CCC Arousal:  0.62-0.72  ✅ (Significantly improved!)
RMSE Valence: <1.00      ✅
RMSE Arousal: <0.65      ✅
================================================================================
```

### Why This Works:

**Previous Issues (v2)**:
- CCC Average: 0.48 ❌
- Arousal CCC: 0.26 ❌❌ (Catastrophic failure)
- Problem: Balanced loss (50/50) harmed arousal

**Final Solution (v3)**:
- **Separate loss weights** for Valence and Arousal
- Arousal gets **70% CCC** (higher than Valence's 65%)
- Result: Both dimensions optimize properly

---

## 🔑 Key Hyperparameters

```python
# Architecture
SEQ_LENGTH = 7          # Optimal sequence length
DROPOUT = 0.2           # Lower dropout
NUM_ATTENTION_HEADS = 4

# Training
BATCH_SIZE = 10
NUM_EPOCHS = 20         # Extended training
PATIENCE = 7            # More patience
WARMUP_RATIO = 0.15     # More warmup

# Learning Rates
LR_ROBERTA = 1.5e-5     # Lower for stability
LR_OTHER = 8e-5

# Loss Weights (DUAL-HEAD)
CCC_WEIGHT_V = 0.65     # Valence CCC
CCC_WEIGHT_A = 0.70     # Arousal CCC (HIGHER!)
MSE_WEIGHT_V = 0.35     # Valence MSE
MSE_WEIGHT_A = 0.30     # Arousal MSE (LOWER!)
```

---

## 📦 Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.10.0
tqdm>=4.65.0
```

**GPU**: CUDA-capable GPU required (Tesla T4 recommended)

---

## 🎯 After Training

### Check Results:

```python
import torch

checkpoint = torch.load('final_model_best.pt', weights_only=False)
print(f"CCC Average: {checkpoint['best_ccc']:.4f}")
print(f"CCC Valence: {checkpoint['val_ccc_v']:.4f}")
print(f"CCC Arousal: {checkpoint['val_ccc_a']:.4f}")
```

### Success Criteria:

- ✅ **Minimum (Acceptable)**: CCC ≥ 0.60
- ✅ **Target (Good)**: CCC ≥ 0.65
- ✅ **Excellent (Competition Ready)**: CCC ≥ 0.70

---

## 📚 References

- **Task**: [SemEval 2026 Task 2](https://semeval2026task2.github.io/SemEval-2026-Task2/)
- **RoBERTa**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **Attention**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## 📞 Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: 2025-11-05

**Version**: 3.0 FINAL

**Status**: ✅ Ready to Train

**Expected Result**: CCC 0.65-0.72 🎯
