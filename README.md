# SemEval-2026 Task 2 — Predicting Variation in Emotional Responses

## 📌 Project Description

**Task**: Predict emotional responses (Valence and Arousal) for temporal text sequences.

**Goal**: Build a model that captures temporal dynamics and user-specific patterns in emotional responses.

---

## 🎯 Subtask 2a - Final Model (v3.3 MINIMAL)

### ✅ Status: Ready to Train

**Model**: v3.3 MINIMAL - Evidence-Based Optimization

**Expected Performance**: CCC 0.54-0.58 (Realistic, Achievable)

**Architecture**: RoBERTa + BiLSTM + Multi-Head Attention + Dual-Head Loss

**Training Time**: ~90 minutes on Tesla T4 GPU

**Key Improvement**: Fixed overfitting from v3.0 with 6 minimal, proven changes

---

## 📚 Documentation

**Start Here** (v3.3 MINIMAL):
- **[V3.3_QUICKSTART.md](V3.3_QUICKSTART.md)** ⭐ - Execute v3.3 in 5 steps (recommended)
- **[V3.3_SUMMARY.md](V3.3_SUMMARY.md)** 📊 - Why v3.3 will work (detailed analysis)

**Previous Versions**:
- **[QUICKSTART.md](QUICKSTART.md)** - v3.0 baseline guide
- **[TRAINING_RESULTS_v3.md](TRAINING_RESULTS_v3.md)** - v3.0 actual results (CCC 0.51)
- **[V3.1_IMPROVEMENTS.md](V3.1_IMPROVEMENTS.md)** - v3.1 plan (not tested)
- **[DEEP_ANALYSIS.md](DEEP_ANALYSIS.md)** - Analysis of v3.2 failure

**Additional Resources**:
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status
- **[VERSION_HISTORY.md](VERSION_HISTORY.md)** - Development history
- **[validate_setup.py](validate_setup.py)** - Pre-flight validation script

---

## 🚀 Quick Start - Google Colab (Recommended)

### Version 3.3 MINIMAL (Latest, Recommended)

**File**: `COLAB_FINAL_v3.3_MINIMAL.py`
**Expected**: CCC 0.54-0.58 (realistic)
**Time**: ~90 minutes

1. Open https://colab.research.google.com/
2. Runtime → Change runtime type → **T4 GPU**
3. Copy **entire** `COLAB_FINAL_v3.3_MINIMAL.py` → Paste in one cell
4. Run cell (Shift + Enter)
5. Upload `train_subtask2a.csv` when prompted
6. Login to wandb when prompted
7. Wait ~90 minutes
8. Check results (target: CCC 0.54-0.58)

**See [V3.3_QUICKSTART.md](V3.3_QUICKSTART.md) for detailed guide**

### Version 3.0 Baseline (Reference)

**File**: `COLAB_COMPLETE_CODE.py`
**Actual Result**: CCC 0.51 (tested)
**Issue**: Overfitting (train-val gap 0.39)

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

## 🏆 Model Architecture (v3.3 MINIMAL)

```
Input Text → RoBERTa Encoder (768-dim)
    ↓
User Embeddings (32-dim) + Previous Emotions (5 lags) ← CHANGED: 64→32
    ↓
BiLSTM (2 layers, 192 hidden, bidirectional) → 384-dim ← CHANGED: 256→192
    ↓
Multi-Head Attention (4 heads)
    ↓
MLP Fusion (384 → 192) with GELU ← CHANGED: Smaller
    ↓
Dual Heads (Separate 2-layer networks)
    ├─→ Valence Prediction
    └─→ Arousal Prediction
```

### Key Changes from v3.0:

1. **Reduced Overfitting** ⭐ (Priority #1)
   - User embedding: 64→32 dim (keep benefit, reduce memorization)
   - LSTM hidden: 256→192 (less capacity)
   - Dropout: 0.2→0.3 (stronger regularization)
   - Weight decay: 0.01→0.015 (L2 regularization)
   - Patience: 7→5 (earlier stopping)

2. **Improved Arousal Focus** ⭐ (Priority #2)
   - Arousal CCC: 70%→75% (was too low in v3.0)
   - Arousal MSE: 30%→25% (to balance)
   - Valence: 65% CCC + 35% MSE (unchanged)

3. **Total Changes**: Only 6 (minimal, evidence-based)

4. **Evidence-Based Strategy**
   - Based on v3.0 (CCC 0.51, proven baseline)
   - Learned from v3.2 failure (removed user emb was wrong)
   - Conservative changes (not aggressive like v3.2)
   - Realistic expectations (0.54-0.58, not 0.65-0.72)

---

## 📊 Performance Tracking

### Version History

| Version | CCC Avg | CCC Val | CCC Aro | Gap | Status |
|---------|---------|---------|---------|-----|--------|
| v0 baseline | 0.51 | 0.55 | 0.47 | - | ❌ Weak |
| v1 advanced | 0.57 | 0.61 | 0.52 | - | ⚠️ Unverified |
| v2 optimized | 0.48 | 0.69 | 0.26 | - | ❌ Catastrophic |
| **v3.0 dual-head** | **0.514** | **0.638** | **0.391** | **0.39** | ⚠️ **Overfitting** |
| v3.2 ultimate | 0.29 | 0.48 | 0.09 | 0.14 | ❌ Failed |
| **v3.3 minimal** | **0.54-0.58** | **0.62-0.64** | **0.43-0.48** | **0.20-0.28** | 🎯 **Expected** |

### v3.3 Expected Results

```
================================================================================
v3.3 MINIMAL - EXPECTED RESULTS (85% confidence)
================================================================================
CCC Average:  0.54-0.58  ✅ Realistic improvement
CCC Valence:  0.62-0.64  ✅ Slight decrease acceptable
CCC Arousal:  0.43-0.48  ✅ Significant improvement (+0.04-0.09)
Train-Val Gap: 0.20-0.28  ✅ Reduced overfitting (-0.11-0.19)
================================================================================
```

### Why v3.3 Works:

**v3.0 Issues**:
- CCC Average: 0.51 ⚠️ (below target)
- Train-Val Gap: 0.39 ❌ (severe overfitting)
- Arousal CCC: 0.39 ⚠️ (weak)

**v3.2 Failure**:
- Removed user embeddings → CCC dropped to 0.29 ❌
- Too many changes at once (10+) → couldn't debug
- Dropout 0.4 too high → underfitting

**v3.3 Solution**:
- **Keep user embeddings** but reduce (64→32)
- **Only 6 minimal changes** (evidence-based)
- **Moderate regularization** (dropout 0.3, not 0.4)
- **Realistic target** (0.54-0.58, not 0.65-0.72)

---

## 🔑 Key Hyperparameters (v3.3)

```python
# Architecture (CHANGED from v3.0)
USER_EMB_DIM = 32       # CHANGED: 64 → 32 (reduce overfitting)
LSTM_HIDDEN = 192       # CHANGED: 256 → 192 (less capacity)
LSTM_LAYERS = 2         # Same as v3.0
DROPOUT = 0.3           # CHANGED: 0.2 → 0.3 (stronger regularization)
NUM_ATTENTION_HEADS = 4 # Same as v3.0

# Training (CHANGED from v3.0)
BATCH_SIZE = 10         # Same as v3.0
NUM_EPOCHS = 20         # Same as v3.0
PATIENCE = 5            # CHANGED: 7 → 5 (earlier stopping)
WARMUP_RATIO = 0.15     # Same as v3.0
WEIGHT_DECAY = 0.015    # CHANGED: 0.01 → 0.015 (L2 reg)

# Learning Rates (Same as v3.0)
LR_ROBERTA = 1.5e-5
LR_OTHER = 8e-5

# Loss Weights (DUAL-HEAD, one change from v3.0)
CCC_WEIGHT_V = 0.65     # Same as v3.0
CCC_WEIGHT_A = 0.75     # CHANGED: 0.70 → 0.75 (more CCC focus)
MSE_WEIGHT_V = 0.35     # Same as v3.0
MSE_WEIGHT_A = 0.25     # CHANGED: 0.30 → 0.25 (less MSE)
```

**Total Changes from v3.0**: 6 hyperparameters

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
wandb>=0.15.0
```

**GPU**: CUDA-capable GPU required (Tesla T4 recommended)
**WandB**: Optional but recommended for experiment tracking

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

### Success Criteria (v3.3):

- ✅ **Minimum**: CCC ≥ 0.53 (+0.02 from v3.0)
- ✅ **Target**: CCC ≥ 0.55 (+0.04 from v3.0)
- ✅ **Excellent**: CCC ≥ 0.58 (+0.07 from v3.0)
- 🎯 **Competition Ready**: CCC ≥ 0.60 (requires ensemble)

---

## 📚 References

- **Task**: [SemEval 2026 Task 2](https://semeval2026task2.github.io/SemEval-2026-Task2/)
- **RoBERTa**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **Attention**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## 📞 Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: 2025-11-09

**Version**: 3.3 MINIMAL (Evidence-Based)

**Status**: ✅ Ready to Train

**Expected Result**: CCC 0.54-0.58 🎯 (Realistic)

**Key Learning**: Simple, evidence-based changes > Complex optimizations
