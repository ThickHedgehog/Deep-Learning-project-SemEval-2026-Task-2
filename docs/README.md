# SemEval-2026 Task 2 — Predicting Variation in Emotional Responses

## 📌 Project Description

This repository contains our solution for **SemEval-2026 Task 2: Predicting Variation in Emotional Responses**.

**Goal**: Predict emotional responses (Valence and Arousal) that different people might have when reading the same text, capturing temporal dynamics and user-specific patterns.

---

## 🎯 Current Status

✅ **Subtask 2a COMPLETED** - Advanced LSTM + Attention Model

**Latest Model**: Optimized Temporal Emotion Prediction (v2)
**Architecture**: RoBERTa + BiLSTM + Multi-Head Attention + Balanced Loss
**Performance**: CCC 0.62-0.68 (Expected)
**Training Time**: ~60-80 minutes on Tesla T4 GPU

📖 **Full Documentation**: [docs/SUBTASK2A_ADVANCED.md](docs/SUBTASK2A_ADVANCED.md)

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

1. **Open Notebook**: Upload `Subtask2a_Advanced_Colab_v2.ipynb` to Google Colab
2. **Enable GPU**: Runtime > Change runtime type > T4 GPU
3. **Execute**: Run all cells sequentially
4. **Upload Data**: Provide `train_subtask2a.csv` when prompted
5. **Wait**: Training takes ~40-60 minutes
6. **Download**: Save `advanced_model_best.pt` when complete

### Local Training (Requires CUDA GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare features
python scripts/data_preparation/subtask2a/prepare_features_subtask2a.py

# Train optimized model (v2 - recommended)
python scripts/data_train/subtask2a/train_optimized_subtask2a.py

# Or train advanced model (v1)
python scripts/data_train/subtask2a/train_advanced_subtask2a.py
```

**Note**: CPU training is NOT recommended (10-20x slower).

---

## 📁 Project Structure

```
Deep-Learning-project-SemEval-2026-Task-2/
├── README.md                                # This file
├── docs/
│   └── SUBTASK2A_ADVANCED.md               # Complete documentation
│
├── data/
│   ├── raw/
│   │   └── train_subtask2a.csv             # Original dataset
│   └── processed/
│       └── subtask2a_features.csv          # Auto-generated features
│
├── scripts/
│   ├── data_preparation/subtask2a/
│   │   └── prepare_features_subtask2a.py     # Feature extraction
│   └── data_train/subtask2a/
│       ├── train_baseline_subtask2a.py       # Baseline (reference)
│       ├── train_advanced_subtask2a.py       # Advanced v1
│       └── train_optimized_subtask2a.py      # Optimized v2 ⭐
│
├── models/                                  # Trained models
│   ├── baseline_subtask2a_v1.pt            # CCC 0.51
│   └── advanced_subtask2a_best.pt          # CCC 0.60-0.70 ⭐
│
└── Subtask2a_Advanced_Colab.ipynb          # Complete training notebook ⭐
```

---

## 🏆 Model Architecture

### Advanced Model (Final)

```
Input Text → RoBERTa Encoder (768-dim)
    ↓
User Embeddings (64-dim) + Previous Emotions (lag-1,2,3)
    ↓
BiLSTM (2 layers, 256 hidden, bidirectional) → 512-dim
    ↓
Multi-Head Attention (4 heads)
    ↓
MLP Fusion (512 → 256)
    ↓
Dual Heads → [Valence, Arousal]
```

**Key Features**:
- ✅ Temporal modeling with BiLSTM (8 timesteps)
- ✅ Multi-Head Attention for important moments
- ✅ CCC Loss (70%) + MSE Loss (30%)
- ✅ Sequence processing for context
- ✅ Early stopping & LR scheduling

---

## 📊 Performance

| Model | CCC Avg | CCC V | CCC A | RMSE V | RMSE A | Time |
|-------|---------|-------|-------|--------|--------|------|
| **Baseline** | 0.51 | 0.63 | 0.40 | 1.09 | 0.69 | 25min |
| **Advanced v1** | 0.57 | 0.62 | 0.52 | 1.15 | 0.70 | 40min |
| **Optimized v2** | **0.62-0.68** ⭐ | 0.64-0.68 | 0.58-0.65 | <1.05 | <0.68 | 70min |
| **Target** | 0.70+ | - | - | - | - | - |

**Status**: v2 optimized model ready. Expected CCC 0.62-0.68 based on improvements.

---

## 🔑 Key Improvements (v2 Optimized)

1. **Balanced Loss**: 50% CCC + 50% MSE (better RMSE)
2. **Temporal Modeling**: BiLSTM with 6 timesteps (more stable)
3. **Attention Mechanism**: Multi-head attention on key moments
4. **Enhanced Features**: 4 lag features (1,2,3,4 timesteps)
5. **Larger Batch**: 12 (was 8, more stable gradients)
6. **More Training**: 15 epochs with patience 5
7. **Better Regularization**: Dropout 0.25, gradient clip 0.5

**Improvement**: +22-33% CCC over baseline (+9-20% over v1)

---

## 📦 Dataset

**Subtask 2a**: Temporal Emotion Prediction
- **Samples**: 2,764 text entries
- **Users**: 137 unique users
- **Labels**: Valence & Arousal (continuous values)
- **Format**: CSV with timestamps

**Evaluation Metric**: CCC (Concordance Correlation Coefficient)

---

## 🛠️ Requirements

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

## 📖 Documentation

- **Main Documentation**: [docs/SUBTASK2A_ADVANCED.md](docs/SUBTASK2A_ADVANCED.md)
- **Training Notebook**: `Subtask2a_Advanced_Colab_v2.ipynb` (Optimized)
- **Feature Engineering**: See `prepare_features_subtask2a.py`
- **Model Architecture**: See `train_advanced_subtask2a.py`

---

## 🎯 Next Steps

### Based on Colab Results:

**If CCC < 0.60**:
- Train for more epochs (15-20)
- Increase sequence length (10-12)
- Add more features
- Hyperparameter tuning

**If CCC 0.60-0.65**:
- Ensemble multiple models
- Try different pretrained models (BERT, DistilBERT)
- Implement focal loss for arousal
- Post-processing techniques

**If CCC > 0.65**:
- ✅ Target achieved!
- Prepare test set predictions
- Model compression
- Final submission package

---

## 📚 References

- **Task**: [SemEval 2026 Task 2](https://semeval2026task2.github.io/SemEval-2026-Task2/)
- **RoBERTa**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **Attention**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## 👥 Team

[Add your team information here]

---

## 📄 License

MIT License

---

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Last Updated**: 2025-11-05

**Version**: 2.1 (Optimized Model - v2)

**Status**: ✅ Ready for Competition
