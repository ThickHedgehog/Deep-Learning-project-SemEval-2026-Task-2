# Subtask 2a - Analysis Scripts

## 📊 Overview

This folder contains analysis scripts for Subtask 2a (Emotion Prediction).

---

## 📁 Files

### `analyze_ensemble_weights_subtask2a.py`

**Purpose**: Analyze ensemble model performance and calculate optimal weights

**Usage**: Google Colab with T4 GPU

**Input**:
- 3 trained models from Google Drive:
  - `subtask2a_seed42_best.pt`
  - `subtask2a_seed123_best.pt`
  - `subtask2a_seed777_best.pt`

**Output**:
- Performance summary for each model
- Ensemble weights (performance-based)
- Expected ensemble CCC range
- `ensemble_results.json` (downloadable)

**Expected Results**:
```
Individual Model Average: CCC 0.5646
Ensemble Weights: 29.8%, 31.5%, 38.7%
Expected Ensemble CCC: 0.5846-0.6046
```

---

## 🚀 Quick Start

```bash
# 1. Upload to Google Colab
# 2. Upload 3 model files to Google Drive
# 3. Run the script
# 4. Download ensemble_results.json
```

---

## 📈 Results

Results are saved to: `results/subtask2a/ensemble_results.json`

Contains:
- Individual model performance metrics
- Ensemble weights
- Expected performance ranges
- Timestamp and device info

---

**Last Updated**: 2025-11-14
