# Subtask 2a - Google Colab Training Scripts

This folder contains production-ready scripts for training models on Google Colab.

## Available Scripts

### ⭐⭐⭐ Recommended: Ensemble Training

**[ENSEMBLE_v3.0_COMPLETE.py](ENSEMBLE_v3.0_COMPLETE.py)** (850 lines)
- Complete training script for v3.0 ensemble
- Configurable seed (42, 123, 777)
- WandB integration
- Expected CCC: 0.510-0.515 per model

**[ENSEMBLE_PREDICTION.py](ENSEMBLE_PREDICTION.py)** (200 lines)
- Combines 3 models with weighted averaging
- Expected ensemble CCC: 0.530-0.550

### Baseline Scripts

**[COLAB_COMPLETE_CODE.py](COLAB_COMPLETE_CODE.py)**
- v3.0 baseline (seed=42)
- Actual result: CCC 0.5144 ✅
- Best single model

### Utilities

**[validate_setup.py](validate_setup.py)**
- Pre-flight validation script
- Checks dependencies, data, GPU

## Usage

### Step 1: Train 3 Models

```python
# Model 1: seed=42 (already trained, CCC 0.5144)
# Model 2: seed=123
RANDOM_SEED = 123  # Change in ENSEMBLE_v3.0_COMPLETE.py
# Run in Colab (~90 min)

# Model 3: seed=777
RANDOM_SEED = 777  # Change in ENSEMBLE_v3.0_COMPLETE.py
# Run in Colab (~90 min)
```

### Step 2: Ensemble Prediction

```python
# Run ENSEMBLE_PREDICTION.py
# Combines 3 models → Final CCC 0.530-0.550
```

## Complete Guide

See [docs/subtask2a/ENSEMBLE_GUIDE.md](../../../docs/subtask2a/ENSEMBLE_GUIDE.md) for detailed instructions.

---

**Last Updated**: 2025-11-14
**Status**: Production-ready
