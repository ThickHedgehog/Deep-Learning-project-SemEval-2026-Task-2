# Quick Start Guide - Final Model v3

## 🚀 Run in Google Colab (5 minutes setup)

### Step 1: Prepare Colab Environment
1. Open https://colab.research.google.com/
2. Create new notebook
3. Go to **Runtime → Change runtime type → T4 GPU**

### Step 2: Copy Training Code
1. Open `COLAB_COMPLETE_CODE.py` in this repository
2. Copy **entire file content** (608 lines)
3. Paste into **single cell** in Colab

### Step 3: Execute Training
1. Run the cell (Shift + Enter)
2. When prompted for WandB API key:
   - Go to https://wandb.ai/authorize
   - Copy your API key
   - Paste into Colab prompt
3. Upload `train_subtask2a.csv` when file upload widget appears
4. Wait ~90-120 minutes for training to complete

### Step 4: Monitor Progress
- WandB dashboard URL will be printed at start
- Watch real-time metrics:
  - CCC Average (target: ≥0.65)
  - CCC Valence (target: ≥0.68)
  - CCC Arousal (target: ≥0.62)
  - Training/validation losses

### Step 5: Download Results
After training completes:
1. Download `final_model_best.pt` from Colab Files panel
2. Check final metrics in WandB dashboard
3. Save WandB run URL for reference

---

## 📊 Expected Results

```
Target Performance (Competition Ready):
├─ CCC Average:  0.65-0.72  🎯
├─ CCC Valence:  0.68-0.72  ✅
├─ CCC Arousal:  0.62-0.72  ✅
├─ RMSE Valence: <1.00      ✅
└─ RMSE Arousal: <0.65      ✅
```

---

## 🔍 Verify Model Performance

```python
import torch

# Load checkpoint
checkpoint = torch.load('final_model_best.pt', weights_only=False)

# Print metrics
print(f"CCC Average: {checkpoint['best_ccc']:.4f}")
print(f"CCC Valence: {checkpoint['val_ccc_v']:.4f}")
print(f"CCC Arousal: {checkpoint['val_ccc_a']:.4f}")
```

Success Criteria:
- ✅ **Minimum**: CCC ≥ 0.60
- ✅ **Good**: CCC ≥ 0.65
- ✅ **Excellent**: CCC ≥ 0.70

---

## 🛠️ Local Training (Optional)

If you have local GPU:

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare features
python scripts/data_preparation/subtask2a/prepare_features_subtask2a.py

# Train model
python scripts/data_train/subtask2a/train_final_subtask2a.py
```

**Note**: Local script does NOT include WandB. Use COLAB_COMPLETE_CODE.py for WandB tracking.

---

## ❓ Troubleshooting

**WandB login fails**:
- Ensure you're logged into wandb.ai
- Get fresh API key from https://wandb.ai/authorize

**GPU not available**:
- Verify Runtime → Change runtime type → T4 GPU is selected
- Try restarting runtime

**File upload fails**:
- Ensure `train_subtask2a.csv` is in correct format
- Check file size (<100MB)

**Training stops early**:
- Check WandB logs for errors
- Verify GPU memory didn't overflow (batch size = 10)

---

## 📁 Output Files

After successful training:
- `final_model_best.pt` - Best model checkpoint
- WandB artifacts - Model versioned in WandB
- WandB dashboard - Complete training history and visualizations

---

**Last Updated**: 2025-11-08
**Version**: 3.0 FINAL
**Status**: ✅ Ready to Execute
