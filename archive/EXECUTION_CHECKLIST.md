# Execution Checklist - Final Model v3

Use this checklist to ensure successful training execution.

---

## ✅ Pre-Execution Checklist

### 1. Local Validation (Optional but Recommended)
- [ ] Run `python validate_setup.py` to check dependencies
- [ ] Verify all files are present
- [ ] Check training data exists: `data/raw/train_subtask2a.csv`

### 2. WandB Account Setup
- [ ] Create account at https://wandb.ai/ (if not done)
- [ ] Get API key from https://wandb.ai/authorize
- [ ] Keep API key ready for Colab login

### 3. File Preparation
- [ ] Open `COLAB_COMPLETE_CODE.py` in text editor
- [ ] Verify file is 608 lines
- [ ] Copy entire file content to clipboard

### 4. Data File
- [ ] Locate `data/raw/train_subtask2a.csv` on your computer
- [ ] Verify file size is ~579 KB (2,764 rows)
- [ ] Have file ready for upload

---

## 🚀 Execution Checklist (Google Colab)

### Step 1: Colab Setup
- [ ] Open https://colab.research.google.com/
- [ ] Click **+ New Notebook**
- [ ] Go to **Runtime** menu
- [ ] Select **Change runtime type**
- [ ] Choose **T4 GPU** from Hardware accelerator dropdown
- [ ] Click **Save**

### Step 2: Code Preparation
- [ ] Paste `COLAB_COMPLETE_CODE.py` content into **single cell**
- [ ] Verify code looks correct (imports at top, training loop, etc.)
- [ ] Do NOT split into multiple cells

### Step 3: Execute Training
- [ ] Click **Run cell** button (or press Shift + Enter)
- [ ] Wait for WandB login prompt
- [ ] Paste your WandB API key when prompted
- [ ] Press Enter to confirm

### Step 4: Data Upload
- [ ] Wait for file upload widget to appear
- [ ] Click **Choose Files** button
- [ ] Select `train_subtask2a.csv` from your computer
- [ ] Wait for upload to complete (green checkmark)

### Step 5: Monitor Training
- [ ] Note the WandB dashboard URL printed in output
- [ ] Open WandB dashboard in new tab
- [ ] Verify training has started (epoch 1/20 should appear)
- [ ] Bookmark WandB run URL for later reference

---

## 📊 During Training Checklist

### Monitor These Metrics (in WandB)
- [ ] `train/loss` - Should decrease over time
- [ ] `val/ccc_avg` - Should increase and reach 0.65+
- [ ] `val/ccc_valence` - Should reach 0.68+
- [ ] `val/ccc_arousal` - Should reach 0.62+
- [ ] `learning_rate` - Should decrease gradually
- [ ] `patience_counter` - Should reset when validation improves

### Expected Timeline
- [ ] Epoch 1/20: ~6-7 minutes
- [ ] Total time: 90-120 minutes
- [ ] Early stopping may trigger if no improvement for 7 epochs

### Warning Signs (Check if you see these)
- ⚠️ `val/ccc_avg` stuck below 0.50 after 5 epochs
- ⚠️ `val/ccc_arousal` below 0.30 (v2 failure pattern)
- ⚠️ Loss increasing instead of decreasing
- ⚠️ CUDA out of memory error
- ⚠️ Training stops before epoch 10

---

## ✅ Post-Training Checklist

### Step 1: Verify Completion
- [ ] Training completed successfully (no errors)
- [ ] Final epoch metrics printed
- [ ] "Best model saved" message appeared
- [ ] WandB artifact uploaded successfully

### Step 2: Check Performance
- [ ] Open WandB dashboard
- [ ] Locate best metrics:
  - [ ] **CCC Average ≥ 0.65** (Target)
  - [ ] **CCC Valence ≥ 0.68** (Target)
  - [ ] **CCC Arousal ≥ 0.62** (Target)
- [ ] Compare to v2 results (should be much better than 0.48)

### Step 3: Download Model
- [ ] In Colab, open **Files** panel (folder icon on left)
- [ ] Locate `final_model_best.pt` file
- [ ] Right-click → **Download**
- [ ] Save to `models/` folder locally
- [ ] Verify file size (~100-200 MB)

### Step 4: Verify Model Checkpoint
- [ ] Run verification code (see below)
- [ ] Confirm CCC metrics match WandB
- [ ] Check epoch number is reasonable (10-20)

### Step 5: Save Results
- [ ] Save WandB run URL
- [ ] Take screenshot of final metrics
- [ ] Note best epoch number
- [ ] Document any observations

---

## 🔍 Model Verification Code

After downloading `final_model_best.pt`, run this locally:

```python
import torch

# Load checkpoint
checkpoint = torch.load('models/final_model_best.pt', weights_only=False)

# Print all metrics
print("=" * 60)
print("FINAL MODEL METRICS")
print("=" * 60)
print(f"Best CCC Average: {checkpoint['best_ccc']:.4f}")
print(f"Best Epoch: {checkpoint['epoch']}")
print()
print("Valence Metrics:")
print(f"  CCC: {checkpoint['val_ccc_v']:.4f}")
print(f"  RMSE: {checkpoint['val_rmse_v']:.4f}")
print()
print("Arousal Metrics:")
print(f"  CCC: {checkpoint['val_ccc_a']:.4f}")
print(f"  RMSE: {checkpoint['val_rmse_a']:.4f}")
print("=" * 60)

# Success criteria
ccc_avg = checkpoint['best_ccc']
if ccc_avg >= 0.70:
    print("✅ EXCELLENT - Competition ready!")
elif ccc_avg >= 0.65:
    print("✅ GOOD - Target achieved!")
elif ccc_avg >= 0.60:
    print("⚠️ ACCEPTABLE - Consider minor tuning")
else:
    print("❌ BELOW TARGET - Check training logs")
```

---

## 📋 Success Criteria

### Minimum (Acceptable)
- [ ] CCC Average ≥ 0.60
- [ ] CCC Valence ≥ 0.60
- [ ] CCC Arousal ≥ 0.55

### Target (Good)
- [ ] CCC Average ≥ 0.65
- [ ] CCC Valence ≥ 0.68
- [ ] CCC Arousal ≥ 0.62

### Excellent (Competition Ready)
- [ ] CCC Average ≥ 0.70
- [ ] CCC Valence ≥ 0.70
- [ ] CCC Arousal ≥ 0.68

---

## 🛠️ Troubleshooting Checklist

### If WandB Login Fails
- [ ] Check internet connection
- [ ] Verify API key is correct
- [ ] Try logging in to wandb.ai in browser first
- [ ] Get fresh API key from https://wandb.ai/authorize

### If File Upload Fails
- [ ] Check file exists and is correct CSV
- [ ] Verify file size is ~579 KB
- [ ] Try refreshing Colab page and re-running
- [ ] Check Colab storage quota

### If GPU Not Available
- [ ] Verify **Runtime → Change runtime type → T4 GPU** is selected
- [ ] Try **Runtime → Factory reset runtime**
- [ ] Check Colab GPU quota (may be limited)
- [ ] Wait and try again later if quota exceeded

### If Training Fails Early
- [ ] Check WandB logs for error messages
- [ ] Verify data uploaded correctly
- [ ] Check GPU memory (should be sufficient with batch_size=10)
- [ ] Look for CUDA errors in output

### If Performance is Low (CCC < 0.60)
- [ ] Check if arousal CCC is very low (< 0.30) - indicates v2 pattern
- [ ] Verify dual-head loss weights are correct in code
- [ ] Check learning rate schedule in WandB
- [ ] Review training curves for instability
- [ ] Check if data was shuffled correctly

---

## 📝 Post-Execution Notes

### Things to Document
- [ ] Final CCC Average: ___________
- [ ] Final CCC Valence: ___________
- [ ] Final CCC Arousal: ___________
- [ ] Best Epoch: ___________
- [ ] Total Training Time: ___________
- [ ] WandB Run URL: ___________
- [ ] Any issues encountered: ___________
- [ ] Any observations: ___________

### Next Steps if Successful (CCC ≥ 0.65)
- [ ] Push trained model to Git LFS (if using)
- [ ] Update README with actual results
- [ ] Prepare for competition submission
- [ ] Test model on validation set (if available)

### Next Steps if Below Target (CCC < 0.65)
- [ ] Review WandB training curves
- [ ] Check if arousal is the problem (like v2)
- [ ] Consider adjusting loss weights
- [ ] Try increasing num_epochs to 25
- [ ] Consider ensemble with multiple runs

---

**Estimated Total Time**: 2-3 hours (including setup, training, verification)

**Last Updated**: 2025-11-08

**Version**: 3.0 FINAL

**Status**: Ready to Execute ✅
