# Part 4: Submission Guide

**Last Updated**: November 21, 2025
**Status**: Ready for test data release
**Expected Performance**: CCC 0.5846-0.6046

---

## 📋 Overview

This guide walks you through the complete process of generating predictions for Subtask 2a and submitting to the SemEval 2026 Task 2 competition.

---

## ✅ Prerequisites

### 1. Trained Models (Complete ✅)
```
models/
├── subtask2a_seed42_best.pt   (CCC 0.5053, 1.5 GB)
├── subtask2a_seed123_best.pt  (CCC 0.5330, 1.5 GB)
└── subtask2a_seed777_best.pt  (CCC 0.6554, 1.5 GB)

Status: ✅ All 3 models trained and ready
```

### 2. Ensemble Weights (Complete ✅)
```
results/subtask2a/ensemble_results.json

Weights:
- seed42:  29.8%
- seed123: 31.5%
- seed777: 38.7%

Status: ✅ Calculated and saved
```

### 3. Prediction Script (Complete ✅)
```
scripts/data_analysis/subtask2a/predict_test_subtask2a.py

Status: ✅ Created and ready to use
```

### 4. Test Data (Pending ⏳)
```
test_subtask2a.csv

Expected columns:
- user_id
- text_id
- text
- timestamp
- (possibly more)

Status: ⏳ Awaiting release (expected mid-December)
```

---

## 🚀 Step-by-Step Process

### Step 1: Obtain Test Data

**When**: Mid-December 2025 (expected)

**Where**:
- Official website: https://semeval2026task2.github.io/SemEval-2026-Task2/
- Codabench: https://www.codabench.org/competitions/9963/

**What to Download**:
- `test_subtask2a.csv` - Test data file
- Check competition page for announcements

**Important**:
- Test data may require registration
- Download as soon as available
- Verify file integrity
- Check format matches training data

---

### Step 2: Prepare Environment

#### Option A: Google Colab (Recommended)

**Setup**:
```python
# 1. Upload files to Colab
from google.colab import files

# Upload these files:
# - test_subtask2a.csv (test data)
# - subtask2a_seed42_best.pt (models)
# - subtask2a_seed123_best.pt
# - subtask2a_seed777_best.pt
# - ensemble_results.json
# - predict_test_subtask2a.py (prediction script)

uploaded = files.upload()

# 2. Install dependencies
!pip install transformers torch pandas numpy scipy scikit-learn tqdm

# 3. Verify uploads
!ls -lh
```

**Alternative: Use Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy files from Drive
!cp /content/drive/MyDrive/models/*.pt .
!cp /content/drive/MyDrive/test_subtask2a.csv .
!cp /content/drive/MyDrive/ensemble_results.json results/subtask2a/
```

#### Option B: Local Machine

**Requirements**:
```bash
pip install torch transformers pandas numpy scipy scikit-learn tqdm
```

**File Organization**:
```
working_directory/
├── models/
│   ├── subtask2a_seed42_best.pt
│   ├── subtask2a_seed123_best.pt
│   └── subtask2a_seed777_best.pt
├── results/subtask2a/
│   └── ensemble_results.json
├── test_subtask2a.csv
└── predict_test_subtask2a.py
```

---

### Step 3: Run Prediction Script

**Command**:
```bash
python predict_test_subtask2a.py
```

**What Happens**:
```
1. Loads ensemble weights
2. Loads and preprocesses test data
3. Creates dataset and dataloader
4. Loads 3 trained models (seed42, 123, 777)
5. Generates predictions for each model
6. Combines predictions using weighted ensemble
7. Aggregates by user_id (one prediction per user)
8. Saves pred_subtask2a.csv
```

**Expected Output**:
```
================================================================================
Subtask 2a - Test Data Prediction with Ensemble
================================================================================

Using device: cuda
GPU: Tesla T4
Memory: 15.78 GB

=== Loading Ensemble Weights ===
✓ Loaded ensemble weights:
  seed42:  0.2983
  seed123: 0.3147
  seed777: 0.3870

=== Loading Test Data ===
✓ Loaded test data: XXXX samples
Columns: ['user_id', 'text_id', 'text', 'timestamp', ...]

=== Preprocessing Test Data ===
Extracting text features...
100%|████████████████████| XXXX/XXXX [00:XX<00:00, XX.XXit/s]
✓ Created 15 text features
✓ Created 5 lag features
✓ Created 12 user statistics

=== Creating Dataset ===
Test dataset: XXXX samples, YY users

=== Generating Predictions with Ensemble ===

Loading model: seed42
✓ Loaded checkpoint (CCC: 0.5053, Epoch: 16)
Generating predictions with seed42...
seed42 prediction: 100%|████████████| XX/XX [XX:XX<00:00, X.XXit/s]
✓ seed42 predictions complete
  Valence range: [-X.XXX, X.XXX]
  Arousal range: [-X.XXX, X.XXX]

Loading model: seed123
✓ Loaded checkpoint (CCC: 0.5330, Epoch: 18)
Generating predictions with seed123...
seed123 prediction: 100%|████████████| XX/XX [XX:XX<00:00, X.XXit/s]
✓ seed123 predictions complete

Loading model: seed777
✓ Loaded checkpoint (CCC: 0.6554, Epoch: 9)
Generating predictions with seed777...
seed777 prediction: 100%|████████████| XX/XX [XX:XX<00:00, X.XXit/s]
✓ seed777 predictions complete

=== Creating Weighted Ensemble ===
seed42: weight 0.2983
seed123: weight 0.3147
seed777: weight 0.3870

✓ Ensemble predictions created
  Valence range: [-X.XXX, X.XXX]
  Arousal range: [-X.XXX, X.XXX]

=== Aggregating Predictions by User ===
✓ Final predictions: YY users

================================================================================
PREDICTION COMPLETE
================================================================================
✓ Saved predictions to: pred_subtask2a.csv
✓ Number of users: YY

Submission format:
   user_id  pred_state_change_valence  pred_state_change_arousal
0        1                      0.523                      0.134
1        3                     -0.234                      0.456
2        5                      1.234                     -0.123
...

Statistics:
  Valence - Mean: X.XXX, Std: X.XXX
  Arousal - Mean: X.XXX, Std: X.XXX

================================================================================
Next Steps:
1. Create submission.zip with this pred_subtask2a.csv file
2. Upload to Codabench: https://www.codabench.org/competitions/9963/
3. Wait for evaluation results
================================================================================
```

**Estimated Time**:
- Small test set (<500 samples): 5-10 minutes
- Medium test set (500-2000 samples): 10-20 minutes
- Large test set (>2000 samples): 20-40 minutes

---

### Step 4: Verify Submission File

**Check File Format**:
```bash
head pred_subtask2a.csv
```

**Expected Format**:
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
1,0.523,0.134
3,-0.234,0.456
5,1.234,-0.123
...
```

**Verification Checklist**:
```
✅ File name: pred_subtask2a.csv (exact, case-sensitive)
✅ Header row: user_id,pred_state_change_valence,pred_state_change_arousal
✅ Column order: user_id, valence, arousal (MUST match exactly)
✅ No missing values (no NaN, no empty cells)
✅ One row per user_id
✅ Predictions are floats (not strings)
✅ Reasonable value ranges:
   - Valence change: typically -4.0 to +4.0
   - Arousal change: typically -2.0 to +2.0
```

**Validation Script**:
```python
import pandas as pd

# Load submission
df = pd.read_csv('pred_subtask2a.csv')

# Check format
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Expected: [\'user_id\', \'pred_state_change_valence\', \'pred_state_change_arousal\']')

# Check for missing values
print(f'\nMissing values:')
print(df.isnull().sum())

# Check value ranges
print(f'\nValue ranges:')
print(f'Valence: [{df["pred_state_change_valence"].min():.3f}, {df["pred_state_change_valence"].max():.3f}]')
print(f'Arousal: [{df["pred_state_change_arousal"].min():.3f}, {df["pred_state_change_arousal"].max():.3f}]')

# Check duplicates
duplicates = df['user_id'].duplicated().sum()
print(f'\nDuplicate user_ids: {duplicates}')

if duplicates == 0 and df.isnull().sum().sum() == 0:
    print('\n✅ Submission file looks good!')
else:
    print('\n❌ Fix issues before submitting')
```

---

### Step 5: Create Submission ZIP

**Structure**:
```
submission.zip
└── pred_subtask2a.csv
```

**How to Create**:

**Option A: Command Line**
```bash
# Linux/Mac
zip submission.zip pred_subtask2a.csv

# Windows PowerShell
Compress-Archive -Path pred_subtask2a.csv -DestinationPath submission.zip

# Windows Command Prompt
# Use File Explorer: Right-click → Send to → Compressed (zipped) folder
```

**Option B: Python**
```python
import zipfile

with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('pred_subtask2a.csv')

print('✓ Created submission.zip')
```

**Option C: Google Colab**
```python
from google.colab import files
import zipfile

# Create ZIP
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('pred_subtask2a.csv')

# Download
files.download('submission.zip')
```

**Verify ZIP**:
```bash
# Linux/Mac/Windows with 7zip
unzip -l submission.zip

# Should show:
# Archive:  submission.zip
#   Length      Date    Time    Name
# ---------  ---------- -----   ----
#      XXXX  MM-DD-YYYY HH:MM   pred_subtask2a.csv
# ---------                     -------
#      XXXX                     1 file
```

---

### Step 6: Submit to Codabench

**URL**: https://www.codabench.org/competitions/9963/

**Process**:

1. **Login / Register**
   - Go to competition page
   - Click "Participate"
   - Create account or login
   - Accept terms and conditions

2. **Navigate to Submission**
   - Click "My Submissions" tab
   - Or "Submit" button

3. **Upload File**
   - Click "Choose File" or drag-and-drop
   - Select `submission.zip`
   - Add submission description (optional):
     ```
     Subtask 2a: RoBERTa-BiLSTM-Attention Ensemble (3 models)
     Expected CCC: 0.5846-0.6046
     ```

4. **Submit**
   - Click "Submit" button
   - Wait for upload confirmation

5. **Wait for Evaluation**
   - Processing time: Usually 5-30 minutes
   - Status will update automatically
   - Refresh page to check

6. **View Results**
   - Once "Finished", click on submission
   - View scores:
     - CCC (Overall)
     - CCC Valence
     - CCC Arousal
     - RMSE Valence
     - RMSE Arousal
   - Compare to leaderboard

---

## 📊 Expected Results

### Performance Expectations

Based on validation performance:

```
Metric                    | Expected Range  | Target
--------------------------|-----------------|--------
CCC (Overall)             | 0.5846-0.6046   | ✅ 0.53-0.55
CCC Valence               | 0.68-0.72       | Good
CCC Arousal               | 0.44-0.48       | Moderate
RMSE Valence              | 0.85-1.10       | Low is better
RMSE Arousal              | 0.68-0.78       | Low is better
```

### Interpretation

**If CCC 0.58-0.60** (Expected):
```
✅ Excellent! Exceeded target by 8-10%
✅ Competitive performance
✅ Ensemble worked as expected
```

**If CCC 0.55-0.58**:
```
✅ Good! Met or slightly exceeded target
✅ Solid performance
✅ Within expected range
```

**If CCC 0.50-0.55**:
```
⚠️ Acceptable, but lower than validation
⚠️ Possible distribution shift
⚠️ Still competitive
→ Analyze what went wrong
```

**If CCC < 0.50**:
```
❌ Lower than expected
❌ Investigate issues:
   - Data preprocessing differences?
   - Format errors?
   - Model loading issues?
→ Debug and resubmit
```

---

## 🔧 Troubleshooting

### Issue 1: Model Loading Error

**Error**:
```
RuntimeError: Error loading model state_dict
```

**Solutions**:
```python
# 1. Check device compatibility
checkpoint = torch.load(model_path, map_location='cpu')  # Load to CPU first

# 2. Check file corruption
import os
print(f'File size: {os.path.getsize(model_path) / 1024**3:.2f} GB')
# Should be ~1.5 GB per model

# 3. Re-download models if needed
```

### Issue 2: Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# 1. Reduce batch size
BATCH_SIZE = 8  # Or even 4

# 2. Use CPU (slower but works)
device = torch.device('cpu')

# 3. Process in smaller chunks
# Split test data and process separately
```

### Issue 3: Missing Columns in Test Data

**Error**:
```
KeyError: 'valence'
```

**Solution**:
```python
# Test data may not have valence/arousal columns
# Script handles this - it will use zero features
# Check the ⚠️ warning message in output
```

### Issue 4: Wrong Submission Format

**Error from Codabench**:
```
Submission format error: Expected columns [user_id, pred_state_change_valence, pred_state_change_arousal]
```

**Solutions**:
```python
# 1. Check column names (case-sensitive)
df.columns = ['user_id', 'pred_state_change_valence', 'pred_state_change_arousal']

# 2. Check column order
df = df[['user_id', 'pred_state_change_valence', 'pred_state_change_arousal']]

# 3. Remove index
df.to_csv('pred_subtask2a.csv', index=False)
```

### Issue 5: Duplicate user_ids

**Error**:
```
ValueError: Duplicate user_id found
```

**Solution**:
```python
# Keep only last prediction per user
df = df.sort_values('timestamp').groupby('user_id').last().reset_index()
```

---

## 📝 Post-Submission Checklist

After submission:

```
✅ Screenshot results page
✅ Save submission file (pred_subtask2a.csv)
✅ Save submission.zip
✅ Note submission ID and timestamp
✅ Document performance:
   - CCC overall
   - CCC valence
   - CCC arousal
   - Leaderboard rank
✅ If results differ from expected:
   - Analyze error patterns
   - Compare to validation set
   - Document insights
✅ Save for final report/paper
```

---

## 🎯 Multiple Submissions

**Allowed**: Yes, check competition rules for limits

**Strategy**:

1. **First Submission** (Baseline):
   - Use current ensemble (expected best)
   - Establish baseline score

2. **Second Submission** (if needed):
   - If first submission underperforms
   - Try different ensemble strategy:
     - Equal weights (1/3 each)
     - Use only best model (seed777)

3. **Third Submission** (if available):
   - Experimental approaches
   - Different aggregation methods

**Record Keeping**:
```
Submission Log:

Submission 1: [Date/Time]
- Description: 3-model ensemble with performance weights
- CCC: X.XXXX
- Valence CCC: X.XXXX
- Arousal CCC: X.XXXX
- Notes: [Observations]

Submission 2: [Date/Time]
- Description: [What changed]
- CCC: X.XXXX
- Comparison: [Better/Worse than #1, why?]
```

---

## 📅 Timeline

```
Now - Mid December:
✅ Preparation complete
✅ Scripts ready
✅ Models ready
⏳ Waiting for test data

Mid December:
□ Test data releases
□ Download immediately
□ Run prediction script (1-2 hours)
□ Verify submission file
□ Create submission.zip
□ Submit to Codabench

After Submission:
□ Document results
□ Analyze performance
□ Begin final report writing

January 9, 2026:
□ Final submission deadline
□ Ensure best submission is selected
```

---

## 🆘 Getting Help

**If you encounter issues**:

1. **Check Script Output**:
   - Read error messages carefully
   - Check file paths
   - Verify data format

2. **Test on Validation Set**:
   - Run script on training data
   - Verify it produces valid output
   - Check predictions make sense

3. **Contact**:
   - Task organizers: nisoni@cs.stonybrook.edu
   - Professor: [Your professor's email]
   - Team member: Coordinate issues

4. **Resources**:
   - Competition FAQ: Check Codabench page
   - Forum: Competition discussion board
   - Documentation: This guide + official docs

---

## ✅ Final Checklist

Before submission deadline:

```
Technical:
□ Test data downloaded
□ All 3 models accessible
□ Ensemble weights file present
□ Prediction script runs without errors
□ Output file has correct format
□ No missing or duplicate values
□ submission.zip created correctly

Administrative:
□ Registered on Codabench
□ Understood submission rules
□ Know submission deadline (Jan 9, 2026)
□ Have backup plan if issues arise

Documentation:
□ Saved all intermediate files
□ Documented process
□ Ready to write about results in final report
```

---

## 🎉 Success Indicators

You're ready to submit when:

```
✅ Script runs successfully start to finish
✅ Output file format validated
✅ Predictions look reasonable (not all zeros, not extreme values)
✅ ZIP file created correctly
✅ Can access Codabench competition page
✅ Confident in process
```

---

**Good luck with your submission!** 🚀

You've done excellent work - now just execute the final steps when test data arrives.

**Remember**: The process and learning matter more than the absolute rank. Document everything for your final report!

---

**Document Status**: ✅ COMPLETE
**Last Updated**: 2025-11-23
**Purpose**: Complete submission guide for Subtask 2a test data prediction
