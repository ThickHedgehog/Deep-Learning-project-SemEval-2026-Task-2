# ============================================================================
# Subtask 2a - Ensemble Weight Analysis
# ============================================================================
#
# Analyzes and calculates ensemble weights from trained models
#
# Use this after training 3 models with seeds 42, 123, 777
#
# Actual individual CCCs: 0.5053, 0.5330, 0.6554 (avg: 0.5646)
# Expected ensemble CCC: 0.5846-0.6046
#
# Model files required:
#   - subtask2a_seed42_best.pt  (CCC 0.5053, Epoch 16, 1.42 GB)
#   - subtask2a_seed123_best.pt (CCC 0.5330, Epoch 18, 1.44 GB)
#   - subtask2a_seed777_best.pt (CCC 0.6554, Epoch 9,  1.44 GB)
#
# ============================================================================

"""
Subtask 2a - Ensemble Weight Analysis
======================================
Analyzes performance of 3 trained models and calculates ensemble weights

Strategy: Weighted averaging based on validation performance

Individual Model Performance:
  - seed42:  CCC 0.5053 (Valence: 0.6532, Arousal: 0.3574)
  - seed123: CCC 0.5330 (Valence: 0.6298, Arousal: 0.4362)
  - seed777: CCC 0.6554 (Valence: 0.7593, Arousal: 0.5516) ⭐

Ensemble Weights (Performance-based):
  - seed42:  29.8%
  - seed123: 31.5%
  - seed777: 38.7% (highest weight due to superior performance)

Expected Ensemble Boost: +0.020 ~ +0.040 CCC
"""

import torch
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
from google.colab import drive, files
import os

print('='*80)
print('Subtask 2a - Ensemble Weight Analysis')
print('='*80)

# ===== DEVICE SETUP =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# ===== MOUNT GOOGLE DRIVE =====
print('\n' + '='*80)
print('MOUNTING GOOGLE DRIVE')
print('='*80)
drive.mount('/content/drive')
print('✓ Google Drive mounted!')

# ===== CONFIGURATION =====
DRIVE_MODEL_PATH = '/content/drive/MyDrive/models'  # Change this to your Drive path

MODEL_PATHS = {
    'seed42': f'{DRIVE_MODEL_PATH}/subtask2a_seed42_best.pt',
    'seed123': f'{DRIVE_MODEL_PATH}/subtask2a_seed123_best.pt',
    'seed777': f'{DRIVE_MODEL_PATH}/subtask2a_seed777_best.pt'
}

print('\n' + '='*80)
print('LOADING MODELS FROM DRIVE')
print('='*80)

# Verify files exist
for seed_name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f'❌ ERROR: File not found: {path}')
        print(f'Please upload {seed_name} model to Google Drive: {DRIVE_MODEL_PATH}/')
        raise FileNotFoundError(f'Missing: {path}')
    else:
        file_size_gb = os.path.getsize(path) / (1024**3)
        print(f'✓ Found {seed_name}: {file_size_gb:.2f} GB')

# Load model checkpoints (Load to CPU for memory efficiency)
checkpoints = {}
for seed_name, path in MODEL_PATHS.items():
    print(f'\nLoading {seed_name}...')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    checkpoints[seed_name] = checkpoint
    print(f'✓ {seed_name} loaded')

print('\n' + '='*80)
print('MODEL PERFORMANCE SUMMARY')
print('='*80)
for seed_name, checkpoint in checkpoints.items():
    print(f'\n{seed_name}:')
    print(f'  CCC Average: {checkpoint["best_ccc"]:.4f}')
    print(f'  CCC Valence: {checkpoint["val_ccc_v"]:.4f}')
    print(f'  CCC Arousal: {checkpoint["val_ccc_a"]:.4f}')
    print(f'  Best Epoch: {checkpoint["epoch"]}')
    print(f'  RMSE Valence: {checkpoint.get("val_rmse_v", 0):.4f}')
    print(f'  RMSE Arousal: {checkpoint.get("val_rmse_a", 0):.4f}')

# Calculate averages
avg_ccc = np.mean([ckpt['best_ccc'] for ckpt in checkpoints.values()])
avg_val = np.mean([ckpt['val_ccc_v'] for ckpt in checkpoints.values()])
avg_aro = np.mean([ckpt['val_ccc_a'] for ckpt in checkpoints.values()])

print('\n' + '-'*80)
print('INDIVIDUAL MODEL AVERAGE:')
print(f'  CCC Average: {avg_ccc:.4f}')
print(f'  CCC Valence: {avg_val:.4f}')
print(f'  CCC Arousal: {avg_aro:.4f}')

print('\n' + '='*80)

# ===== WEIGHTED ENSEMBLE =====
print('CALCULATING ENSEMBLE WEIGHTS')
print('='*80)

# Calculate weights based on performance
cccs = {name: ckpt['best_ccc'] for name, ckpt in checkpoints.items()}
total_ccc = sum(cccs.values())
weights = {name: ccc / total_ccc for name, ccc in cccs.items()}

print('\nPerformance-based Weights:')
for name, weight in weights.items():
    print(f'  {name}: {weight*100:5.1f}% (CCC: {cccs[name]:.4f})')

# Expected ensemble improvement
expected_boost_min = 0.020
expected_boost_max = 0.040
expected_ensemble_min = avg_ccc + expected_boost_min
expected_ensemble_max = avg_ccc + expected_boost_max

print(f'\nExpected Ensemble Performance:')
print(f'  Individual Average: {avg_ccc:.4f}')
print(f'  Expected Boost: +{expected_boost_min:.3f} ~ +{expected_boost_max:.3f}')
print(f'  Expected Ensemble: {expected_ensemble_min:.4f} ~ {expected_ensemble_max:.4f}')

print('\n' + '='*80)

# ===== SAVE RESULTS TO JSON =====
print('SAVING RESULTS')
print('='*80)

# Create results dictionary
results = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'device': str(device),
    'individual_models': {
        'seed42': {
            'ccc': float(checkpoints['seed42']['best_ccc']),
            'valence_ccc': float(checkpoints['seed42']['val_ccc_v']),
            'arousal_ccc': float(checkpoints['seed42']['val_ccc_a']),
            'epoch': int(checkpoints['seed42']['epoch']),
            'rmse_valence': float(checkpoints['seed42'].get('val_rmse_v', 0)),
            'rmse_arousal': float(checkpoints['seed42'].get('val_rmse_a', 0))
        },
        'seed123': {
            'ccc': float(checkpoints['seed123']['best_ccc']),
            'valence_ccc': float(checkpoints['seed123']['val_ccc_v']),
            'arousal_ccc': float(checkpoints['seed123']['val_ccc_a']),
            'epoch': int(checkpoints['seed123']['epoch']),
            'rmse_valence': float(checkpoints['seed123'].get('val_rmse_v', 0)),
            'rmse_arousal': float(checkpoints['seed123'].get('val_rmse_a', 0))
        },
        'seed777': {
            'ccc': float(checkpoints['seed777']['best_ccc']),
            'valence_ccc': float(checkpoints['seed777']['val_ccc_v']),
            'arousal_ccc': float(checkpoints['seed777']['val_ccc_a']),
            'epoch': int(checkpoints['seed777']['epoch']),
            'rmse_valence': float(checkpoints['seed777'].get('val_rmse_v', 0)),
            'rmse_arousal': float(checkpoints['seed777'].get('val_rmse_a', 0))
        }
    },
    'averages': {
        'ccc': float(avg_ccc),
        'valence_ccc': float(avg_val),
        'arousal_ccc': float(avg_aro)
    },
    'ensemble': {
        'weights': {
            'seed42': float(weights['seed42']),
            'seed123': float(weights['seed123']),
            'seed777': float(weights['seed777'])
        },
        'expected_boost_min': float(expected_boost_min),
        'expected_boost_max': float(expected_boost_max),
        'expected_ccc_min': float(expected_ensemble_min),
        'expected_ccc_max': float(expected_ensemble_max)
    },
    'notes': {
        'method': 'Performance-based weighted averaging',
        'boost_range': '+0.020 ~ +0.040 CCC',
        'best_model': 'seed777 (CCC 0.6554)',
        'recommendation': 'Use weighted ensemble for best performance'
    }
}

# Save to JSON file
json_filename = 'ensemble_results.json'
with open(json_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✓ Results saved to: {json_filename}')

# Download JSON file
print('\nDownloading results file...')
files.download(json_filename)
print('✓ Download complete!')

print('\n' + '='*80)
print('EXPECTED RESULTS')
print('='*80)
print('ACTUAL INDIVIDUAL MODELS:')
print('  seed42:  CCC 0.5053 (Valence: 0.6532, Arousal: 0.3574)')
print('  seed123: CCC 0.5330 (Valence: 0.6298, Arousal: 0.4362)')
print('  seed777: CCC 0.6554 (Valence: 0.7593, Arousal: 0.5516) ⭐')
print('  Average: CCC 0.5646')
print()
print('EXPECTED ENSEMBLE:')
print('  Weighted ensemble: CCC 0.5846-0.6046')
print('  Simple average:    CCC 0.5826-0.6026')
print()
print('Recommendation: Use weighted ensemble for best performance')
print('Expected improvement: +0.020 ~ +0.040 CCC over individual average')
print('='*80)

# ===== OPTIONAL: TEST DATA PREDICTION =====
"""
To generate actual predictions on test data, you need:

1. Test dataset (test_subtask2a.csv)
2. Full model architecture (FinalEmotionModel class)
3. Preprocessing pipeline (tokenizer, feature engineering)

Example code for test prediction:

# Load test data
test_df = pd.read_csv('test_subtask2a.csv')
# ... preprocess test_df same as training ...

# Create test dataset and loader
test_dataset = EmotionDataset(test_df, tokenizer, SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get predictions from each model
all_predictions = {}

for seed_name, checkpoint in checkpoints.items():
    print(f'\\nGenerating predictions with {seed_name}...')

    # Load model
    model = FinalEmotionModel(num_users=test_dataset.num_users)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Move to GPU for faster inference
    model.eval()

    valence_preds = []
    arousal_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_idx = batch['user_idx'].to(device)
            temporal_features = batch['temporal_features'].to(device)
            user_stats = batch['user_stats'].to(device)
            text_features = batch['text_features'].to(device)

            valence_pred, arousal_pred = model(
                input_ids, attention_mask, user_idx,
                temporal_features, user_stats, text_features
            )

            valence_preds.extend(valence_pred.cpu().numpy())
            arousal_preds.extend(arousal_pred.cpu().numpy())

    all_predictions[seed_name] = {
        'valence': np.array(valence_preds),
        'arousal': np.array(arousal_preds)
    }

# Weighted ensemble
ensemble_valence = np.zeros_like(all_predictions['seed42']['valence'])
ensemble_arousal = np.zeros_like(all_predictions['seed42']['arousal'])

for seed_name, preds in all_predictions.items():
    weight = weights[seed_name]
    ensemble_valence += weight * preds['valence']
    ensemble_arousal += weight * preds['arousal']

print('\\n' + '='*80)
print('ENSEMBLE PREDICTION COMPLETE')
print('='*80)

# Save predictions
submission = pd.DataFrame({
    'text_id': test_df['text_id'],
    'valence': ensemble_valence.flatten(),
    'arousal': ensemble_arousal.flatten()
})

submission.to_csv('ensemble_predictions.csv', index=False)
print('✓ Predictions saved to: ensemble_predictions.csv')

# Download submission file
files.download('ensemble_predictions.csv')
"""

print('\n' + '='*80)
print('ENSEMBLE WEIGHT CALCULATION COMPLETE')
print('='*80)
print('✓ All models loaded successfully')
print('✓ Ensemble weights calculated')
print('✓ Results saved to JSON')
print('✓ Expected performance analyzed')
print()
print('Next steps (if needed):')
print('  1. Load test data (test_subtask2a.csv)')
print('  2. Implement full model architecture')
print('  3. Generate predictions using ensemble weights')
print('  4. Save predictions for submission')
print('='*80)
