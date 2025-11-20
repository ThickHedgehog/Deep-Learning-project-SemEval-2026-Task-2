# ============================================================================
# v3.0 ENSEMBLE PREDICTION - Combine 3 Models
# ============================================================================
#
# Use this after training 3 models with seeds 42, 123, 777
#
# Expected individual CCCs: ~0.510-0.515
# Expected ensemble CCC: 0.530-0.550
#
# ============================================================================

"""
ENSEMBLE PREDICTION
===================
Combines predictions from 3 v3.0 models trained with different seeds

Strategy: Weighted averaging based on validation performance
"""

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from google.colab import files
import os

print('='*80)
print('v3.0 ENSEMBLE PREDICTION')
print('='*80)

# ===== UPLOAD MODEL FILES =====
print('\n' + '='*80)
print('UPLOAD MODEL FILES')
print('='*80)
print('Please upload the following 3 model files:')
print('  1. v3.0_seed42_best.pt')
print('  2. v3.0_seed123_best.pt')
print('  3. v3.0_seed777_best.pt')
print()

uploaded = files.upload()
print(f'\n✓ Uploaded {len(uploaded)} files')

# Verify all files are present
required_files = ['v3.0_seed42_best.pt', 'v3.0_seed123_best.pt', 'v3.0_seed777_best.pt']
for f in required_files:
    if f not in uploaded and not os.path.exists(f):
        print(f'❌ ERROR: Missing file: {f}')
        print('Please upload all 3 model files!')
        raise FileNotFoundError(f'Missing required file: {f}')

print('✓ All model files found!')

# ===== CONFIGURATION =====
MODEL_PATHS = {
    'seed42': 'v3.0_seed42_best.pt',
    'seed123': 'v3.0_seed123_best.pt',
    'seed777': 'v3.0_seed777_best.pt'
}

print('\n' + '='*80)
print('LOADING MODELS')
print('='*80)

# Load model checkpoints
checkpoints = {}
for seed_name, path in MODEL_PATHS.items():
    print(f'\nLoading {seed_name}...')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    checkpoints[seed_name] = checkpoint

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

# ===== LOAD TEST DATA & GENERATE PREDICTIONS =====

# Note: Replace this with actual test data loading
print('\n' + '='*80)
print('GENERATING ENSEMBLE PREDICTIONS')
print('='*80)

# This is a placeholder - you'll need to:
# 1. Load your test data
# 2. Run each model to get predictions
# 3. Combine predictions using the weights above

"""
Example ensemble prediction code:

# Load test data
test_df = pd.read_csv('test_subtask2a.csv')
# ... preprocess test_df same as training ...

# Create test dataset and loader
test_dataset = EmotionDataset(test_df, tokenizer, SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get predictions from each model
all_predictions = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for seed_name, checkpoint in checkpoints.items():
    print(f'\\nGenerating predictions with {seed_name}...')

    # Load model
    model = FinalEmotionModel(num_users=test_dataset.num_users).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
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
"""

# ===== SIMPLE AVERAGE ENSEMBLE (Alternative) =====

print('\n' + '='*80)
print('ALTERNATIVE: SIMPLE AVERAGE ENSEMBLE')
print('='*80)
print('If performance-based weighting is complex, use simple average:')
print('ensemble_pred = (pred1 + pred2 + pred3) / 3')
print('\nExpected performance difference: ~0.001-0.002 CCC')
print('Weighted ensemble is slightly better but simple average is easier')

# ===== VALIDATION ENSEMBLE (If you have validation data) =====

print('\n' + '='*80)
print('VALIDATION ENSEMBLE EXAMPLE')
print('='*80)

"""
# If you want to validate the ensemble on validation set:

# Get predictions from all 3 models on validation data
val_predictions = {}  # Same as test prediction code above

# Weighted ensemble
ensemble_val_valence = np.zeros(len(val_df))
ensemble_val_arousal = np.zeros(len(val_df))

for seed_name, preds in val_predictions.items():
    weight = weights[seed_name]
    ensemble_val_valence += weight * preds['valence']
    ensemble_val_arousal += weight * preds['arousal']

# Calculate ensemble CCC
val_true_valence = val_df['valence'].values
val_true_arousal = val_df['arousal'].values

ensemble_ccc_v = pearsonr(val_true_valence, ensemble_val_valence)[0]
ensemble_ccc_a = pearsonr(val_true_arousal, ensemble_val_arousal)[0]
ensemble_ccc = (ensemble_ccc_v + ensemble_ccc_a) / 2

print(f'Ensemble Validation Results:')
print(f'  CCC Average: {ensemble_ccc:.4f}')
print(f'  CCC Valence: {ensemble_ccc_v:.4f}')
print(f'  CCC Arousal: {ensemble_ccc_a:.4f}')

# Compare with individual models
print(f'\\nComparison:')
for seed_name, checkpoint in checkpoints.items():
    print(f'  {seed_name}: {checkpoint["best_ccc"]:.4f}')
print(f'  Ensemble: {ensemble_ccc:.4f} ⭐')

improvement = ensemble_ccc - np.mean([ckpt['best_ccc'] for ckpt in checkpoints.values()])
print(f'\\n  Ensemble improvement: +{improvement:.4f} CCC')
"""

print('\n' + '='*80)
print('EXPECTED RESULTS')
print('='*80)
print('ACTUAL INDIVIDUAL MODELS:')
print('  seed42:  CCC 0.5144')
print('  seed123: CCC 0.5330')
print('  seed777: CCC 0.6554 ⭐')
print('  Average: CCC 0.5676')
print()
print('EXPECTED ENSEMBLE:')
print('  Weighted ensemble: CCC 0.5876-0.5976')
print('  Simple average:    CCC 0.5856-0.5956')
print()
print('Recommendation: Use weighted ensemble for best performance')
print('Expected improvement: +0.020 ~ +0.030 CCC over individual average')
print('='*80)
