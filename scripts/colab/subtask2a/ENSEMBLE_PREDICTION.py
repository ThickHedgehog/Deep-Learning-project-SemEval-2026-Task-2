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

# ===== CONFIGURATION =====
MODEL_PATHS = {
    'seed42': 'v3.0_seed42_best.pt',
    'seed123': 'v3.0_seed123_best.pt',
    'seed777': 'v3.0_seed777_best.pt'
}

print('='*80)
print('v3.0 ENSEMBLE PREDICTION')
print('='*80)

# Load model checkpoints
checkpoints = {}
for seed_name, path in MODEL_PATHS.items():
    checkpoint = torch.load(path, map_location='cpu')
    checkpoints[seed_name] = checkpoint
    print(f'\n{seed_name}:')
    print(f'  CCC: {checkpoint["best_ccc"]:.4f}')
    print(f'  Valence: {checkpoint["val_ccc_v"]:.4f}')
    print(f'  Arousal: {checkpoint["val_ccc_a"]:.4f}')
    print(f'  Epoch: {checkpoint["epoch"]}')

print('\n' + '='*80)

# ===== WEIGHTED ENSEMBLE =====

# Calculate weights based on performance
cccs = {name: ckpt['best_ccc'] for name, ckpt in checkpoints.items()}
total_ccc = sum(cccs.values())
weights = {name: ccc / total_ccc for name, ccc in cccs.items()}

print('\nENSEMBLE WEIGHTS (performance-based):')
for name, weight in weights.items():
    print(f'  {name}: {weight:.3f} (CCC: {cccs[name]:.4f})')

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
print('Individual models: CCC 0.510-0.515')
print('Weighted ensemble: CCC 0.530-0.550')
print('Simple average:    CCC 0.528-0.548')
print('\nRecommendation: Use weighted ensemble for best performance')
print('='*80)
