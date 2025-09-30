"""
Example usage of prepared Subtask 1 data

This script demonstrates how to load and use the processed data
for Subtask 1: Longitudinal Affect Assessment.

Usage:
    python example_usage.py
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def load_processed_data():
    """Load the processed data files."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data/processed"
    
    # Load CSV files
    train_df = pd.read_csv(data_dir / "subtask1_train.csv")
    val_df = pd.read_csv(data_dir / "subtask1_val.csv") 
    test_df = pd.read_csv(data_dir / "subtask1_test.csv")
    
    # Load sequences
    with open(data_dir / "subtask1_sequences.json", 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    
    # Load metadata
    with open(data_dir / "subtask1_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return {
        "train": train_df,
        "val": val_df, 
        "test": test_df,
        "sequences": sequences,
        "metadata": metadata
    }


def analyze_temporal_patterns(data):
    """Analyze temporal patterns in the data."""
    print("=== TEMPORAL PATTERN ANALYSIS ===")
    
    # Look at a few sequences
    sequences = data["sequences"]
    print(f"Total sequences: {len(sequences)}")
    
    # Analyze sequence from a user with many entries
    user_seq_counts = {}
    for seq in sequences:
        user_id = seq['user_id']
        user_seq_counts[user_id] = user_seq_counts.get(user_id, 0) + 1
    
    # Get user with most sequences
    max_user = max(user_seq_counts.keys(), key=lambda x: user_seq_counts[x])
    user_sequences = [seq for seq in sequences if seq['user_id'] == max_user]
    
    print(f"\\nUser {max_user} has {len(user_sequences)} sequences")
    print("\\nFirst sequence from this user:")
    first_seq = user_sequences[0]
    
    for i, (text, valence, arousal) in enumerate(zip(
        first_seq['texts'], 
        first_seq['valences'], 
        first_seq['arousals']
    )):
        print(f"  Step {i+1}:")
        print(f"    Text: {text[:100]}..." if len(text) > 100 else f"    Text: {text}")
        print(f"    Valence: {valence}, Arousal: {arousal}")
        print()


def emotion_analysis(data):
    """Analyze emotion patterns."""
    print("=== EMOTION PATTERN ANALYSIS ===")
    
    train_df = data["train"]
    
    # Basic emotion statistics
    print("Valence distribution:")
    print(f"  Min: {train_df['valence'].min()}, Max: {train_df['valence'].max()}")
    print(f"  Mean: {train_df['valence'].mean():.2f}, Std: {train_df['valence'].std():.2f}")
    print(f"  Quartiles: {train_df['valence'].quantile([0.25, 0.5, 0.75]).values}")
    
    print("\\nArousal distribution:")
    print(f"  Min: {train_df['arousal'].min()}, Max: {train_df['arousal'].max()}")
    print(f"  Mean: {train_df['arousal'].mean():.2f}, Std: {train_df['arousal'].std():.2f}")
    print(f"  Quartiles: {train_df['arousal'].quantile([0.25, 0.5, 0.75]).values}")
    
    # Correlation analysis
    correlation = train_df['valence'].corr(train_df['arousal'])
    print(f"\\nValence-Arousal correlation: {correlation:.3f}")
    
    # Text type differences
    essay_mask = ~train_df['is_words']
    words_mask = train_df['is_words']
    
    print("\\nEmotion by text type:")
    print(f"  Essays - Valence: {train_df[essay_mask]['valence'].mean():.2f}, "
          f"Arousal: {train_df[essay_mask]['arousal'].mean():.2f}")
    print(f"  Words - Valence: {train_df[words_mask]['valence'].mean():.2f}, "
          f"Arousal: {train_df[words_mask]['arousal'].mean():.2f}")


def sequence_features_analysis(data):
    """Analyze sequence-level features."""
    print("=== SEQUENCE FEATURES ANALYSIS ===")
    
    sequences = data["sequences"]
    
    # Analyze emotion trajectories
    emotion_changes = []
    for seq in sequences:
        valences = seq['valences']
        arousals = seq['arousals']
        
        # Calculate changes within sequence
        val_change = valences[-1] - valences[0]  # First to last
        aro_change = arousals[-1] - arousals[0]
        
        emotion_changes.append({
            'valence_change': val_change,
            'arousal_change': aro_change,
            'valence_std': np.std(valences),
            'arousal_std': np.std(arousals)
        })
    
    changes_df = pd.DataFrame(emotion_changes)
    
    print("Emotion changes across sequences:")
    print(f"  Valence change - Mean: {changes_df['valence_change'].mean():.3f}, "
          f"Std: {changes_df['valence_change'].std():.3f}")
    print(f"  Arousal change - Mean: {changes_df['arousal_change'].mean():.3f}, "
          f"Std: {changes_df['arousal_change'].std():.3f}")
    
    print("\\nEmotion variability within sequences:")
    print(f"  Valence std - Mean: {changes_df['valence_std'].mean():.3f}")
    print(f"  Arousal std - Mean: {changes_df['arousal_std'].mean():.3f}")


def text_analysis(data):
    """Analyze text characteristics."""
    print("=== TEXT ANALYSIS ===")
    
    train_df = data["train"]
    
    # Text length analysis
    print("Text characteristics:")
    print(f"  Length - Mean: {train_df['text_length'].mean():.0f}, "
          f"Std: {train_df['text_length'].std():.0f}")
    print(f"  Word count - Mean: {train_df['word_count'].mean():.0f}, "
          f"Std: {train_df['word_count'].std():.0f}")
    
    # Analyze relationship between text features and emotions
    length_val_corr = train_df['text_length'].corr(train_df['valence'])
    length_aro_corr = train_df['text_length'].corr(train_df['arousal'])
    
    print("\\nText-emotion correlations:")
    print(f"  Text length vs Valence: {length_val_corr:.3f}")
    print(f"  Text length vs Arousal: {length_aro_corr:.3f}")
    
    # Show some example texts
    print("\\nExample texts:")
    print("\\nHigh valence (positive emotion):")
    high_val = train_df.nlargest(3, 'valence')
    for _, row in high_val.iterrows():
        text = row['text_cleaned'][:150] + "..." if len(row['text_cleaned']) > 150 else row['text_cleaned']
        print(f"  V={row['valence']}, A={row['arousal']}: {text}")
    
    print("\\nLow valence (negative emotion):")
    low_val = train_df.nsmallest(3, 'valence')
    for _, row in low_val.iterrows():
        text = row['text_cleaned'][:150] + "..." if len(row['text_cleaned']) > 150 else row['text_cleaned']
        print(f"  V={row['valence']}, A={row['arousal']}: {text}")


def modeling_recommendations():
    """Provide modeling recommendations based on analysis."""
    print("=== MODELING RECOMMENDATIONS ===")
    
    recommendations = [
        "1. TEXT ENCODING:",
        "   • Use pre-trained language models (BERT, RoBERTa) for text representation",
        "   • Consider separate encoding for essays vs. word lists",
        "   • Apply text normalization but preserve emotional cues",
        "",
        "2. TEMPORAL MODELING:",
        "   • Implement LSTM/GRU layers to capture temporal dependencies",
        "   • Use attention mechanisms to focus on important time steps",
        "   • Consider bidirectional processing for better context",
        "",
        "3. USER MODELING:",
        "   • Add user embeddings for personalization",
        "   • Consider user-specific emotion baselines",
        "   • Handle varying sequence lengths per user",
        "",
        "4. MULTI-TASK LEARNING:",
        "   • Joint prediction of valence and arousal",
        "   • Consider auxiliary tasks (text type classification)",
        "   • Use loss functions appropriate for regression",
        "",
        "5. DATA HANDLING:",
        "   • Use temporal splits to avoid data leakage",
        "   • Handle both essay and word-list inputs appropriately",
        "   • Consider data augmentation for sequence diversity",
        "",
        "6. EVALUATION:",
        "   • Use correlation metrics (Pearson, Spearman)",
        "   • Consider user-level evaluation metrics",
        "   • Evaluate on both seen and unseen users"
    ]
    
    for rec in recommendations:
        print(rec)


def main():
    """Run the complete example analysis."""
    print("SUBTASK 1: LONGITUDINAL AFFECT ASSESSMENT - DATA USAGE EXAMPLE")
    print("=" * 70)
    
    try:
        # Load processed data
        print("Loading processed data...")
        data = load_processed_data()
        print(f"✓ Loaded data: {data['metadata']['data_splits']['total_sequences']} sequences")
        
        # Run analyses
        analyze_temporal_patterns(data)
        emotion_analysis(data)
        sequence_features_analysis(data)
        text_analysis(data)
        modeling_recommendations()
        
        print("\\n" + "=" * 70)
        print("DATA ANALYSIS COMPLETE!")
        print("The processed data is ready for model training.")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"Error: Processed data files not found: {e}")
        print("Please run 'python simple_data_prep.py' first.")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()