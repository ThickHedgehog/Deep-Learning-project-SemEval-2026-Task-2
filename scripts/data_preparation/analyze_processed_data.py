"""
Data analysis script for processed subtask1 data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_processed_data():
    """Analyze the processed subtask1 data."""
    
    # Load processed data
    data_path = Path("data/processed/subtask1_processed.csv")
    df = pd.read_csv(data_path)
    
    logger.info("=== PROCESSED DATA ANALYSIS ===")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique users: {df['user_id'].nunique()}")
    
    # Basic statistics
    print("\n=== EMOTION STATISTICS ===")
    print("Valence statistics:")
    print(df['valence'].describe())
    print("\nArousal statistics:")
    print(df['arousal'].describe())
    
    # Text features statistics
    print("\n=== TEXT FEATURES STATISTICS ===")
    text_features = [col for col in df.columns if col.startswith('feature_')]
    print(f"Number of text features: {len(text_features)}")
    
    for feature in text_features:
        print(f"{feature}: mean={df[feature].mean():.2f}, std={df[feature].std():.2f}")
    
    # User statistics
    print("\n=== USER STATISTICS ===")
    user_counts = df['user_id'].value_counts()
    print(f"Records per user - Min: {user_counts.min()}, Max: {user_counts.max()}, Mean: {user_counts.mean():.1f}")
    
    # Collection phase distribution
    print("\n=== COLLECTION PHASE DISTRIBUTION ===")
    phase_dist = df['collection_phase'].value_counts().sort_index()
    for phase, count in phase_dist.items():
        print(f"Phase {phase}: {count} records ({count/len(df)*100:.1f}%)")
    
    # Text type distribution
    print("\n=== TEXT TYPE DISTRIBUTION ===")
    text_type_dist = df['is_words'].value_counts()
    print(f"Full text (False): {text_type_dist.get(False, 0)} records")
    print(f"Word lists (True): {text_type_dist.get(True, 0)} records")
    
    # Temporal distribution
    print("\n=== TEMPORAL DISTRIBUTION ===")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
    
    # Save summary statistics
    summary_stats = {
        'total_records': len(df),
        'unique_users': df['user_id'].nunique(),
        'valence_mean': df['valence'].mean(),
        'valence_std': df['valence'].std(),
        'arousal_mean': df['arousal'].mean(),
        'arousal_std': df['arousal'].std(),
        'avg_text_length': df['feature_length'].mean(),
        'avg_word_count': df['feature_word_count'].mean(),
        'text_features_count': len(text_features)
    }
    
    print("\n=== SUMMARY STATISTICS ===")
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return df, summary_stats

def create_sample_dataset(df, sample_size=100):
    """Create a small sample dataset for testing."""
    
    logger.info(f"Creating sample dataset with {sample_size} records")
    
    # Sample stratified by user to maintain user diversity
    sample_df = df.groupby('user_id').apply(
        lambda x: x.sample(min(len(x), max(1, sample_size // df['user_id'].nunique())))
    ).reset_index(drop=True)
    
    # If we need more records, sample additional ones
    if len(sample_df) < sample_size:
        remaining = sample_size - len(sample_df)
        additional = df[~df.index.isin(sample_df.index)].sample(min(remaining, len(df) - len(sample_df)))
        sample_df = pd.concat([sample_df, additional]).reset_index(drop=True)
    
    # Save sample dataset
    sample_path = Path("data/processed/subtask1_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    
    logger.info(f"Sample dataset saved: {sample_path}")
    logger.info(f"Sample shape: {sample_df.shape}")
    logger.info(f"Sample users: {sample_df['user_id'].nunique()}")
    
    return sample_df

if __name__ == "__main__":
    # Analyze processed data
    df, stats = analyze_processed_data()
    
    # Create sample dataset
    sample_df = create_sample_dataset(df, sample_size=200)
    
    print("\n=== ANALYSIS COMPLETED ===")
    print("Files created:")
    print("- data/processed/subtask1_processed.csv (full dataset)")
    print("- data/processed/subtask1_sample.csv (sample dataset)")