"""
Data preprocessing script for SemEval-2026 Task 2 - Subtask 1
This script processes the raw subtask1 data and creates a cleaned, feature-enriched dataset
ready for model training.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path to import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import TextPreprocessor, TemporalPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw subtask1 data from CSV file.
    
    Args:
        file_path: Path to the raw CSV file
        
    Returns:
        pandas DataFrame with raw data
    """
    logger.info(f"Loading raw data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records from raw data")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Unique users: {df['user_id'].nunique()}")
    
    return df


def analyze_data_distribution(df: pd.DataFrame) -> None:
    """
    Analyze and log data distribution statistics.
    
    Args:
        df: Input DataFrame
    """
    logger.info("=== DATA DISTRIBUTION ANALYSIS ===")
    
    # Basic statistics
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique users: {df['user_id'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Text type distribution
    if 'is_words' in df.columns:
        text_types = df['is_words'].value_counts()
        logger.info(f"Text types - Full text: {text_types.get(False, 0)}, Word lists: {text_types.get(True, 0)}")
    
    # Collection phase distribution
    if 'collection_phase' in df.columns:
        phase_dist = df['collection_phase'].value_counts().sort_index()
        logger.info(f"Collection phases: {dict(phase_dist)}")
    
    # Emotion distribution
    logger.info(f"Valence range: {df['valence'].min()} to {df['valence'].max()}")
    logger.info(f"Arousal range: {df['arousal'].min()} to {df['arousal'].max()}")
    logger.info(f"Valence mean: {df['valence'].mean():.3f}, std: {df['valence'].std():.3f}")
    logger.info(f"Arousal mean: {df['arousal'].mean():.3f}, std: {df['arousal'].std():.3f}")
    
    # Records per user statistics
    user_counts = df['user_id'].value_counts()
    logger.info(f"Records per user - Min: {user_counts.min()}, Max: {user_counts.max()}, Mean: {user_counts.mean():.1f}")


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the input data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("=== DATA CLEANING AND VALIDATION ===")
    initial_count = len(df)
    
    # Remove rows with missing essential data
    essential_columns = ['user_id', 'text', 'timestamp', 'valence', 'arousal']
    df = df.dropna(subset=essential_columns)
    logger.info(f"Removed {initial_count - len(df)} rows with missing essential data")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure emotion values are numeric
    df['valence'] = pd.to_numeric(df['valence'], errors='coerce')
    df['arousal'] = pd.to_numeric(df['arousal'], errors='coerce')
    
    # Remove rows with invalid emotion values
    before_emotion_filter = len(df)
    df = df.dropna(subset=['valence', 'arousal'])
    logger.info(f"Removed {before_emotion_filter - len(df)} rows with invalid emotion values")
    
    # Clean text data - remove empty or very short texts
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.len() >= 5]  # At least 5 characters
    logger.info(f"Removed rows with very short texts. Final count: {len(df)}")
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    return df


def preprocess_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with preprocessed text and features
    """
    logger.info("=== TEXT PREPROCESSING ===")
    
    # Initialize text preprocessor with moderate cleaning
    # We keep punctuation and case to preserve emotional expression
    text_preprocessor = TextPreprocessor(
        lowercase=False,  # Keep original case for emotion analysis
        remove_punctuation=False,  # Keep punctuation for emotional cues
        remove_stopwords=False,  # Keep stopwords for context
        lemmatize=False,  # Keep original word forms
        remove_html=True,  # Remove any HTML tags
        remove_urls=True,  # Remove URLs
        normalize_whitespace=True,  # Clean up spacing
        min_length=5,  # Minimum text length
        max_length=5000  # Maximum text length
    )
    
    # Apply preprocessing
    df_processed = text_preprocessor.preprocess_dataframe(df, text_column='text')
    
    logger.info(f"Text preprocessing completed. Added features: {[col for col in df_processed.columns if col.startswith('feature_')]}")
    
    return df_processed


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with temporal features
    """
    logger.info("=== ADDING TEMPORAL FEATURES ===")
    
    # Initialize temporal preprocessor
    temporal_preprocessor = TemporalPreprocessor(
        min_sequences_per_user=3,  # Minimum 3 records per user
        normalize_emotions=False  # Keep original emotion scale
    )
    
    # Process temporal data
    df_temporal = temporal_preprocessor.process_temporal_data(
        df,
        text_column='text',
        valence_column='valence',
        arousal_column='arousal',
        user_id_column='user_id',
        timestamp_column='timestamp'
    )
    
    temporal_features = [col for col in df_temporal.columns if col not in df.columns]
    logger.info(f"Added temporal features: {temporal_features}")
    
    return df_temporal


def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add metadata-based features to the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with metadata features
    """
    logger.info("=== ADDING METADATA FEATURES ===")
    
    # Text type feature (if available)
    if 'is_words' in df.columns:
        df['is_word_list'] = df['is_words'].astype(int)
    
    # Collection phase feature (if available)
    if 'collection_phase' in df.columns:
        df['collection_phase_encoded'] = df['collection_phase'].astype(int)
    
    # User-specific features
    user_stats = df.groupby('user_id').agg({
        'valence': ['count', 'mean', 'std'],
        'arousal': ['mean', 'std'],
        'timestamp': ['min', 'max']
    }).round(4)
    
    # Flatten column names
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    user_stats = user_stats.add_prefix('user_')
    
    # Merge user statistics back to main dataframe
    df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    
    # Calculate user experience (days since first entry)
    df['user_days_active'] = (df['timestamp'] - df['user_timestamp_min']).dt.days
    
    logger.info(f"Added metadata features for {df['user_id'].nunique()} users")
    
    return df


def create_final_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Create and save the final processed dataset.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the processed data
    """
    logger.info("=== CREATING FINAL DATASET ===")
    
    # Select and order columns for the final dataset
    # Core columns
    core_columns = [
        'user_id', 'text_id', 'text', 'text_cleaned', 'timestamp',
        'valence', 'arousal'
    ]
    
    # Metadata columns
    metadata_columns = [col for col in df.columns if col in [
        'collection_phase', 'is_words', 'is_word_list', 'collection_phase_encoded'
    ]]
    
    # Text features
    text_feature_columns = [col for col in df.columns if col.startswith('feature_')]
    
    # Temporal features  
    temporal_columns = [col for col in df.columns if col in [
        'hour', 'day_of_week', 'day_of_month', 'month',
        'time_since_start', 'valence_rolling_mean', 'arousal_rolling_mean',
        'valence_rolling_std', 'arousal_rolling_std', 'valence_trend', 'arousal_trend'
    ]]
    
    # User features
    user_feature_columns = [col for col in df.columns if col.startswith('user_') and col != 'user_id']
    
    # Combine all columns
    final_columns = (core_columns + metadata_columns + text_feature_columns + 
                    temporal_columns + user_feature_columns)
    
    # Filter to existing columns
    final_columns = [col for col in final_columns if col in df.columns]
    
    # Create final dataset
    df_final = df[final_columns].copy()
    
    # Remove any remaining NaN values in critical columns
    df_final = df_final.dropna(subset=['valence', 'arousal', 'text_cleaned'])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df_final.to_csv(output_path, index=False)
    
    logger.info(f"Final dataset saved to: {output_path}")
    logger.info(f"Final dataset shape: {df_final.shape}")
    logger.info(f"Final columns: {list(df_final.columns)}")
    
    # Log final statistics
    logger.info("=== FINAL DATASET STATISTICS ===")
    logger.info(f"Total records: {len(df_final)}")
    logger.info(f"Unique users: {df_final['user_id'].nunique()}")
    logger.info(f"Feature columns: {len([col for col in df_final.columns if col.startswith('feature_')])}")
    logger.info(f"Temporal features: {len([col for col in df_final.columns if col in temporal_columns])}")
    logger.info(f"User features: {len([col for col in df_final.columns if col.startswith('user_')])}")


def main():
    """Main function to run the data preprocessing pipeline."""
    
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    raw_data_path = base_path / "data" / "raw" / "TRAIN_RELEASE_3SEP2025" / "train_subtask1.csv"
    processed_data_path = base_path / "data" / "processed" / "subtask1_processed.csv"
    
    logger.info("=== STARTING SUBTASK1 DATA PREPROCESSING ===")
    logger.info(f"Input file: {raw_data_path}")
    logger.info(f"Output file: {processed_data_path}")
    
    try:
        # Step 1: Load raw data
        df = load_raw_data(str(raw_data_path))
        
        # Step 2: Analyze data distribution
        analyze_data_distribution(df)
        
        # Step 3: Clean and validate data
        df = clean_and_validate_data(df)
        
        # Step 4: Preprocess text data
        df = preprocess_text_data(df)
        
        # Step 5: Add temporal features
        df = add_temporal_features(df)
        
        # Step 6: Add metadata features
        df = add_metadata_features(df)
        
        # Step 7: Create final dataset
        create_final_dataset(df, str(processed_data_path))
        
        logger.info("=== DATA PREPROCESSING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()