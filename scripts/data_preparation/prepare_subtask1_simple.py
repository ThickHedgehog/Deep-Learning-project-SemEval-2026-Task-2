"""
Simplified data preprocessing script for SemEval-2026 Task 2 - Subtask 1
This script processes the raw subtask1 data with basic preprocessing
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import re

# Add src to path to import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preprocessing_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def basic_text_cleaning(text):
    """Basic text cleaning without external libraries."""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text


def extract_basic_features(text):
    """Extract basic text features without external libraries."""
    features = {}
    
    # Basic features
    features['length'] = len(text)
    words = text.split()
    features['word_count'] = len(words)
    features['sentence_count'] = max(1, text.count('.') + text.count('!') + text.count('?'))
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['period_count'] = text.count('.')
    features['comma_count'] = text.count(',')
    
    # Case features
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['capitalized_words'] = sum(1 for word in words if word and word[0].isupper())
    
    return features


def load_and_process_data(file_path):
    """Load and process the data with basic preprocessing."""
    logger.info(f"Loading data from: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Basic data cleaning
    logger.info("Starting data cleaning...")
    
    # Remove rows with missing essential data
    essential_columns = ['user_id', 'text', 'timestamp', 'valence', 'arousal']
    initial_count = len(df)
    df = df.dropna(subset=essential_columns)
    logger.info(f"Removed {initial_count - len(df)} rows with missing data")
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Clean text
    logger.info("Cleaning text...")
    df['text_cleaned'] = df['text'].apply(basic_text_cleaning)
    
    # Remove very short texts
    df = df[df['text_cleaned'].str.len() >= 5]
    logger.info(f"After text cleaning: {len(df)} records")
    
    # Extract basic text features
    logger.info("Extracting text features...")
    feature_dicts = df['text_cleaned'].apply(extract_basic_features)
    feature_df = pd.DataFrame(feature_dicts.tolist())
    
    # Add feature columns
    for col in feature_df.columns:
        df[f'feature_{col}'] = feature_df[col]
    
    logger.info(f"Extracted {len(feature_df.columns)} text features")
    
    # Add temporal features
    logger.info("Adding temporal features...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    
    # Add user-level features
    logger.info("Adding user features...")
    user_stats = df.groupby('user_id').agg({
        'valence': ['count', 'mean', 'std'],
        'arousal': ['mean', 'std'],
        'timestamp': ['min', 'max']
    }).round(4)
    
    # Flatten column names
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    user_stats = user_stats.add_prefix('user_')
    
    # Fill NaN std values for users with only one record
    user_stats = user_stats.fillna(0)
    
    # Merge back to main dataframe
    df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    
    # Calculate days active
    df['user_days_active'] = (df['timestamp'] - df['user_timestamp_min']).dt.days
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    logger.info("Data processing completed")
    return df


def save_processed_data(df, output_path):
    """Save the processed data to CSV."""
    logger.info(f"Saving processed data to: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed dataset with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Log final statistics
    logger.info("=== FINAL STATISTICS ===")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique users: {df['user_id'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Text features: {len([col for col in df.columns if col.startswith('feature_')])}")
    logger.info(f"User features: {len([col for col in df.columns if col.startswith('user_')])}")
    
    # Sample of the data
    logger.info("=== SAMPLE DATA ===")
    logger.info(f"First 3 rows:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        logger.info(f"User {row['user_id']}: valence={row['valence']}, arousal={row['arousal']}, "
                   f"text_length={row['feature_length']}, words={row['feature_word_count']}")


def main():
    """Main function."""
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    raw_data_path = base_path / "data" / "raw" / "TRAIN_RELEASE_3SEP2025" / "train_subtask1.csv"
    processed_data_path = base_path / "data" / "processed" / "subtask1_processed.csv"
    
    logger.info("=== STARTING SUBTASK1 DATA PREPROCESSING (SIMPLIFIED) ===")
    
    try:
        # Process data
        df = load_and_process_data(str(raw_data_path))
        
        # Save processed data
        save_processed_data(df, str(processed_data_path))
        
        logger.info("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()