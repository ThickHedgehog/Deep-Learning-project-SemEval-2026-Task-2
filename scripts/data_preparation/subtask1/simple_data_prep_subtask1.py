"""
Simple Subtask 1 Data Preparation Script
Prepares dataset for Subtask 1: Longitudinal Affect Assessment
"""

import pandas as pd
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_text(text):
    """Enhanced text cleaning with apostrophe normalization."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Fix apostrophes with spaces (I ' ve -> I've)
    text = re.sub(r"\s'\s", "'", text)
    text = re.sub(r"\s'([a-z])", r"'\1", text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove excessive dots
    text = re.sub(r'\.{3,}', '...', text)
    
    return text


def extract_text_features(text):
    """Extract detailed text features."""
    if not text:
        return {
            'text_length': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'exclamation_count': 0, 'question_count': 0,
            'uppercase_ratio': 0, 'capitalized_words': 0
        }
    
    words = text.split()
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'sentence_count': max(1, text.count('.') + text.count('!') + text.count('?')),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'capitalized_words': sum(1 for w in words if w and w[0].isupper())
    }


def prepare_data(data_path, output_path):
    logger.info("=== STARTING DATA PREPARATION ===")
    
    # Load
    logger.info(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records from {df['user_id'].nunique()} users")
    
    # Prepare
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Clean text
    logger.info("Cleaning text...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text_cleaned'].str.len() >= 10]  # At least 10 chars
    logger.info(f"Removed {initial_count - len(df)} too short texts")
    
    # Extract detailed text features
    logger.info("Extracting text features...")
    text_features = df['text_cleaned'].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # Add text features to dataframe
    for col in text_features_df.columns:
        df[col] = text_features_df[col].values
    
    # Add temporal features
    logger.info("Adding temporal features...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # User-level features
    logger.info("Adding user-level features...")
    user_stats = df.groupby('user_id').agg({
        'text_id': 'count',
        'valence': ['mean', 'std'],
        'arousal': ['mean', 'std'],
        'timestamp': ['min', 'max']
    }).round(4)
    
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    user_stats = user_stats.add_prefix('user_')
    user_stats = user_stats.fillna(0)  # Fill NaN std for single-entry users
    
    # Merge user stats
    df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    
    # User temporal features
    logger.info("Adding user temporal features...")
    for uid in df['user_id'].unique():
        mask = df['user_id'] == uid
        udata = df[mask].copy()
        
        # Entry number
        df.loc[mask, 'entry_number'] = list(range(len(udata)))
        
        # Hours since start
        first_ts = udata['timestamp'].min()
        df.loc[mask, 'hours_since_start'] = (udata['timestamp'] - first_ts).dt.total_seconds() / 3600
        
        # Days active
        last_ts = udata['timestamp'].max()
        days_active = (last_ts - first_ts).days + 1
        df.loc[mask, 'user_days_active'] = days_active
        
        # Entry frequency (entries per day)
        df.loc[mask, 'user_entry_frequency'] = len(udata) / max(1, days_active)
        
        # Time gap from previous entry
        time_gaps = udata['timestamp'].diff().dt.total_seconds() / 3600
        df.loc[mask, 'time_gap_hours'] = time_gaps.fillna(0)
        
        # Rolling stats (window of 3)
        if len(udata) >= 3:
            vr = udata['valence'].rolling(window=3, min_periods=1)
            ar = udata['arousal'].rolling(window=3, min_periods=1)
            df.loc[mask, 'valence_rolling_mean'] = vr.mean()
            df.loc[mask, 'arousal_rolling_mean'] = ar.mean()
            df.loc[mask, 'valence_rolling_std'] = vr.std().fillna(0)
            df.loc[mask, 'arousal_rolling_std'] = ar.std().fillna(0)
            df.loc[mask, 'valence_diff'] = udata['valence'].diff().fillna(0)
            df.loc[mask, 'arousal_diff'] = udata['arousal'].diff().fillna(0)
        else:
            df.loc[mask, 'valence_rolling_mean'] = udata['valence']
            df.loc[mask, 'arousal_rolling_mean'] = udata['arousal']
            df.loc[mask, 'valence_rolling_std'] = 0
            df.loc[mask, 'arousal_rolling_std'] = 0
            df.loc[mask, 'valence_diff'] = 0
            df.loc[mask, 'arousal_diff'] = 0
    
    # Encode categorical features
    logger.info("Encoding categorical features...")
    df['is_words_encoded'] = df['is_words'].astype(int)
    df['collection_phase_encoded'] = df['collection_phase']
    
    # Save
    logger.info(f"Saving: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Statistics
    logger.info("=== PROCESSING COMPLETE ===")
    logger.info(f"Final dataset: {len(df)} records from {df['user_id'].nunique()} users")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Feature columns added: {len(df.columns) - 8}")
    logger.info("Text features: text_length, word_count, sentence_count, avg_word_length, etc.")
    logger.info("Temporal features: hour, day_of_week, month, is_weekend, hours_since_start, time_gap_hours")
    logger.info("User features: user stats, entry_number, user_days_active, user_entry_frequency")
    logger.info("Rolling features: valence/arousal rolling mean/std, diff")
    
    return df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "train_subtask1.csv"
    output_path = project_root / "data" / "processed" / "subtask1_processed.csv"
    
    try:
        df = prepare_data(data_path, output_path)
        print(f"\n{'='*60}\nSUBTASK 1 DATA PREPARATION COMPLETED!\n{'='*60}")
        print(f"Output: {output_path}\nRecords: {len(df)}, Users: {df['user_id'].nunique()}")
        print(f"Columns: {list(df.columns)}\n\nReady for training!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
