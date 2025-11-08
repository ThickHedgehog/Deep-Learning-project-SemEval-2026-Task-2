"""
Feature Preparation for Subtask 2a
===================================
Prepares features for temporal emotion prediction from text.
Based on comprehensive data analysis findings.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_text(text):
    """Enhanced text cleaning while preserving emotion indicators."""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Fix apostrophes with spaces (I ' ve -> I've)
    text = re.sub(r"\s'\s", "'", text)
    text = re.sub(r"\s'([a-z])", r"'\1", text)

    # Normalize whitespace (but keep punctuation for emotion)
    text = re.sub(r'\s+', ' ', text).strip()

    # Keep excessive dots as they might indicate emotion
    # text = re.sub(r'\.{3,}', '...', text)  # Commented out to preserve

    return text


def extract_text_features(text):
    """Extract comprehensive text features for emotion prediction."""
    if not text:
        return {
            'text_length': 0, 'word_count': 0, 'char_count': 0,
            'sentence_count': 0, 'avg_word_length': 0,
            'exclamation_count': 0, 'question_count': 0,
            'uppercase_ratio': 0, 'capitalized_words': 0,
            'first_person_count': 0, 'has_uppercase_words': 0
        }

    words = text.split()
    text_lower = text.lower()

    # Count first-person pronouns (indicator of personal narrative)
    first_person_words = ['i', 'me', 'my', 'mine', 'myself']
    first_person_count = sum(text_lower.count(f' {word} ') +
                             text_lower.startswith(word + ' ')
                             for word in first_person_words)

    # Check for fully uppercase words (emotion emphasis)
    has_uppercase_words = int(any(word.isupper() and len(word) > 1 for word in words))

    return {
        'text_length': len(text),
        'char_count': len(text.replace(' ', '')),
        'word_count': len(words),
        'sentence_count': max(1, text.count('.') + text.count('!') + text.count('?')),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'capitalized_words': sum(1 for w in words if w and w[0].isupper()),
        'first_person_count': first_person_count,
        'has_uppercase_words': has_uppercase_words
    }


def extract_emotion_word_features(text):
    """Extract emotion-specific word features."""
    if not text:
        return {
            'positive_word_count': 0, 'negative_word_count': 0,
            'high_arousal_word_count': 0, 'low_arousal_word_count': 0,
            'sentiment_score': 0
        }

    text_lower = text.lower()

    # Positive emotion words (from analysis)
    positive_words = ['happy', 'great', 'good', 'excellent', 'wonderful', 'amazing',
                     'best', 'better', 'love', 'joy', 'pleased', 'excited', 'glad',
                     'fantastic', 'perfect', 'awesome', 'nice', 'fine', 'well',
                     'calm', 'content', 'relaxed']

    # Negative emotion words (from analysis)
    negative_words = ['sad', 'bad', 'terrible', 'horrible', 'awful', 'worse', 'worst',
                     'hate', 'angry', 'upset', 'depressed', 'anxious', 'worried',
                     'stressed', 'miserable', 'unhappy', 'tired', 'exhausted',
                     'frustrated', 'annoyed', 'nervous']

    # High arousal words
    high_arousal_words = ['excited', 'anxious', 'active', 'nervous', 'energetic',
                         'stressed', 'angry', 'jittery', 'frantic', 'hyper']

    # Low arousal words
    low_arousal_words = ['tired', 'calm', 'relaxed', 'sleepy', 'peaceful',
                        'lazy', 'sluggish', 'quiet', 'drained', 'exhausted']

    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)
    high_arousal_count = sum(text_lower.count(word) for word in high_arousal_words)
    low_arousal_count = sum(text_lower.count(word) for word in low_arousal_words)

    return {
        'positive_word_count': positive_count,
        'negative_word_count': negative_count,
        'high_arousal_word_count': high_arousal_count,
        'low_arousal_word_count': low_arousal_count,
        'sentiment_score': positive_count - negative_count
    }


def detect_tense(text):
    """Detect predominant tense in text."""
    if not text:
        return 'unknown'

    text_lower = text.lower()

    # Past tense indicators
    past_patterns = ['was', 'were', 'had', 'did', 'went', 'felt', 'got', 'came', 'made', 'took']
    past_count = sum(text_lower.count(word) for word in past_patterns)

    # Present tense indicators
    present_patterns = ['am', 'is', 'are', 'feel', 'think', 'have', 'do', 'go', 'get', 'make']
    present_count = sum(text_lower.count(word) for word in present_patterns)

    # Future tense indicators
    future_patterns = ['will', 'going to', 'gonna', 'shall', 'would']
    future_count = sum(text_lower.count(pattern) for pattern in future_patterns)

    counts = {'past': past_count, 'present': present_count, 'future': future_count}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else 'unknown'


def add_temporal_features(df):
    """Add temporal features based on timestamps."""
    logger.info("Adding temporal features...")

    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical encoding for hour (important for temporal patterns)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclical encoding for day of week
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_user_features(df):
    """Add user-level features."""
    logger.info("Adding user-level features...")

    # User statistics
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

    return df


def add_sequence_features(df):
    """Add sequence-based features (previous emotions, position, etc.)."""
    logger.info("Adding sequence features...")

    for uid in df['user_id'].unique():
        mask = df['user_id'] == uid
        udata = df[mask].copy()

        # Sequence position features
        n_entries = len(udata)
        df.loc[mask, 'entry_number'] = list(range(n_entries))
        df.loc[mask, 'relative_position'] = np.linspace(0, 1, n_entries)

        # Time-based features
        first_ts = udata['timestamp'].min()
        df.loc[mask, 'hours_since_start'] = (udata['timestamp'] - first_ts).dt.total_seconds() / 3600

        # Time gap from previous entry
        time_gaps = udata['timestamp'].diff().dt.total_seconds() / 3600
        df.loc[mask, 'time_gap_hours'] = time_gaps.fillna(0)

        # Log-scaled time gap (better for modeling)
        df.loc[mask, 'time_gap_log'] = np.log1p(time_gaps.fillna(0))

        # Previous emotions (lag features) - Critical for temporal prediction
        df.loc[mask, 'valence_lag1'] = udata['valence'].shift(1).fillna(udata['valence'].mean())
        df.loc[mask, 'arousal_lag1'] = udata['arousal'].shift(1).fillna(udata['arousal'].mean())
        df.loc[mask, 'valence_lag2'] = udata['valence'].shift(2).fillna(udata['valence'].mean())
        df.loc[mask, 'arousal_lag2'] = udata['arousal'].shift(2).fillna(udata['arousal'].mean())
        df.loc[mask, 'valence_lag3'] = udata['valence'].shift(3).fillna(udata['valence'].mean())
        df.loc[mask, 'arousal_lag3'] = udata['arousal'].shift(3).fillna(udata['arousal'].mean())

        # Rolling statistics (moving average)
        if len(udata) >= 3:
            vr = udata['valence'].rolling(window=3, min_periods=1)
            ar = udata['arousal'].rolling(window=3, min_periods=1)
            df.loc[mask, 'valence_rolling_mean'] = vr.mean()
            df.loc[mask, 'arousal_rolling_mean'] = ar.mean()
            df.loc[mask, 'valence_rolling_std'] = vr.std().fillna(0)
            df.loc[mask, 'arousal_rolling_std'] = ar.std().fillna(0)
        else:
            df.loc[mask, 'valence_rolling_mean'] = udata['valence']
            df.loc[mask, 'arousal_rolling_mean'] = udata['arousal']
            df.loc[mask, 'valence_rolling_std'] = 0
            df.loc[mask, 'arousal_rolling_std'] = 0

        # Emotion change velocity (first derivative)
        valence_diff = udata['valence'].diff().fillna(0)
        arousal_diff = udata['arousal'].diff().fillna(0)
        df.loc[mask, 'valence_velocity'] = valence_diff
        df.loc[mask, 'arousal_velocity'] = arousal_diff

    return df


def prepare_subtask2a_features(data_path, output_path):
    """Main feature preparation pipeline for Subtask 2a."""
    logger.info("="*80)
    logger.info("SUBTASK 2A FEATURE PREPARATION")
    logger.info("="*80)

    # Load data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records from {df['user_id'].nunique()} users")

    # Sort by user and timestamp (critical for temporal features)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # Clean text
    logger.info("Cleaning text...")
    df['text_cleaned'] = df['text'].apply(clean_text)

    # Remove very short texts
    df = df[df['text_cleaned'].str.len() >= 5]
    logger.info(f"Removed {initial_count - len(df)} very short texts")

    # Extract text features
    logger.info("Extracting text features...")
    text_features = df['text_cleaned'].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist())
    for col in text_features_df.columns:
        df[col] = text_features_df[col].values

    # Extract emotion word features
    logger.info("Extracting emotion word features...")
    emotion_features = df['text_cleaned'].apply(extract_emotion_word_features)
    emotion_features_df = pd.DataFrame(emotion_features.tolist())
    for col in emotion_features_df.columns:
        df[col] = emotion_features_df[col].values

    # Detect tense
    logger.info("Detecting tense...")
    df['text_tense'] = df['text_cleaned'].apply(detect_tense)
    df['tense_past'] = (df['text_tense'] == 'past').astype(int)
    df['tense_present'] = (df['text_tense'] == 'present').astype(int)
    df['tense_future'] = (df['text_tense'] == 'future').astype(int)

    # Add temporal features
    df = add_temporal_features(df)

    # Add user features
    df = add_user_features(df)

    # Add sequence features (including previous emotions)
    df = add_sequence_features(df)

    # Encode categorical features
    logger.info("Encoding categorical features...")
    df['is_words_encoded'] = df['is_words'].astype(int)
    df['collection_phase_encoded'] = df['collection_phase']

    # Save processed data
    logger.info(f"Saving processed data to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print statistics
    logger.info("="*80)
    logger.info("FEATURE PREPARATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Final dataset: {len(df)} records from {df['user_id'].nunique()} users")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Original columns: 10")
    logger.info(f"Feature columns added: {len(df.columns) - 10}")

    logger.info("\nFeature categories:")
    logger.info("  - Text features: 11 (length, word count, punctuation, etc.)")
    logger.info("  - Emotion word features: 5 (positive/negative/arousal word counts)")
    logger.info("  - Tense features: 4 (past/present/future indicators)")
    logger.info("  - Temporal features: 10 (hour, day, cyclical encodings, time gaps)")
    logger.info("  - User features: 7 (user statistics)")
    logger.info("  - Sequence features: 16 (position, lag emotions, rolling stats)")

    logger.info(f"\nOutput saved to: {output_path}")
    logger.info("Ready for model training!")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "raw" / "train_subtask2a.csv"
    output_path = project_root / "data" / "processed" / "subtask2a_features.csv"

    try:
        df = prepare_subtask2a_features(data_path, output_path)

        print("\n" + "="*80)
        print("SUBTASK 2A FEATURE PREPARATION COMPLETED!")
        print("="*80)
        print(f"Output: {output_path}")
        print(f"Records: {len(df)}")
        print(f"Users: {df['user_id'].nunique()}")
        print(f"Features: {len(df.columns)} columns")
        print("\nKey features for modeling:")
        print("  ✓ Text embeddings (use text_cleaned for RoBERTa)")
        print("  ✓ User embeddings (use user_id)")
        print("  ✓ Previous emotions (valence_lag1, arousal_lag1, etc.)")
        print("  ✓ Temporal features (hour_sin/cos, time_gap_log)")
        print("  ✓ Linguistic features (sentiment_score, emotion word counts)")
        print("\nNext step: Train model with these features!")

    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
