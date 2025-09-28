"""
Text preprocessing utilities for SemEval-2026 Task 2
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities for emotion analysis.
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        remove_html: bool = True,
        remove_urls: bool = True,
        normalize_whitespace: bool = True,
        min_length: int = 10,
        max_length: int = 5000
    ):
        """
        Initialize the text preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            remove_html: Whether to remove HTML tags
            remove_urls: Whether to remove URLs
            normalize_whitespace: Whether to normalize whitespace
            min_length: Minimum text length
            max_length: Maximum text length
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.normalize_whitespace = normalize_whitespace
        self.min_length = min_length
        self.max_length = max_length

        # Download required NLTK data
        self._download_nltk_data()

        # Initialize tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

        # Initialize spaCy model for advanced preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Some features may be limited.")
            self.nlp = None

    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        # Tokenize and process words
        if self.remove_stopwords or self.lemmatize:
            words = word_tokenize(text)

            if self.remove_stopwords:
                words = [word for word in words if word.lower() not in self.stop_words]

            if self.lemmatize:
                words = [self.lemmatizer.lemmatize(word) for word in words]

            text = ' '.join(words)

        return text

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract text features that might be relevant for emotion prediction.

        Args:
            text: Input text

        Returns:
            Dictionary of text features
        """
        features = {}

        # Basic features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0

        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['period_count'] = text.count('.')
        features['comma_count'] = text.count(',')

        # Case features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['capitalized_words'] = sum(1 for word in text.split() if word and word[0].isupper())

        # Sentiment features using TextBlob
        try:
            blob = TextBlob(text)
            features['polarity'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['polarity'] = 0.0
            features['subjectivity'] = 0.0

        # Advanced features using spaCy
        if self.nlp:
            try:
                doc = self.nlp(text)
                features['named_entities'] = len(doc.ents)
                features['pos_noun_ratio'] = sum(1 for token in doc if token.pos_ == 'NOUN') / len(doc) if doc else 0
                features['pos_verb_ratio'] = sum(1 for token in doc if token.pos_ == 'VERB') / len(doc) if doc else 0
                features['pos_adj_ratio'] = sum(1 for token in doc if token.pos_ == 'ADJ') / len(doc) if doc else 0
                features['pos_adv_ratio'] = sum(1 for token in doc if token.pos_ == 'ADV') / len(doc) if doc else 0
            except:
                logger.warning("Error processing text with spaCy")

        return features

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess text in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with preprocessed text
        """
        df = df.copy()

        # Clean text
        df[f'{text_column}_cleaned'] = df[text_column].apply(self.clean_text)

        # Filter by length
        if self.min_length > 0:
            df = df[df[f'{text_column}_cleaned'].str.len() >= self.min_length]

        if self.max_length > 0:
            df = df[df[f'{text_column}_cleaned'].str.len() <= self.max_length]

        # Extract features
        feature_dicts = df[f'{text_column}_cleaned'].apply(self.extract_features)
        feature_df = pd.DataFrame(feature_dicts.tolist())

        # Add feature columns to the dataframe
        for col in feature_df.columns:
            df[f'feature_{col}'] = feature_df[col]

        logger.info(f"Preprocessed {len(df)} texts, extracted {len(feature_df.columns)} features")

        return df


class TemporalPreprocessor:
    """
    Preprocessing utilities specifically for temporal emotion data.
    """

    def __init__(
        self,
        text_preprocessor: Optional[TextPreprocessor] = None,
        time_window: str = '1D',  # pandas time window
        min_sequences_per_user: int = 5,
        normalize_emotions: bool = True
    ):
        """
        Initialize the temporal preprocessor.

        Args:
            text_preprocessor: Text preprocessor instance
            time_window: Time window for grouping temporal data
            min_sequences_per_user: Minimum number of sequences required per user
            normalize_emotions: Whether to normalize emotion values
        """
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.time_window = time_window
        self.min_sequences_per_user = min_sequences_per_user
        self.normalize_emotions = normalize_emotions

    def process_temporal_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        valence_column: str = 'valence',
        arousal_column: str = 'arousal',
        user_id_column: str = 'user_id',
        timestamp_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Process temporal emotion data.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            valence_column: Name of the valence column
            arousal_column: Name of the arousal column
            user_id_column: Name of the user ID column
            timestamp_column: Name of the timestamp column

        Returns:
            Processed DataFrame
        """
        df = df.copy()

        # Convert timestamp to datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Sort by user and timestamp
        df = df.sort_values([user_id_column, timestamp_column])

        # Preprocess text
        df = self.text_preprocessor.preprocess_dataframe(df, text_column)

        # Filter users with insufficient data
        user_counts = df[user_id_column].value_counts()
        valid_users = user_counts[user_counts >= self.min_sequences_per_user].index
        df = df[df[user_id_column].isin(valid_users)]

        # Normalize emotion values if requested
        if self.normalize_emotions:
            df[f'{valence_column}_normalized'] = self._normalize_emotions(df[valence_column])
            df[f'{arousal_column}_normalized'] = self._normalize_emotions(df[arousal_column])

        # Add temporal features
        df = self._add_temporal_features(df, user_id_column, timestamp_column, valence_column, arousal_column)

        logger.info(f"Processed temporal data: {len(df)} samples from {df[user_id_column].nunique()} users")

        return df

    def _normalize_emotions(self, emotions: pd.Series) -> pd.Series:
        """Normalize emotion values to [-1, 1] range."""
        min_val = emotions.min()
        max_val = emotions.max()

        if max_val == min_val:
            return emotions * 0

        return 2 * (emotions - min_val) / (max_val - min_val) - 1

    def _add_temporal_features(
        self,
        df: pd.DataFrame,
        user_id_column: str,
        timestamp_column: str,
        valence_column: str,
        arousal_column: str
    ) -> pd.DataFrame:
        """Add temporal features to the dataframe."""

        # Time-based features
        df['hour'] = df[timestamp_column].dt.hour
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['day_of_month'] = df[timestamp_column].dt.day
        df['month'] = df[timestamp_column].dt.month

        # User-specific temporal features
        for user_id in df[user_id_column].unique():
            user_mask = df[user_id_column] == user_id
            user_data = df[user_mask].copy()

            # Time since first entry for this user
            first_timestamp = user_data[timestamp_column].min()
            df.loc[user_mask, 'time_since_start'] = (user_data[timestamp_column] - first_timestamp).dt.total_seconds()

            # Rolling statistics
            window_size = min(5, len(user_data))
            if window_size > 1:
                df.loc[user_mask, 'valence_rolling_mean'] = user_data[valence_column].rolling(window=window_size, min_periods=1).mean()
                df.loc[user_mask, 'arousal_rolling_mean'] = user_data[arousal_column].rolling(window=window_size, min_periods=1).mean()
                df.loc[user_mask, 'valence_rolling_std'] = user_data[valence_column].rolling(window=window_size, min_periods=1).std().fillna(0)
                df.loc[user_mask, 'arousal_rolling_std'] = user_data[arousal_column].rolling(window=window_size, min_periods=1).std().fillna(0)

                # Emotion trends
                df.loc[user_mask, 'valence_trend'] = user_data[valence_column].diff().fillna(0)
                df.loc[user_mask, 'arousal_trend'] = user_data[arousal_column].diff().fillna(0)

        return df

    def create_temporal_splits(
        self,
        df: pd.DataFrame,
        user_id_column: str = 'user_id',
        timestamp_column: str = 'timestamp',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits.

        Args:
            df: Input DataFrame
            user_id_column: Name of the user ID column
            timestamp_column: Name of the timestamp column
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        train_dfs = []
        val_dfs = []
        test_dfs = []

        for user_id in df[user_id_column].unique():
            user_data = df[df[user_id_column] == user_id].sort_values(timestamp_column)
            n_samples = len(user_data)

            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))

            train_dfs.append(user_data.iloc[:train_end])
            val_dfs.append(user_data.iloc[train_end:val_end])
            test_dfs.append(user_data.iloc[val_end:])

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        logger.info(f"Created temporal splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df