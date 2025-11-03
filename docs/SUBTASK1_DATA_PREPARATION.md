# Subtask 1: Longitudinal Affect Assessment - Data Preparation

## Overview

This document describes the complete data preparation pipeline for **Subtask 1: Longitudinal Affect Assessment** of SemEval-2026 Task 2. The task involves predicting Valence & Arousal scores for sequences of texts in chronological order.

## Task Description

**Subtask 1** requires:
- Input: A sequence of m texts (essays or feeling words) in chronological order: e₁, e₂, ..., eₘ
- Output: Valence & Arousal (V&A) scores for each text: (v₁, a₁), (v₂, a₂), ..., (vₘ, aₘ)
- Challenge: Handle both **seen users** (appear in training) and **unseen users** (not in training)

## Dataset Analysis

### Basic Statistics
- **Total samples**: 2,764 text entries
- **Unique users**: 137 users
- **Date range**: March 2021 to December 2024 (3+ years)
- **Collection phases**: 7 different phases
- **Text types**: 
  - Essays (narrative descriptions): 1,331 samples
  - Word lists (emotion keywords): 1,433 samples

### Emotion Distribution
- **Valence**: Range [-2.0, 2.0], Mean: 0.22, Std: 1.29
- **Arousal**: Range [0.0, 2.0], Mean: 0.75, Std: 0.75
- **Correlation**: Valence-Arousal correlation is weak (r=0.038)

### Temporal Characteristics
- **Time span**: 1,365 days of data collection
- **Users with sufficient data**: 130 users (≥3 entries each)
- **Entries per user**: Min: 3, Max: 206, Average: 21.2, Median: 14.0
- **Sequences created**: 2,236 temporal sequences (length=5)

## Data Preparation Pipeline

### 1. Text Preprocessing (`src/data/preprocessor.py`)

**TextPreprocessor** features:
- Basic text cleaning (whitespace normalization, HTML removal)
- Feature extraction (length, word count, sentiment, POS ratios)
- Configurable cleaning options (preserves emotion indicators)

**TemporalPreprocessor** features:
- User-wise temporal processing
- Rolling statistics (emotion trends)
- Temporal feature engineering
- Train/val/test temporal splitting

### 2. Dataset Classes (`src/data/dataset.py`)

**EmotionDataset**: Basic emotion prediction from single texts
- Text tokenization with transformers
- Emotion label handling
- Support for both valence and arousal prediction

**TemporalEmotionDataset**: Sequential emotion prediction
- Creates temporal sequences of configurable length
- Handles varying user sequence lengths
- Supports overlapping sequences for training

**EvaluationDataset**: Evaluation-specific functionality
- Preserves original data for analysis
- Metadata handling for detailed evaluation

### 3. Data Loading (`src/data/loader.py`)

**DataLoader**: General-purpose data loading
- PyTorch DataLoader integration
- Custom collation for temporal data
- Distributed training support

**TemporalDataLoader**: Specialized for temporal sequences
- Sequence padding and masking
- Temporal-aware batching
- Support for variable sequence lengths

### 4. Simple Data Preparation (`simple_data_prep.py`)

**Purpose**: Comprehensive feature engineering script that transforms raw data into ML-ready dataset.

**Location**: `scripts/data_preparation/simple_data_prep.py`

**Features Engineered** (33 new features):

1. **Text Features** (9): text_length, word_count, sentence_count, avg_word_length, exclamation_count, question_count, uppercase_ratio, capitalized_words
2. **Temporal Features** (6): hour, day_of_week, month, is_weekend, hours_since_start, time_gap_hours
3. **User Statistics** (10): user_text_id_count, user_valence_mean/std, user_arousal_mean/std, user_timestamp_min/max, entry_number, user_days_active, user_entry_frequency
4. **Rolling Features** (6): valence/arousal rolling_mean/std (window=3), valence/arousal_diff
5. **Encoded Features** (2): is_words_encoded, collection_phase_encoded

**Text Cleaning**:
- Apostrophe normalization (I ' ve → I've)
- Whitespace normalization
- Punctuation standardization

**Output**: 
- File: `data/processed/subtask1_processed.csv`
- Size: 2,764 rows × 41 columns (8 original + 33 engineered)
- No train/val/test split applied (flexible for custom strategies)

**Design**:
- Per-user temporal processing
- Missing value handling (NaN → 0)
- Chronological order preservation
- Detailed logging

## Files Generated

**Primary Output**:
```
data/processed/
└── subtask1_processed.csv       # Complete dataset: 2,764 samples × 41 columns
                                 # (8 original + 33 engineered features)
```

**Feature Breakdown**:
- 8 original columns: user_id, text_id, text, timestamp, collection_phase, is_words, valence, arousal
- 1 cleaned text: text_cleaned
- 9 text features: text_length, word_count, sentence_count, etc.
- 6 temporal features: hour, day_of_week, month, is_weekend, hours_since_start, time_gap_hours
- 10 user statistics: user_text_id_count, user_valence_mean/std, user_arousal_mean/std, etc.
- 5 user temporal: entry_number, hours_since_start, user_days_active, user_entry_frequency, time_gap_hours
- 6 rolling statistics: valence/arousal rolling_mean/std, valence/arousal_diff
- 2 encoded categorical: is_words_encoded, collection_phase_encoded

## Key Insights from Analysis

### Text Characteristics
- **Essays**: Average 259 characters, 54 words
- **Word lists**: Average 43 characters, 9 words
- **Text-emotion correlation**: Weak correlation between text length and emotions

### Temporal Patterns
- **Emotion stability**: Low variability within sequences (std ~0.8 for valence)
- **Emotion changes**: Small average changes across sequences
- **User differences**: High variability between users (1-202 sequences per user)

### Modeling Challenges
1. **Mixed input types**: Essays vs. word lists require different processing
2. **User variability**: Large differences in sequence lengths and patterns
3. **Temporal sparsity**: Irregular time intervals between entries
4. **Seen vs. unseen users**: Need generalization to new users

## Usage Instructions

### Quick Start
Run the data preparation script:
```bash
python scripts/data_preparation/simple_data_prep.py
```

Output: `data/processed/subtask1_processed.csv` (2,764 samples × 41 columns)

### Data Splitting Options
After loading the processed CSV with pandas, you can split the data using:
- **Random split**: Simple sklearn train_test_split (not recommended for temporal data)
- **Temporal split**: Chronological split maintaining time order (recommended)
- **User-wise temporal split**: Per-user chronological split (best for seen/unseen user evaluation)

## Modeling Recommendations

### 1. Text Encoding
- **Pre-trained models**: Use BERT/RoBERTa for text representation
- **Input handling**: Separate processing for essays vs. word lists
- **Feature preservation**: Keep punctuation and case for emotion analysis

### 2. Temporal Modeling
- **Sequence models**: LSTM/GRU for temporal dependencies
- **Attention mechanisms**: Focus on important time steps
- **Bidirectional processing**: Better context understanding

### 3. User Modeling
- **User embeddings**: Personalization for seen users
- **Domain adaptation**: Transfer learning for unseen users
- **Baseline modeling**: User-specific emotion baselines

### 4. Architecture Design
- **Multi-task learning**: Joint valence and arousal prediction
- **Loss functions**: MSE or Huber loss for regression
- **Regularization**: Dropout and weight decay for generalization

### 5. Evaluation Strategy
- **Metrics**: Pearson/Spearman correlation, RMSE
- **Cross-validation**: Temporal splits to avoid leakage
- **User-level evaluation**: Performance on seen vs. unseen users

## Implementation Notes

### Dependencies
- **Core**: pandas, numpy, json
- **ML**: torch, transformers (for advanced features)
- **NLP**: nltk, spacy (optional for advanced preprocessing)

### Performance Considerations
- **Memory**: Large sequences may require gradient checkpointing
- **Training**: Use temporal batching for efficiency
- **Inference**: Cache user embeddings for faster prediction

### Data Handling
- **Temporal splits**: Maintain chronological order
- **Sequence overlap**: Use for training, avoid for evaluation
- **Missing data**: Handle irregular time intervals appropriately

## Conclusion

The data preparation pipeline provides a robust foundation for Subtask 1 modeling. The `simple_data_prep.py` script generates a comprehensive processed dataset with:

**Key Outputs**:
- **Single unified dataset**: 2,764 samples × 41 columns
- **Rich feature set**: 33 engineered features covering text, temporal, user, and sequential aspects
- **Clean data**: Advanced text cleaning with apostrophe normalization and whitespace handling
- **Temporal preservation**: Chronological ordering maintained for proper sequence modeling
- **Flexible workflow**: No pre-applied splits, allowing custom train/val/test strategies

**Feature Categories**:
1. **Text Features** (9): Character/word counts, sentence structure, punctuation patterns
2. **Temporal Features** (6): Time-based patterns (hour, day, month, weekend, gaps)
3. **User Features** (10): User-level statistics and activity patterns
4. **Rolling Features** (6): Sequential patterns with rolling windows
5. **Encoded Features** (2): Categorical variables converted to numeric

**Advantages**:
- **Comprehensive**: All necessary features for emotion prediction
- **Production-ready**: Proper handling of edge cases and missing values
- **Well-documented**: Detailed logging and clear feature naming
- **Flexible**: Supports multiple modeling approaches (BERT+LSTM, user embeddings, etc.)
- **Lightweight**: Minimal dependencies (pandas, numpy, re)

**Next Steps**:
1. Apply appropriate train/val/test splitting strategy
2. Implement PyTorch Dataset class for efficient loading
3. Build model architecture (BERT encoder + LSTM temporal modeling)
4. Train and evaluate on both seen and unseen users

The pipeline is designed to handle the unique challenges of longitudinal affect assessment while providing maximum flexibility for different modeling approaches and evaluation strategies.