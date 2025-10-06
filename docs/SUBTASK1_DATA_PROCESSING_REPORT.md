# Subtask1 Data Processing Report

## Overview
This document describes the data processing pipeline for Subtask1 of the SemEval-2026 Task 2 project. The data has been successfully processed and prepared for further use in machine learning applications.

## Input Data
- **File**: `data/raw/TRAIN_RELEASE_3SEP2025/train_subtask1.csv`
- **Records**: 2,764
- **Users**: 137
- **Time Period**: from 2021-03-11 to 2024-12-19 (1,379 days)

### Original Data Structure:
- `user_id`: User identifier
- `text_id`: Text identifier
- `text`: Text content
- `timestamp`: Timestamp
- `collection_phase`: Data collection phase (1-7)
- `is_words`: Text type (True = word list, False = full text)
- `valence`: Emotion valence (-2.0 to 2.0)
- `arousal`: Emotion arousal (0.0 to 2.0)

## Processing Pipeline

### 1. Data Cleaning
- Removal of records with missing critical data
- Conversion of timestamps to datetime format
- Removal of very short texts (< 5 characters)
- Basic text cleaning (whitespace normalization, HTML and URL removal)

### 2. Text Feature Extraction
**11 text features** were extracted:
- `feature_length`: Text length
- `feature_word_count`: Word count
- `feature_sentence_count`: Sentence count
- `feature_avg_word_length`: Average word length
- `feature_avg_sentence_length`: Average sentence length
- `feature_exclamation_count`: Exclamation mark count
- `feature_question_count`: Question mark count
- `feature_period_count`: Period count
- `feature_comma_count`: Comma count
- `feature_uppercase_ratio`: Uppercase letter ratio
- `feature_capitalized_words`: Number of capitalized words

### 3. Temporal Features
**5 temporal features** were added:
- `hour`: Hour of day
- `day_of_week`: Day of week
- `day_of_month`: Day of month
- `month`: Month
- `year`: Year

### 4. User Features
**9 user features** were added:
- `user_valence_count`: Number of user records
- `user_valence_mean`: User's average valence
- `user_valence_std`: Valence standard deviation
- `user_arousal_mean`: User's average arousal
- `user_arousal_std`: Arousal standard deviation
- `user_timestamp_min`: User's first timestamp
- `user_timestamp_max`: User's last timestamp
- `user_days_active`: Number of days active

## Processing Results

### Output Files
1. **`data/processed/subtask1_processed.csv`** - complete processed dataset
   - Size: 2,764 records × 33 columns
   - Includes all original data + extracted features

2. **`data/processed/subtask1_sample.csv`** - sample data for testing
   - Size: 200 records × 33 columns
   - Stratified sample by users

### Processed Data Statistics

#### Emotional Parameters:
- **Valence**: mean = 0.217, std = 1.292, range = [-2.0, 2.0]
- **Arousal**: mean = 0.751, std = 0.754, range = [0.0, 2.0]

#### Text Characteristics:
- **Average text length**: 158 characters
- **Average word count**: 33 words
- **Average sentence count**: 2.5

#### Distribution by Collection Phase:
- Phase 1: 229 records (8.3%)
- Phase 2: 369 records (13.4%)
- Phase 3: 289 records (10.5%)
- Phase 4: 163 records (5.9%)
- Phase 5: 81 records (2.9%)
- Phase 6: 76 records (2.7%)
- Phase 7: 1,557 records (56.3%)

#### Text Types:
- Full texts: 1,331 records (48.2%)
- Word lists: 1,433 records (51.8%)

#### Users:
- Records per user: from 2 to 206 (average = 20.2)

## Scripts Used

1. **`scripts/data_preparation/prepare_subtask1_simple.py`**
   - Main data processing script
   - Performs complete processing from raw data to ready dataset

2. **`scripts/data_preparation/analyze_processed_data.py`**
   - Analysis of processed data
   - Sample dataset creation
   - Statistics generation

## Data Quality

### Strengths:
- ✅ All records contain complete emotion data
- ✅ Wide temporal range (3+ years)
- ✅ Diverse user base (137 unique users)
- ✅ Good representation of different text types
- ✅ Rich set of extracted features

### Characteristics:
- 📊 Uneven distribution across collection phases (56% in phase 7)
- 📊 Large variation in user activity (2-206 records)
- 📊 Mix of full texts and word lists
- 📊 Predominance of neutral and positive emotions

## Next Steps

The processed data is ready for:
1. **Dataset splitting** (train/validation/test)
2. **PyTorch dataset creation** using classes from `src/data/dataset.py`
3. **Model training** for valence and arousal prediction
4. **Temporal modeling** using user sequences

## Reproducibility

To reproduce the processing, run:
```bash
# Basic processing
python scripts/data_preparation/prepare_subtask1_simple.py

# Results analysis
python scripts/data_preparation/analyze_processed_data.py
```

All logs are saved to files and displayed in the console. Logs are now stored in the `logs/` directory.