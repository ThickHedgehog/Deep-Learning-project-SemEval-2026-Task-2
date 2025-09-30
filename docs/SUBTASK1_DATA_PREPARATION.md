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

A lightweight, dependency-minimal script that:
- Loads and analyzes the raw dataset
- Applies basic text cleaning
- Creates temporal sequences
- Generates train/val/test splits
- Saves processed data in multiple formats
- Generates comprehensive analysis reports

## Files Generated

After running the data preparation pipeline:

```
data/processed/
├── subtask1_train.csv          # Training data (1,865 samples)
├── subtask1_val.csv            # Validation data (410 samples)  
├── subtask1_test.csv           # Test data (475 samples)
├── subtask1_sequences.json     # Temporal sequences (2,236 sequences)
├── subtask1_metadata.json      # Dataset metadata
└── subtask1_analysis_report.txt # Comprehensive analysis report
```

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
```bash
# Run complete data preparation
python simple_data_prep.py

# Analyze the prepared data
python example_usage.py
```

### Advanced Usage with PyTorch
```python
from prepare_subtask1_data import Subtask1DataPreparator

# Initialize preparator
preparator = Subtask1DataPreparator(
    sequence_length=5,
    min_sequences_per_user=3
)

# Run complete pipeline
results = preparator.prepare_subtask1_data()

# Access datasets and dataloaders
train_loader = results["dataloaders"]["train"]
val_loader = results["dataloaders"]["val"]
```

### Loading Processed Data
```python
import pandas as pd
import json

# Load CSV data
train_df = pd.read_csv("data/processed/subtask1_train.csv")
val_df = pd.read_csv("data/processed/subtask1_val.csv")
test_df = pd.read_csv("data/processed/subtask1_test.csv")

# Load sequences
with open("data/processed/subtask1_sequences.json", 'r') as f:
    sequences = json.load(f)
```

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

The data preparation pipeline provides a robust foundation for Subtask 1 modeling. The processed data includes:
- Clean, feature-rich temporal sequences
- Proper train/val/test splits maintaining temporal order
- Comprehensive metadata and analysis
- Multiple format options for different use cases

The pipeline is designed to handle the unique challenges of longitudinal affect assessment while providing flexibility for different modeling approaches.