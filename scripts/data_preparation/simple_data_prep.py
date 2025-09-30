"""
Simple Subtask 1 Data Analysis and Preparation Script

This script analyzes the dataset for Subtask 1 and prepares it for training
without requiring heavy ML dependencies. It focuses on data understanding
and basic preprocessing.

Usage:
    python simple_data_prep.py
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleSubtask1DataPrep:
    """
    Simple data preparation for Subtask 1: Longitudinal Affect Assessment.
    
    This class performs:
    1. Data loading and analysis
    2. Basic text cleaning
    3. Temporal sequence creation
    4. Train/val/test splitting
    5. Statistics generation
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        sequence_length: int = 5,
        min_sequences_per_user: int = 3
    ):
        """Initialize the data preparator."""
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        
        self.data_path = data_path or str(project_root / "data/raw/train_subtask1.csv")
        self.output_dir = Path(output_dir) if output_dir else (project_root / "data/processed")
        self.sequence_length = sequence_length
        self.min_sequences_per_user = min_sequences_per_user
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load and analyze the raw data."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Generate comprehensive analysis
        self._log_data_analysis(df)
        
        return df
    
    def _log_data_analysis(self, df: pd.DataFrame) -> None:
        """Log comprehensive data analysis."""
        logger.info("=== DATASET ANALYSIS ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Unique users: {df['user_id'].nunique()}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Collection phases: {sorted(df['collection_phase'].unique())}")
        
        # Text type analysis
        essay_count = (~df['is_words']).sum()
        words_count = df['is_words'].sum()
        logger.info(f"Text types - Essays: {essay_count}, Word lists: {words_count}")
        
        # Emotion analysis
        logger.info(f"Valence - Range: [{df['valence'].min()}, {df['valence'].max()}], "
                   f"Mean: {df['valence'].mean():.2f}, Std: {df['valence'].std():.2f}")
        logger.info(f"Arousal - Range: [{df['arousal'].min()}, {df['arousal'].max()}], "
                   f"Mean: {df['arousal'].mean():.2f}, Std: {df['arousal'].std():.2f}")
        
        # User statistics
        user_stats = df.groupby('user_id').agg({
            'text_id': 'count',
            'timestamp': ['min', 'max'],
            'collection_phase': 'nunique'
        })
        user_stats.columns = ['num_entries', 'first_timestamp', 'last_timestamp', 'num_phases']
        
        logger.info(f"Users with >={self.min_sequences_per_user} entries: "
                   f"{(user_stats['num_entries'] >= self.min_sequences_per_user).sum()}")
        logger.info(f"Average entries per user: {user_stats['num_entries'].mean():.1f}")
        logger.info(f"User entry distribution: "
                   f"Min: {user_stats['num_entries'].min()}, "
                   f"Max: {user_stats['num_entries'].max()}, "
                   f"Median: {user_stats['num_entries'].median():.1f}")
    
    def clean_text(self, text: str) -> str:
        """Apply basic text cleaning."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive dots (common in the dataset)
        text = re.sub(r'\\.{3,}', '...', text)
        
        # Basic cleaning while preserving emotion indicators
        # Keep punctuation as it's important for emotion analysis
        
        return text
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to the data."""
        logger.info("Starting data preprocessing...")
        
        df = df.copy()
        
        # Clean text
        df['text_cleaned'] = df['text'].apply(self.clean_text)
        
        # Filter out empty texts
        df = df[df['text_cleaned'].str.len() > 0]
        
        # Filter users with sufficient data
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_sequences_per_user].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Add text features
        df['text_length'] = df['text_cleaned'].str.len()
        df['word_count'] = df['text_cleaned'].str.split().str.len()
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        logger.info(f"After preprocessing: {len(df)} samples from {df['user_id'].nunique()} users")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # User-specific temporal features
        for user_id in df['user_id'].unique():
            user_mask = df['user_id'] == user_id
            user_data = df[user_mask].copy()
            
            # Time since first entry for this user (in hours)
            first_timestamp = user_data['timestamp'].min()
            time_diff = (user_data['timestamp'] - first_timestamp).dt.total_seconds() / 3600
            df.loc[user_mask, 'hours_since_start'] = time_diff
            
            # Rolling emotion statistics (for sequences of length 3+)
            if len(user_data) >= 3:
                window_size = min(3, len(user_data))
                valence_rolling = user_data['valence'].rolling(window=window_size, min_periods=1)
                arousal_rolling = user_data['arousal'].rolling(window=window_size, min_periods=1)
                
                df.loc[user_mask, 'valence_rolling_mean'] = valence_rolling.mean()
                df.loc[user_mask, 'arousal_rolling_mean'] = arousal_rolling.mean()
                df.loc[user_mask, 'valence_rolling_std'] = valence_rolling.std().fillna(0)
                df.loc[user_mask, 'arousal_rolling_std'] = arousal_rolling.std().fillna(0)
                
                # Emotion change trends
                df.loc[user_mask, 'valence_diff'] = user_data['valence'].diff().fillna(0)
                df.loc[user_mask, 'arousal_diff'] = user_data['arousal'].diff().fillna(0)
        
        return df
    
    def create_temporal_sequences(self, df: pd.DataFrame) -> List[Dict]:
        """Create temporal sequences for training."""
        logger.info("Creating temporal sequences...")
        
        sequences = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].copy()
            
            # Skip users with insufficient data
            if len(user_data) < self.sequence_length:
                continue
            
            # Create overlapping sequences
            for i in range(len(user_data) - self.sequence_length + 1):
                sequence_data = user_data.iloc[i:i + self.sequence_length]
                
                sequences.append({
                    'user_id': user_id,
                    'sequence_id': f"{user_id}_{i}",
                    'start_idx': i,
                    'texts': sequence_data['text_cleaned'].tolist(),
                    'valences': sequence_data['valence'].tolist(),
                    'arousals': sequence_data['arousal'].tolist(),
                    'timestamps': sequence_data['timestamp'].tolist(),
                    'text_features': {
                        'lengths': sequence_data['text_length'].tolist(),
                        'word_counts': sequence_data['word_count'].tolist()
                    }
                })
        
        logger.info(f"Created {len(sequences)} temporal sequences")
        return sequences
    
    def create_temporal_splits(
        self, 
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create temporal train/validation/test splits."""
        logger.info("Creating temporal data splits...")
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].sort_values('timestamp')
            n_samples = len(user_data)
            
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_dfs.append(user_data.iloc[:train_end])
            val_dfs.append(user_data.iloc[train_end:val_end])
            test_dfs.append(user_data.iloc[val_end:])
        
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        sequences: List[Dict]
    ) -> Dict[str, str]:
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        file_paths = {}
        
        # Save DataFrames
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            file_path = self.output_dir / f"subtask1_{name}.csv"
            df.to_csv(file_path, index=False)
            file_paths[f"{name}_csv"] = str(file_path)
            logger.info(f"Saved {name} CSV: {len(df)} samples to {file_path}")
        
        # Save sequences as JSON
        sequences_file = self.output_dir / "subtask1_sequences.json"
        with open(sequences_file, 'w', encoding='utf-8') as f:
            # Convert timestamps and numpy types to JSON-serializable formats
            sequences_serializable = []
            for seq in sequences:
                seq_copy = {
                    'user_id': int(seq['user_id']),
                    'sequence_id': str(seq['sequence_id']),
                    'start_idx': int(seq['start_idx']),
                    'texts': seq['texts'],
                    'valences': [float(v) for v in seq['valences']],
                    'arousals': [float(a) for a in seq['arousals']],
                    'timestamps': [ts.isoformat() for ts in seq['timestamps']],
                    'text_features': {
                        'lengths': [int(length) for length in seq['text_features']['lengths']],
                        'word_counts': [int(word_count) for word_count in seq['text_features']['word_counts']]
                    }
                }
                sequences_serializable.append(seq_copy)
            
            json.dump(sequences_serializable, f, indent=2, ensure_ascii=False)
        
        file_paths["sequences_json"] = str(sequences_file)
        logger.info(f"Saved sequences JSON: {len(sequences)} sequences to {sequences_file}")
        
        # Save metadata
        metadata = {
            "creation_time": datetime.now().isoformat(),
            "sequence_length": self.sequence_length,
            "min_sequences_per_user": self.min_sequences_per_user,
            "data_splits": {
                "train_samples": len(train_df),
                "val_samples": len(val_df), 
                "test_samples": len(test_df),
                "total_sequences": len(sequences)
            },
            "user_statistics": {
                "total_users": train_df['user_id'].nunique() + val_df['user_id'].nunique() + test_df['user_id'].nunique(),
                "train_users": train_df['user_id'].nunique() if not train_df.empty else 0,
                "val_users": val_df['user_id'].nunique() if not val_df.empty else 0,
                "test_users": test_df['user_id'].nunique() if not test_df.empty else 0
            }
        }
        
        metadata_file = self.output_dir / "subtask1_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        file_paths["metadata_json"] = str(metadata_file)
        logger.info(f"Saved metadata to {metadata_file}")
        
        return file_paths
    
    def generate_analysis_report(self, df: pd.DataFrame, sequences: List[Dict]) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("SUBTASK 1: LONGITUDINAL AFFECT ASSESSMENT - DATA ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"  • Total samples: {len(df)}")
        report.append(f"  • Unique users: {df['user_id'].nunique()}")
        report.append(f"  • Temporal sequences created: {len(sequences)}")
        report.append("")
        
        # Text analysis
        essay_mask = ~df['is_words']
        words_mask = df['is_words']
        
        report.append("TEXT ANALYSIS:")
        report.append(f"  • Essays: {essay_mask.sum()} samples")
        report.append(f"    - Avg length: {df[essay_mask]['text_length'].mean():.0f} chars")
        report.append(f"    - Avg words: {df[essay_mask]['word_count'].mean():.0f}")
        report.append(f"  • Word lists: {words_mask.sum()} samples")
        report.append(f"    - Avg length: {df[words_mask]['text_length'].mean():.0f} chars")
        report.append(f"    - Avg words: {df[words_mask]['word_count'].mean():.0f}")
        report.append("")
        
        # Emotion analysis
        report.append("EMOTION ANALYSIS:")
        report.append(f"  • Valence - Range: [{df['valence'].min()}, {df['valence'].max()}]")
        report.append(f"    - Mean: {df['valence'].mean():.2f}, Std: {df['valence'].std():.2f}")
        report.append(f"  • Arousal - Range: [{df['arousal'].min()}, {df['arousal'].max()}]")
        report.append(f"    - Mean: {df['arousal'].mean():.2f}, Std: {df['arousal'].std():.2f}")
        report.append("")
        
        # Temporal analysis
        time_span = (df['timestamp'].max() - df['timestamp'].min()).days
        report.append("TEMPORAL ANALYSIS:")
        report.append(f"  • Time span: {time_span} days")
        report.append(f"  • Entries per user: {df.groupby('user_id').size().describe()}")
        report.append("")
        
        # Sequence analysis
        user_seq_counts = {}
        for seq in sequences:
            user_id = seq['user_id']
            user_seq_counts[user_id] = user_seq_counts.get(user_id, 0) + 1
        
        report.append("SEQUENCE ANALYSIS:")
        report.append(f"  • Users with sequences: {len(user_seq_counts)}")
        report.append(f"  • Sequences per user - Min: {min(user_seq_counts.values())}, "
                     f"Max: {max(user_seq_counts.values())}, "
                     f"Avg: {np.mean(list(user_seq_counts.values())):.1f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS FOR MODELING:")
        report.append("  • Use BERT/RoBERTa for text encoding")
        report.append("  • Implement LSTM/GRU for temporal modeling")
        report.append("  • Consider user embeddings for personalization")
        report.append("  • Use both essay and word-list data (different input processing)")
        report.append("  • Apply temporal attention mechanisms")
        report.append("")
        
        return "\\n".join(report)
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete data preparation and analysis pipeline."""
        logger.info("=== STARTING COMPLETE SUBTASK 1 ANALYSIS ===")
        
        # Step 1: Load and analyze
        df = self.load_and_analyze_data()
        
        # Step 2: Preprocess
        processed_df = self.preprocess_data(df)
        
        # Step 3: Create sequences
        sequences = self.create_temporal_sequences(processed_df)
        
        # Step 4: Create splits
        train_df, val_df, test_df = self.create_temporal_splits(processed_df)
        
        # Step 5: Save data
        file_paths = self.save_processed_data(train_df, val_df, test_df, sequences)
        
        # Step 6: Generate report
        report = self.generate_analysis_report(processed_df, sequences)
        
        # Save report
        report_file = self.output_dir / "subtask1_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Analysis report saved to {report_file}")
        logger.info("=== COMPLETE SUBTASK 1 ANALYSIS FINISHED ===")
        
        return {
            "processed_data": processed_df,
            "sequences": sequences,
            "splits": {"train": train_df, "val": val_df, "test": test_df},
            "file_paths": file_paths,
            "report": report
        }


def main():
    """Run the complete data preparation pipeline."""
    preparator = SimpleSubtask1DataPrep(
        sequence_length=5,  # Good balance for capturing temporal patterns
        min_sequences_per_user=3  # Ensure enough data per user
    )
    
    try:
        results = preparator.run_complete_analysis()
        
        print("\\n" + "="*60)
        print("SUBTASK 1 DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\\nFiles created:")
        for key, path in results["file_paths"].items():
            print(f"  • {key}: {path}")
        
        print("\\n" + results["report"])
        
        return True
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)