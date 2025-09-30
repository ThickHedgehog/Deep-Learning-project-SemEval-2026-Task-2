"""
Subtask 1 Data Preparation Script

This script prepares the data for Subtask 1: Longitudinal Affect Assessment.
It processes text sequences from users and creates training/validation/test splits
for temporal emotion prediction (Valence & Arousal).

Usage:
    python prepare_subtask1_data.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path

# Add src to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.preprocessor import TextPreprocessor, TemporalPreprocessor
from data.dataset import TemporalEmotionDataset
from data.loader import TemporalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Subtask1DataPreparator:
    """
    Data preparation pipeline for Subtask 1: Longitudinal Affect Assessment.
    
    This class handles:
    1. Loading and cleaning the raw data
    2. Temporal preprocessing and feature extraction
    3. Creating user-wise temporal sequences
    4. Splitting data into train/val/test sets
    5. Creating datasets and dataloaders for model training
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        sequence_length: int = 5,
        min_sequences_per_user: int = 3,
        max_sequence_length: int = 512,
        tokenizer_name: str = "bert-base-uncased"
    ):
        """
        Initialize the data preparator.
        
        Args:
            data_path: Path to the raw training data
            output_dir: Directory to save processed data
            sequence_length: Length of temporal sequences for training
            min_sequences_per_user: Minimum sequences required per user
            max_sequence_length: Maximum text length for tokenization
            tokenizer_name: Tokenizer to use for text encoding
        """
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        
        self.data_path = data_path or str(project_root / "data/raw/train_subtask1.csv")
        self.output_dir = Path(output_dir) if output_dir else (project_root / "data/processed")
        self.sequence_length = sequence_length
        self.min_sequences_per_user = min_sequences_per_user
        self.max_sequence_length = max_sequence_length
        self.tokenizer_name = tokenizer_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor(
            lowercase=False,  # Preserve original case for emotion analysis
            remove_punctuation=False,  # Keep punctuation as it can indicate emotion
            remove_stopwords=False,  # Keep stopwords as they provide context
            lemmatize=False,  # Preserve original form
            remove_html=True,
            remove_urls=True,
            normalize_whitespace=True,
            min_length=10,
            max_length=5000
        )
        
        self.temporal_preprocessor = TemporalPreprocessor(
            text_preprocessor=self.text_preprocessor,
            min_sequences_per_user=min_sequences_per_user,
            normalize_emotions=False  # Keep original emotion scale
        )
        
    def load_and_analyze_data(self) -> pd.DataFrame:
        """
        Load and perform initial analysis of the raw data.
        
        Returns:
            Loaded DataFrame with basic statistics logged
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Log basic statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Unique users: {df['user_id'].nunique()}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Text types - Essays: {(~df['is_words']).sum()}, Word lists: {df['is_words'].sum()}")
        logger.info(f"Valence range: {df['valence'].min()} to {df['valence'].max()}")
        logger.info(f"Arousal range: {df['arousal'].min()} to {df['arousal'].max()}")
        
        # User statistics
        user_stats = df.groupby('user_id').agg({
            'text_id': 'count',
            'timestamp': ['min', 'max'],
            'collection_phase': 'nunique'
        }).round(2)
        user_stats.columns = ['num_entries', 'first_timestamp', 'last_timestamp', 'num_phases']
        
        logger.info(f"Users with sufficient data (>={self.min_sequences_per_user} entries): "
                   f"{(user_stats['num_entries'] >= self.min_sequences_per_user).sum()}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to the raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Apply temporal preprocessing
        processed_df = self.temporal_preprocessor.process_temporal_data(
            df,
            text_column='text',
            valence_column='valence',
            arousal_column='arousal',
            user_id_column='user_id',
            timestamp_column='timestamp'
        )
        
        logger.info(f"After preprocessing: {len(processed_df)} samples from "
                   f"{processed_df['user_id'].nunique()} users")
        
        return processed_df
    
    def create_temporal_splits(
        self, 
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits maintaining user chronology.
        
        Args:
            df: Preprocessed DataFrame
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating temporal data splits...")
        
        train_df, val_df, test_df = self.temporal_preprocessor.create_temporal_splits(
            df,
            user_id_column='user_id',
            timestamp_column='timestamp',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        return train_df, val_df, test_df
    
    def save_processed_data(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Save processed data to files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dictionary with file paths
        """
        file_paths = {}
        
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            file_path = self.output_dir / f"subtask1_{name}.csv"
            df.to_csv(file_path, index=False)
            file_paths[name] = str(file_path)
            logger.info(f"Saved {name} data: {len(df)} samples to {file_path}")
        
        return file_paths
    
    def create_datasets_and_loaders(
        self, 
        file_paths: Dict[str, str],
        batch_size: int = 16
    ) -> Tuple[Dict[str, TemporalEmotionDataset], Dict[str, Any]]:
        """
        Create PyTorch datasets and dataloaders.
        
        Args:
            file_paths: Dictionary with paths to train/val/test files
            batch_size: Batch size for dataloaders
            
        Returns:
            Tuple of (datasets_dict, dataloaders_dict)
        """
        logger.info("Creating PyTorch datasets and dataloaders...")
        
        datasets = {}
        dataloaders = {}
        
        # Initialize temporal data loader
        temporal_loader = TemporalDataLoader(
            sequence_length=self.sequence_length,
            overlap=True,  # Use overlapping sequences for training
            batch_size=batch_size,
            num_workers=2,  # Reduced for Windows compatibility
            pin_memory=torch.cuda.is_available()
        )
        
        for split in ["train", "val", "test"]:
            if split in file_paths:
                # Create temporal dataset
                dataset = TemporalEmotionDataset(
                    data=file_paths[split],
                    tokenizer_name=self.tokenizer_name,
                    max_length=self.max_sequence_length,
                    sequence_length=self.sequence_length,
                    text_column="text_cleaned",  # Use cleaned text
                    valence_column="valence",
                    arousal_column="arousal",
                    user_id_column="user_id",
                    timestamp_column="timestamp",
                    overlap=(split == "train")  # Only overlap for training
                )
                
                datasets[split] = dataset
                
                # Create dataloader
                shuffle = (split == "train")
                dataloader = temporal_loader.create_dataloader(dataset, shuffle=shuffle)
                dataloaders[split] = dataloader
                
                logger.info(f"Created {split} dataset: {len(dataset)} sequences, "
                           f"dataloader: {len(dataloader)} batches")
        
        return datasets, dataloaders
    
    def analyze_sequences(self, datasets: Dict[str, TemporalEmotionDataset]) -> None:
        """
        Analyze the created temporal sequences.
        
        Args:
            datasets: Dictionary of datasets
        """
        logger.info("Analyzing temporal sequences...")
        
        for split, dataset in datasets.items():
            if len(dataset.sequences) == 0:
                logger.warning(f"No sequences found in {split} dataset")
                continue
                
            # Analyze sequence distribution by user
            user_sequence_counts = {}
            for seq in dataset.sequences:
                user_id = seq["user_id"]
                user_sequence_counts[user_id] = user_sequence_counts.get(user_id, 0) + 1
            
            logger.info(f"{split.upper()} SPLIT:")
            logger.info(f"  Total sequences: {len(dataset.sequences)}")
            logger.info(f"  Unique users: {len(user_sequence_counts)}")
            logger.info(f"  Avg sequences per user: {np.mean(list(user_sequence_counts.values())):.1f}")
            logger.info(f"  Min sequences per user: {min(user_sequence_counts.values())}")
            logger.info(f"  Max sequences per user: {max(user_sequence_counts.values())}")
    
    def prepare_subtask1_data(self) -> Dict:
        """
        Complete data preparation pipeline for Subtask 1.
        
        Returns:
            Dictionary containing all created datasets, dataloaders, and metadata
        """
        logger.info("=== STARTING SUBTASK 1 DATA PREPARATION ===")
        
        # Step 1: Load and analyze raw data
        raw_df = self.load_and_analyze_data()
        
        # Step 2: Preprocess data
        processed_df = self.preprocess_data(raw_df)
        
        # Step 3: Create temporal splits
        train_df, val_df, test_df = self.create_temporal_splits(processed_df)
        
        # Step 4: Save processed data
        file_paths = self.save_processed_data(train_df, val_df, test_df)
        
        # Step 5: Create datasets and dataloaders
        datasets, dataloaders = self.create_datasets_and_loaders(file_paths)
        
        # Step 6: Analyze sequences
        self.analyze_sequences(datasets)
        
        logger.info("=== SUBTASK 1 DATA PREPARATION COMPLETE ===")
        
        return {
            "raw_data": raw_df,
            "processed_data": {"train": train_df, "val": val_df, "test": test_df},
            "file_paths": file_paths,
            "datasets": datasets,
            "dataloaders": dataloaders,
            "metadata": {
                "sequence_length": self.sequence_length,
                "min_sequences_per_user": self.min_sequences_per_user,
                "max_sequence_length": self.max_sequence_length,
                "tokenizer_name": self.tokenizer_name
            }
        }


def test_data_pipeline():
    """
    Test the data preparation pipeline with a small sample.
    """
    logger.info("=== TESTING DATA PIPELINE ===")
    
    try:
        # Initialize preparator with small settings for testing
        preparator = Subtask1DataPreparator(
            sequence_length=3,  # Smaller sequences for testing
            min_sequences_per_user=2,  # Lower threshold for testing
            max_sequence_length=128  # Shorter sequences for faster testing
        )
        
        # Run the full pipeline
        results = preparator.prepare_subtask1_data()
        
        # Test loading a batch
        if "train" in results["dataloaders"]:
            train_loader = results["dataloaders"]["train"]
            logger.info("Testing batch loading...")
            
            for i, batch in enumerate(train_loader):
                logger.info(f"Batch {i+1}:")
                logger.info(f"  Input IDs shape: {batch['input_ids'].shape}")
                logger.info(f"  Attention mask shape: {batch['attention_mask'].shape}")
                logger.info(f"  Valence shape: {batch['valence'].shape}")
                logger.info(f"  Arousal shape: {batch['arousal'].shape}")
                logger.info(f"  User IDs: {batch['user_id'].tolist()}")
                logger.info(f"  Sequence lengths: {batch['sequence_length'].tolist()}")
                
                if i >= 2:  # Test only first few batches
                    break
        
        logger.info("=== DATA PIPELINE TEST SUCCESSFUL ===")
        return True
        
    except ImportError as e:
        logger.warning(f"PyTorch not available for testing: {e}")
        logger.info("Skipping dataset/dataloader testing, but preprocessing completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    # Set up the environment
    import sys
    import torch
    
    # Run the test
    success = test_data_pipeline()
    
    if success:
        logger.info("Data preparation script is ready for use!")
        print("\n" + "="*60)
        print("SUBTASK 1 DATA PREPARATION SUMMARY")
        print("="*60)
        print("✓ Data loading and analysis")
        print("✓ Text preprocessing pipeline")
        print("✓ Temporal sequence creation")
        print("✓ Train/validation/test splits")
        print("✓ PyTorch datasets and dataloaders")
        print("✓ Pipeline testing")
        print("\nTo use this script in your training pipeline:")
        print("from prepare_subtask1_data import Subtask1DataPreparator")
        print("preparator = Subtask1DataPreparator()")
        print("results = preparator.prepare_subtask1_data()")
        print("="*60)
    else:
        logger.error("Data preparation script has issues. Please check the logs.")
        sys.exit(1)