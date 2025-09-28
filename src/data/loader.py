"""
Data loading utilities for SemEval-2026 Task 2
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader, DistributedSampler
import pandas as pd
from typing import Dict, Optional, Union, List
import logging
from .dataset import EmotionDataset, TemporalEmotionDataset, EvaluationDataset
from .preprocessor import TextPreprocessor, TemporalPreprocessor

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for emotion prediction tasks.
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        distributed: bool = False
    ):
        """
        Initialize the data loader.

        Args:
            batch_size: Batch size for training
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            distributed: Whether to use distributed training
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.distributed = distributed

    def create_dataloader(
        self,
        dataset: Union[EmotionDataset, TemporalEmotionDataset],
        shuffle: bool = True,
        sampler: Optional[torch.utils.data.Sampler] = None
    ) -> TorchDataLoader:
        """
        Create a PyTorch DataLoader.

        Args:
            dataset: Dataset instance
            shuffle: Whether to shuffle the data
            sampler: Custom sampler to use

        Returns:
            PyTorch DataLoader
        """
        if self.distributed and sampler is None:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Disable shuffle when using DistributedSampler

        return TorchDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching.

        Args:
            batch: List of samples

        Returns:
            Batched data
        """
        # Handle regular emotion dataset
        if "sequence_length" not in batch[0]:
            return self._collate_emotion_batch(batch)
        # Handle temporal emotion dataset
        else:
            return self._collate_temporal_batch(batch)

    def _collate_emotion_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for regular emotion dataset."""
        collated = {}

        # Standard fields
        for key in ["input_ids", "attention_mask", "valence", "arousal", "emotions"]:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])

        # Optional fields
        for key in ["user_id", "timestamp"]:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])

        # Handle original data for evaluation dataset
        if "original_data" in batch[0]:
            collated["original_data"] = [item["original_data"] for item in batch]

        return collated

    def _collate_temporal_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for temporal emotion dataset."""
        collated = {}

        # Get maximum sequence length in the batch
        max_seq_len = max(item["sequence_length"].item() for item in batch)

        # Pad sequences to the same length
        for key in ["input_ids", "attention_mask", "valence", "arousal", "emotions", "timestamps"]:
            if key in batch[0]:
                padded_tensors = []
                for item in batch:
                    tensor = item[key]
                    seq_len = item["sequence_length"].item()

                    if seq_len < max_seq_len:
                        # Calculate padding dimensions
                        if tensor.dim() == 2:  # [seq_len, feature_dim]
                            padding = (0, 0, 0, max_seq_len - seq_len)
                        elif tensor.dim() == 3:  # [seq_len, max_length, feature_dim]
                            padding = (0, 0, 0, 0, 0, max_seq_len - seq_len)
                        else:  # [seq_len]
                            padding = (0, max_seq_len - seq_len)

                        padded_tensor = torch.nn.functional.pad(tensor, padding, value=0)
                    else:
                        padded_tensor = tensor

                    padded_tensors.append(padded_tensor)

                collated[key] = torch.stack(padded_tensors)

        # Handle non-padded fields
        for key in ["user_id", "sequence_length"]:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])

        # Create attention masks for sequences
        seq_lengths = collated["sequence_length"]
        batch_size = len(seq_lengths)
        sequence_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

        for i, seq_len in enumerate(seq_lengths):
            sequence_mask[i, :seq_len] = True

        collated["sequence_mask"] = sequence_mask

        return collated

    def load_data_from_file(
        self,
        file_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        text_column: str = "text",
        valence_column: str = "valence",
        arousal_column: str = "arousal",
        user_id_column: str = "user_id",
        timestamp_column: str = "timestamp",
        dataset_type: str = "emotion",
        **kwargs
    ) -> Union[EmotionDataset, TemporalEmotionDataset]:
        """
        Load dataset from file.

        Args:
            file_path: Path to the data file
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            text_column: Name of the text column
            valence_column: Name of the valence column
            arousal_column: Name of the arousal column
            user_id_column: Name of the user ID column
            timestamp_column: Name of the timestamp column
            dataset_type: Type of dataset ("emotion", "temporal", "evaluation")
            **kwargs: Additional arguments for dataset

        Returns:
            Dataset instance
        """
        if dataset_type == "emotion":
            return EmotionDataset(
                data=file_path,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                text_column=text_column,
                valence_column=valence_column,
                arousal_column=arousal_column,
                user_id_column=user_id_column,
                timestamp_column=timestamp_column
            )
        elif dataset_type == "temporal":
            return TemporalEmotionDataset(
                data=file_path,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                text_column=text_column,
                valence_column=valence_column,
                arousal_column=arousal_column,
                user_id_column=user_id_column,
                timestamp_column=timestamp_column,
                **kwargs
            )
        elif dataset_type == "evaluation":
            return EvaluationDataset(
                data=file_path,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                text_column=text_column,
                valence_column=valence_column,
                arousal_column=arousal_column,
                user_id_column=user_id_column,
                timestamp_column=timestamp_column
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


class TemporalDataLoader(DataLoader):
    """
    Specialized data loader for temporal emotion data.
    """

    def __init__(
        self,
        sequence_length: int = 10,
        overlap: bool = True,
        temporal_batch_strategy: str = "pad",  # "pad" or "pack"
        **kwargs
    ):
        """
        Initialize the temporal data loader.

        Args:
            sequence_length: Length of temporal sequences
            overlap: Whether to create overlapping sequences
            temporal_batch_strategy: Strategy for batching temporal sequences
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.temporal_batch_strategy = temporal_batch_strategy

    def create_temporal_dataloaders(
        self,
        train_data: Union[str, pd.DataFrame],
        val_data: Union[str, pd.DataFrame],
        test_data: Optional[Union[str, pd.DataFrame]] = None,
        tokenizer_name: str = "bert-base-uncased",
        **kwargs
    ) -> Dict[str, TorchDataLoader]:
        """
        Create train, validation, and test dataloaders for temporal data.

        Args:
            train_data: Training data file path or DataFrame
            val_data: Validation data file path or DataFrame
            test_data: Test data file path or DataFrame (optional)
            tokenizer_name: Name of the tokenizer to use
            **kwargs: Additional arguments for dataset creation

        Returns:
            Dictionary containing dataloaders
        """
        dataloaders = {}

        # Create training dataloader
        train_dataset = TemporalEmotionDataset(
            data=train_data,
            tokenizer_name=tokenizer_name,
            sequence_length=self.sequence_length,
            overlap=self.overlap,
            **kwargs
        )
        dataloaders["train"] = self.create_dataloader(train_dataset, shuffle=True)

        # Create validation dataloader
        val_dataset = TemporalEmotionDataset(
            data=val_data,
            tokenizer_name=tokenizer_name,
            sequence_length=self.sequence_length,
            overlap=False,  # No overlap for validation
            **kwargs
        )
        dataloaders["val"] = self.create_dataloader(val_dataset, shuffle=False)

        # Create test dataloader if provided
        if test_data is not None:
            test_dataset = TemporalEmotionDataset(
                data=test_data,
                tokenizer_name=tokenizer_name,
                sequence_length=self.sequence_length,
                overlap=False,  # No overlap for testing
                **kwargs
            )
            dataloaders["test"] = self.create_dataloader(test_dataset, shuffle=False)

        logger.info(f"Created temporal dataloaders: train={len(dataloaders['train'])}, val={len(dataloaders['val'])}")
        if "test" in dataloaders:
            logger.info(f"Test dataloader: {len(dataloaders['test'])} batches")

        return dataloaders


def create_dataloaders_from_config(config: Dict) -> Dict[str, TorchDataLoader]:
    """
    Create dataloaders from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary containing dataloaders
    """
    data_config = config["data"]
    training_config = config["training"]

    # Initialize preprocessors
    text_preprocessor = TextPreprocessor(
        lowercase=data_config.get("text_cleaning", {}).get("lowercase", False),
        remove_html=data_config.get("text_cleaning", {}).get("remove_html", True),
        remove_urls=data_config.get("text_cleaning", {}).get("remove_urls", True),
        normalize_whitespace=data_config.get("text_cleaning", {}).get("normalize_whitespace", True),
        min_length=data_config.get("min_sequence_length", 10),
        max_length=data_config.get("max_sequence_length", 512)
    )

    temporal_preprocessor = TemporalPreprocessor(
        text_preprocessor=text_preprocessor,
        min_sequences_per_user=5
    )

    # Create data loader
    if config.get("model", {}).get("temporal_model_type"):
        # Temporal data loader
        loader = TemporalDataLoader(
            batch_size=training_config["batch_size"],
            sequence_length=config["model"].get("temporal_window_size", 10),
            num_workers=config.get("hardware", {}).get("num_workers", 4),
            pin_memory=config.get("hardware", {}).get("pin_memory", True)
        )
    else:
        # Regular data loader
        loader = DataLoader(
            batch_size=training_config["batch_size"],
            num_workers=config.get("hardware", {}).get("num_workers", 4),
            pin_memory=config.get("hardware", {}).get("pin_memory", True)
        )

    return loader