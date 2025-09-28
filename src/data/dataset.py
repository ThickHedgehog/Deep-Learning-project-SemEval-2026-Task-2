"""
Dataset classes for SemEval-2026 Task 2: Temporal Emotion Prediction
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """
    Base dataset class for emotion prediction from text.
    Handles valence and arousal prediction from ecological essays.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        text_column: str = "text",
        valence_column: str = "valence",
        arousal_column: str = "arousal",
        user_id_column: str = "user_id",
        timestamp_column: str = "timestamp"
    ):
        """
        Initialize the emotion dataset.

        Args:
            data: DataFrame or path to CSV file containing the data
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            text_column: Name of the text column
            valence_column: Name of the valence column
            arousal_column: Name of the arousal column
            user_id_column: Name of the user ID column
            timestamp_column: Name of the timestamp column
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.text_column = text_column
        self.valence_column = valence_column
        self.arousal_column = arousal_column
        self.user_id_column = user_id_column
        self.timestamp_column = timestamp_column

        # Validate required columns
        required_columns = [text_column, valence_column, arousal_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Initialized dataset with {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Dictionary containing tokenized text and emotion labels
        """
        row = self.data.iloc[idx]

        # Get text and ensure it's a string
        text = str(row[self.text_column])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Get emotion labels
        valence = float(row[self.valence_column])
        arousal = float(row[self.arousal_column])

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "valence": torch.tensor(valence, dtype=torch.float),
            "arousal": torch.tensor(arousal, dtype=torch.float),
            "emotions": torch.tensor([valence, arousal], dtype=torch.float)
        }

        # Add optional fields if available
        if self.user_id_column in self.data.columns:
            item["user_id"] = torch.tensor(row[self.user_id_column], dtype=torch.long)

        if self.timestamp_column in self.data.columns:
            item["timestamp"] = torch.tensor(row[self.timestamp_column], dtype=torch.float)

        return item


class TemporalEmotionDataset(Dataset):
    """
    Dataset class for temporal emotion prediction.
    Groups data by user and creates temporal sequences.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        sequence_length: int = 10,
        text_column: str = "text",
        valence_column: str = "valence",
        arousal_column: str = "arousal",
        user_id_column: str = "user_id",
        timestamp_column: str = "timestamp",
        overlap: bool = True
    ):
        """
        Initialize the temporal emotion dataset.

        Args:
            data: DataFrame or path to CSV file containing the data
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            sequence_length: Length of temporal sequences
            text_column: Name of the text column
            valence_column: Name of the valence column
            arousal_column: Name of the arousal column
            user_id_column: Name of the user ID column
            timestamp_column: Name of the timestamp column
            overlap: Whether to create overlapping sequences
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.sequence_length = sequence_length
        self.text_column = text_column
        self.valence_column = valence_column
        self.arousal_column = arousal_column
        self.user_id_column = user_id_column
        self.timestamp_column = timestamp_column
        self.overlap = overlap

        # Sort by user and timestamp
        self.data = self.data.sort_values([user_id_column, timestamp_column])

        # Create sequences
        self.sequences = self._create_sequences()

        logger.info(f"Created {len(self.sequences)} temporal sequences")

    def _create_sequences(self) -> List[Dict]:
        """Create temporal sequences from the data."""
        sequences = []

        for user_id in self.data[self.user_id_column].unique():
            user_data = self.data[self.data[self.user_id_column] == user_id]

            if len(user_data) < self.sequence_length:
                continue

            step = 1 if self.overlap else self.sequence_length

            for i in range(0, len(user_data) - self.sequence_length + 1, step):
                sequence_data = user_data.iloc[i:i + self.sequence_length]
                sequences.append({
                    "user_id": user_id,
                    "data": sequence_data,
                    "start_idx": i
                })

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a temporal sequence from the dataset.

        Args:
            idx: Index of the sequence

        Returns:
            Dictionary containing temporal sequence data
        """
        sequence = self.sequences[idx]
        sequence_data = sequence["data"]

        # Tokenize all texts in the sequence
        input_ids = []
        attention_masks = []
        valences = []
        arousals = []
        timestamps = []

        for _, row in sequence_data.iterrows():
            text = str(row[self.text_column])

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids.append(encoding["input_ids"].squeeze(0))
            attention_masks.append(encoding["attention_mask"].squeeze(0))
            valences.append(float(row[self.valence_column]))
            arousals.append(float(row[self.arousal_column]))
            timestamps.append(float(row[self.timestamp_column]))

        return {
            "input_ids": torch.stack(input_ids),  # [seq_len, max_length]
            "attention_mask": torch.stack(attention_masks),  # [seq_len, max_length]
            "valence": torch.tensor(valences, dtype=torch.float),  # [seq_len]
            "arousal": torch.tensor(arousals, dtype=torch.float),  # [seq_len]
            "emotions": torch.tensor(list(zip(valences, arousals)), dtype=torch.float),  # [seq_len, 2]
            "timestamps": torch.tensor(timestamps, dtype=torch.float),  # [seq_len]
            "user_id": torch.tensor(sequence["user_id"], dtype=torch.long),
            "sequence_length": torch.tensor(len(sequence_data), dtype=torch.long)
        }


class EvaluationDataset(EmotionDataset):
    """
    Dataset class for evaluation that preserves original data structure.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with additional metadata for evaluation.
        """
        item = super().__getitem__(idx)

        # Add original row data for evaluation
        row = self.data.iloc[idx]
        item["original_data"] = {
            "text": str(row[self.text_column]),
            "valence": float(row[self.valence_column]),
            "arousal": float(row[self.arousal_column])
        }

        if self.user_id_column in self.data.columns:
            item["original_data"]["user_id"] = row[self.user_id_column]

        if self.timestamp_column in self.data.columns:
            item["original_data"]["timestamp"] = row[self.timestamp_column]

        return item