"""
Input/Output utilities for SemEval-2026 Task 2 project.
"""

import json
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def save_json(data: Any, file_path: str, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        logger.debug(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        raise


def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data to pickle file.

    Args:
        data: Data to save
        file_path: Path to save file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Successfully saved pickle to {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle to {file_path}: {str(e)}")
        raise


def load_pickle(file_path: str) -> Any:
    """
    Load data from pickle file.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.debug(f"Successfully loaded pickle from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {str(e)}")
        raise


def save_csv(data: Union[pd.DataFrame, List[Dict]], file_path: str, **kwargs) -> None:
    """
    Save data to CSV file.

    Args:
        data: Data to save (DataFrame or list of dicts)
        file_path: Path to save file
        **kwargs: Additional arguments for pandas.to_csv
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        df.to_csv(file_path, index=False, **kwargs)
        logger.debug(f"Successfully saved CSV to {file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV to {file_path}: {str(e)}")
        raise


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pandas.read_csv

    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.debug(f"Successfully loaded CSV from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV from {file_path}: {str(e)}")
        raise


def save_numpy(data: np.ndarray, file_path: str) -> None:
    """
    Save numpy array to file.

    Args:
        data: Numpy array to save
        file_path: Path to save file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.save(file_path, data)
        logger.debug(f"Successfully saved numpy array to {file_path}")
    except Exception as e:
        logger.error(f"Error saving numpy array to {file_path}: {str(e)}")
        raise


def load_numpy(file_path: str) -> np.ndarray:
    """
    Load numpy array from file.

    Args:
        file_path: Path to numpy file

    Returns:
        Loaded numpy array
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Numpy file not found: {file_path}")

    try:
        data = np.load(file_path)
        logger.debug(f"Successfully loaded numpy array from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading numpy array from {file_path}: {str(e)}")
        raise


def save_torch_model(model: torch.nn.Module, file_path: str, save_state_dict: bool = True) -> None:
    """
    Save PyTorch model to file.

    Args:
        model: PyTorch model to save
        file_path: Path to save file
        save_state_dict: Whether to save only state_dict (recommended) or entire model
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if save_state_dict:
            torch.save(model.state_dict(), file_path)
        else:
            torch.save(model, file_path)
        logger.info(f"Successfully saved model to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {str(e)}")
        raise


def load_torch_model(model: torch.nn.Module, file_path: str, map_location: str = 'cpu') -> torch.nn.Module:
    """
    Load PyTorch model from file.

    Args:
        model: Model instance to load state_dict into
        file_path: Path to model file
        map_location: Device to map tensors to

    Returns:
        Loaded model
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    try:
        state_dict = torch.load(file_path, map_location=map_location)
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded model from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {str(e)}")
        raise


def save_checkpoint(
    checkpoint_data: Dict[str, Any],
    file_path: str,
    is_best: bool = False,
    keep_last_n: int = 5
) -> None:
    """
    Save training checkpoint.

    Args:
        checkpoint_data: Dictionary containing checkpoint data
        file_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        keep_last_n: Number of recent checkpoints to keep
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save checkpoint
        torch.save(checkpoint_data, file_path)
        logger.info(f"Saved checkpoint to {file_path}")

        # Save best model copy if this is the best
        if is_best:
            best_path = file_path.parent / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Saved best model to {best_path}")

        # Clean up old checkpoints
        if keep_last_n > 0:
            checkpoint_pattern = f"{file_path.stem}_epoch_*.pt"
            existing_checkpoints = sorted(
                file_path.parent.glob(checkpoint_pattern),
                key=lambda x: x.stat().st_mtime
            )

            if len(existing_checkpoints) > keep_last_n:
                for old_checkpoint in existing_checkpoints[:-keep_last_n]:
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    except Exception as e:
        logger.error(f"Error saving checkpoint to {file_path}: {str(e)}")
        raise


def load_checkpoint(file_path: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        file_path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        Loaded checkpoint data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    try:
        checkpoint = torch.load(file_path, map_location=map_location)
        logger.info(f"Successfully loaded checkpoint from {file_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint from {file_path}: {str(e)}")
        raise


def create_submission_file(
    predictions: Union[List[Dict], pd.DataFrame],
    file_path: str,
    format_type: str = 'csv'
) -> None:
    """
    Create submission file for competition.

    Args:
        predictions: Predictions data
        file_path: Path to save submission file
        format_type: Format type ('csv', 'json', 'tsv')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(predictions, list):
            df = pd.DataFrame(predictions)
        else:
            df = predictions

        if format_type.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format_type.lower() == 'tsv':
            df.to_csv(file_path, sep='\t', index=False)
        elif format_type.lower() == 'json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        logger.info(f"Successfully created submission file: {file_path}")

    except Exception as e:
        logger.error(f"Error creating submission file {file_path}: {str(e)}")
        raise


def ensure_dir_exists(dir_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        dir_path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    if file_path.exists():
        return file_path.stat().st_size
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


def list_files_recursive(directory: str, pattern: str = "*") -> List[Path]:
    """
    List files recursively in directory.

    Args:
        directory: Directory to search
        pattern: File pattern to match

    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return list(directory.rglob(pattern))