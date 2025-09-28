"""
Logging utilities for SemEval-2026 Task 2 project.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format the message
        formatted = super().format(record)

        # Add color if terminal supports it
        if sys.stderr.isatty():
            formatted = f"{color}{formatted}{reset}"

        return formatted


class ProgressHandler(logging.Handler):
    """Custom handler for progress tracking."""

    def __init__(self):
        super().__init__()
        self.progress_info = {}

    def emit(self, record: logging.LogRecord):
        """Handle progress log records."""
        if hasattr(record, 'progress_type'):
            self.progress_info[record.progress_type] = {
                'message': record.getMessage(),
                'timestamp': datetime.fromtimestamp(record.created)
            }


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    use_json: bool = False,
    use_colors: bool = True,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        log_file: Specific log file name
        use_json: Whether to use JSON formatting for file logs
        use_colors: Whether to use colored output for console
        experiment_name: Name of the experiment for log file naming

    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if use_colors:
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_dir or log_file:
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            if not log_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                exp_suffix = f"_{experiment_name}" if experiment_name else ""
                log_file = f"emotion_prediction{exp_suffix}_{timestamp}.log"

            log_file_path = log_dir / log_file
        else:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)

        if use_json:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file_path}")

    # Progress handler for tracking
    progress_handler = ProgressHandler()
    root_logger.addHandler(progress_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    return wrapper


def log_training_progress(
    epoch: int,
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log training progress information.

    Args:
        epoch: Current epoch
        step: Current step
        total_steps: Total steps
        loss: Current loss
        learning_rate: Current learning rate
        metrics: Additional metrics to log
    """
    logger = get_logger(__name__)

    progress_percent = (step / total_steps) * 100
    log_message = (
        f"Epoch {epoch} | Step {step}/{total_steps} ({progress_percent:.1f}%) | "
        f"Loss: {loss:.4f} | LR: {learning_rate:.2e}"
    )

    if metrics:
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        log_message += f" | {metrics_str}"

    # Log with progress tracking
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, log_message, (), None
    )
    record.progress_type = 'training'
    logger.handle(record)


def log_evaluation_results(
    dataset_name: str,
    metrics: Dict[str, float],
    epoch: Optional[int] = None
) -> None:
    """
    Log evaluation results.

    Args:
        dataset_name: Name of the evaluated dataset
        metrics: Evaluation metrics
        epoch: Current epoch (if applicable)
    """
    logger = get_logger(__name__)

    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    log_message = f"Evaluation on {dataset_name}{epoch_str} | {metrics_str}"

    # Log with evaluation tracking
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, log_message, (), None
    )
    record.progress_type = 'evaluation'
    record.extra_fields = {'dataset': dataset_name, 'metrics': metrics}
    if epoch is not None:
        record.extra_fields['epoch'] = epoch
    logger.handle(record)


def log_hyperparameters(hyperparams: Dict[str, Any]) -> None:
    """
    Log hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameters
    """
    logger = get_logger(__name__)

    logger.info("Hyperparameters:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")

    # Log with hyperparameter tracking
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, "Hyperparameters logged", (), None
    )
    record.progress_type = 'hyperparameters'
    record.extra_fields = {'hyperparameters': hyperparams}
    logger.handle(record)


def log_model_info(model, total_params: Optional[int] = None) -> None:
    """
    Log model information.

    Args:
        model: Model instance
        total_params: Total number of parameters
    """
    logger = get_logger(__name__)

    model_name = model.__class__.__name__
    if total_params is None:
        total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {model_name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Log with model tracking
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, f"Model info: {model_name}", (), None
    )
    record.progress_type = 'model_info'
    record.extra_fields = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    logger.handle(record)