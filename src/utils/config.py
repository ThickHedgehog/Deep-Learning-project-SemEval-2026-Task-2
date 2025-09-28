"""
Configuration management for SemEval-2026 Task 2 project.
"""

import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "bert_emotion"
    base_model: str = "bert-base-uncased"
    num_emotions: int = 2
    hidden_size: int = 768
    dropout_rate: float = 0.1
    max_length: int = 512
    freeze_encoder: bool = False
    output_range: List[float] = None

    def __post_init__(self):
        if self.output_range is None:
            self.output_range = [-2.0, 2.0]


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "data/raw"
    processed_path: str = "data/processed"
    train_file: str = "train_subtask2b.csv"
    test_file: str = "test_subtask2b.csv"
    val_split: float = 0.2
    random_seed: int = 42
    max_sequence_length: int = 512
    min_text_length: int = 10
    user_split: bool = True
    temporal_window: int = 5


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    early_stopping_patience: int = 3
    scheduler_type: str = "linear"
    optimizer_type: str = "adamw"


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str = "emotion_prediction"
    output_dir: str = "experiments/results"
    logging_dir: str = "experiments/logs"
    checkpoint_dir: str = "experiments/checkpoints"
    wandb_project: str = "semeval-2026-task2"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True
    use_mlflow: bool = False
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    experiment: ExperimentConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )

    def save(self, path: str) -> None:
        """Save config to file."""
        save_config(self, path)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from file."""
        return load_config(path)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using default config.")
        return Config()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        config = Config.from_dict(config_dict)
        logger.info(f"Successfully loaded config from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        logger.info("Using default config instead.")
        return Config()


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Config object to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = config.to_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        logger.info(f"Successfully saved config to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {str(e)}")
        raise


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge base config with override values.

    Args:
        base_config: Base configuration
        override_config: Override values

    Returns:
        Merged configuration
    """
    base_dict = base_config.to_dict()

    def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    merged_dict = deep_update(base_dict, override_config)
    return Config.from_dict(merged_dict)


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate model config
    if config.model.num_emotions < 1:
        errors.append("num_emotions must be >= 1")

    if config.model.hidden_size < 1:
        errors.append("hidden_size must be >= 1")

    if not (0 <= config.model.dropout_rate <= 1):
        errors.append("dropout_rate must be between 0 and 1")

    # Validate data config
    if config.data.val_split < 0 or config.data.val_split >= 1:
        errors.append("val_split must be between 0 and 1")

    if config.data.max_sequence_length < 1:
        errors.append("max_sequence_length must be >= 1")

    # Validate training config
    if config.training.batch_size < 1:
        errors.append("batch_size must be >= 1")

    if config.training.learning_rate <= 0:
        errors.append("learning_rate must be > 0")

    if config.training.num_epochs < 1:
        errors.append("num_epochs must be >= 1")

    return errors