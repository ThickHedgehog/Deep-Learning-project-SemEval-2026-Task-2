"""
Training script for SemEval-2026 Task 2 models
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import wandb
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data import create_dataloaders_from_config
from src.models import BertEmotionModel, TemporalEmotionModel
from src.training import Trainer
from src.evaluation import MetricsCalculator
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train emotion prediction models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    """Create model based on configuration."""
    model_config = config['model']
    model_type = model_config.get('type', 'bert')

    if model_type == 'bert':
        model = BertEmotionModel(
            model_name=model_config.get('base_model', 'bert-base-uncased'),
            num_emotions=model_config.get('num_emotions', 2),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            freeze_encoder=model_config.get('freeze_encoder', False),
            pooling_strategy=model_config.get('pooling_strategy', 'cls')
        )
    elif model_type == 'temporal':
        model = TemporalEmotionModel(
            base_encoder=model_config.get('base_encoder', 'bert-base-uncased'),
            temporal_model_type=model_config.get('temporal_model_type', 'lstm'),
            temporal_hidden_size=model_config.get('temporal_hidden_size', 256),
            num_temporal_layers=model_config.get('num_temporal_layers', 2),
            bidirectional=model_config.get('bidirectional', True),
            use_attention=model_config.get('use_attention', True),
            num_emotions=model_config.get('num_emotions', 2),
            dropout_rate=model_config.get('dropout_rate', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def setup_experiment(config: dict, args) -> str:
    """Setup experiment directory and logging."""
    # Create experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config['model'].get('type', 'bert')
        experiment_name = f"{model_type}_{timestamp}"

    # Create output directory
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(
        log_file=os.path.join(experiment_dir, 'train.log'),
        level=log_level
    )

    # Save config
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Configuration saved to: {config_path}")

    return experiment_dir


def setup_wandb(config: dict, experiment_dir: str, args):
    """Setup Weights & Biases logging."""
    if args.no_wandb:
        return None

    try:
        # Initialize wandb
        run = wandb.init(
            project=config.get('project', {}).get('name', 'semeval-2026-task2'),
            name=os.path.basename(experiment_dir),
            config=config,
            dir=experiment_dir,
            resume="allow" if args.resume else None
        )

        logger.info(f"Weights & Biases initialized: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        return None


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed = config.get('reproducibility', {}).get('seed', 42)
    set_seed(seed)

    # Setup experiment
    experiment_dir = setup_experiment(config, args)

    # Setup wandb
    wandb_run = setup_wandb(config, experiment_dir, args)

    try:
        logger.info("Starting training...")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Output directory: {experiment_dir}")

        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Model parameters: {model.count_parameters():,}")

        # Create data loaders
        logger.info("Creating data loaders...")
        data_loader = create_dataloaders_from_config(config)

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            config=config,
            output_dir=experiment_dir,
            wandb_run=wandb_run
        )

        # Load checkpoint if resuming
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Train model
        logger.info("Starting training...")
        trainer.train(data_loader)

        # Save final model
        final_model_path = os.path.join(experiment_dir, 'final_model.pt')
        trainer.save_model(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # Evaluate model
        logger.info("Evaluating model...")
        metrics_calculator = MetricsCalculator()

        # Get validation predictions
        val_predictions, val_targets = trainer.predict_dataloader(data_loader['val'])
        val_metrics = metrics_calculator.calculate_metrics(
            val_targets, val_predictions,
            temporal=config['model'].get('temporal_model_type') is not None
        )

        logger.info("Validation metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save metrics
        metrics_path = os.path.join(experiment_dir, 'final_metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, indent=2)

        if wandb_run:
            wandb_run.log({"final_" + k: v for k, v in val_metrics.items()})
            wandb_run.finish()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if wandb_run:
            wandb_run.finish(exit_code=1)
        raise

    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()