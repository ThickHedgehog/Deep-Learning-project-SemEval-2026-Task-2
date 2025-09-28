"""
Submission formatting and Codabench integration for SemEval-2026 Task 2
"""

import pandas as pd
import numpy as np
import json
import zipfile
import os
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SubmissionFormatter:
    """
    Format predictions for SemEval-2026 Task 2 submission.
    """

    def __init__(self, output_format: str = "csv"):
        """
        Initialize submission formatter.

        Args:
            output_format: Output format ("csv", "json", "tsv")
        """
        self.output_format = output_format.lower()
        if self.output_format not in ["csv", "json", "tsv"]:
            raise ValueError(f"Unsupported output format: {output_format}")

    def format_predictions(
        self,
        predictions: np.ndarray,
        ids: Optional[List[Union[str, int]]] = None,
        user_ids: Optional[List[Union[str, int]]] = None,
        timestamps: Optional[List[Union[str, float]]] = None
    ) -> Dict:
        """
        Format predictions for submission.

        Args:
            predictions: Emotion predictions [n_samples, 2] (valence, arousal)
            ids: Sample IDs
            user_ids: User IDs (optional)
            timestamps: Timestamps (optional)

        Returns:
            Formatted predictions dictionary
        """
        n_samples = len(predictions)

        # Generate IDs if not provided
        if ids is None:
            ids = list(range(1, n_samples + 1))

        # Ensure predictions are numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        # Ensure predictions have correct shape
        if predictions.ndim == 1:
            if len(predictions) == 2:
                # Single prediction
                predictions = predictions.reshape(1, -1)
            else:
                # Multiple single-dimension predictions
                predictions = predictions.reshape(-1, 1)

        if predictions.shape[1] < 2:
            # Add arousal dimension if missing
            arousal = np.zeros((predictions.shape[0], 1))
            predictions = np.concatenate([predictions, arousal], axis=1)

        # Extract valence and arousal
        valence = predictions[:, 0]
        arousal = predictions[:, 1]

        # Create submission dictionary
        submission = {
            "id": ids,
            "valence": valence.tolist(),
            "arousal": arousal.tolist()
        }

        # Add optional fields
        if user_ids is not None:
            submission["user_id"] = user_ids

        if timestamps is not None:
            submission["timestamp"] = timestamps

        return submission

    def save_predictions(
        self,
        predictions: np.ndarray,
        output_path: str,
        ids: Optional[List[Union[str, int]]] = None,
        user_ids: Optional[List[Union[str, int]]] = None,
        timestamps: Optional[List[Union[str, float]]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save formatted predictions to file.

        Args:
            predictions: Emotion predictions
            output_path: Output file path
            ids: Sample IDs
            user_ids: User IDs (optional)
            timestamps: Timestamps (optional)
            metadata: Additional metadata

        Returns:
            Path to saved file
        """
        # Format predictions
        submission = self.format_predictions(
            predictions, ids, user_ids, timestamps
        )

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save based on format
        if self.output_format == "csv":
            self._save_csv(submission, output_path)
        elif self.output_format == "json":
            self._save_json(submission, output_path, metadata)
        elif self.output_format == "tsv":
            self._save_tsv(submission, output_path)

        logger.info(f"Predictions saved to: {output_path}")
        return output_path

    def _save_csv(self, submission: Dict, output_path: str):
        """Save as CSV format."""
        df = pd.DataFrame(submission)
        df.to_csv(output_path, index=False)

    def _save_json(self, submission: Dict, output_path: str, metadata: Optional[Dict] = None):
        """Save as JSON format."""
        output_data = {
            "predictions": submission,
            "metadata": metadata or {}
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def _save_tsv(self, submission: Dict, output_path: str):
        """Save as TSV format."""
        df = pd.DataFrame(submission)
        df.to_csv(output_path, sep='\t', index=False)

    def validate_submission(self, submission_path: str) -> bool:
        """
        Validate submission format.

        Args:
            submission_path: Path to submission file

        Returns:
            True if valid, False otherwise
        """
        try:
            if self.output_format == "csv":
                df = pd.read_csv(submission_path)
            elif self.output_format == "tsv":
                df = pd.read_csv(submission_path, sep='\t')
            elif self.output_format == "json":
                with open(submission_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data.get("predictions", data))

            # Check required columns
            required_columns = ["id", "valence", "arousal"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check data types and ranges
            if not pd.api.types.is_numeric_dtype(df['valence']):
                logger.error("Valence column must be numeric")
                return False

            if not pd.api.types.is_numeric_dtype(df['arousal']):
                logger.error("Arousal column must be numeric")
                return False

            # Check value ranges (typically -1 to 1 for emotions)
            if df['valence'].min() < -2 or df['valence'].max() > 2:
                logger.warning("Valence values outside typical range [-1, 1]")

            if df['arousal'].min() < -2 or df['arousal'].max() > 2:
                logger.warning("Arousal values outside typical range [-1, 1]")

            # Check for missing values
            if df[required_columns].isnull().any().any():
                logger.error("Missing values found in required columns")
                return False

            logger.info(f"Submission validation passed: {len(df)} predictions")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


class CodebenchSubmission:
    """
    Handle Codabench-specific submission formatting and packaging.
    """

    def __init__(self, team_name: str, model_name: str):
        """
        Initialize Codabench submission handler.

        Args:
            team_name: Name of the team
            model_name: Name/description of the model
        """
        self.team_name = team_name
        self.model_name = model_name

    def create_submission_package(
        self,
        predictions_path: str,
        output_dir: str,
        system_description: Optional[str] = None,
        code_files: Optional[List[str]] = None
    ) -> str:
        """
        Create complete submission package for Codabench.

        Args:
            predictions_path: Path to predictions file
            output_dir: Output directory for submission package
            system_description: Path to system description file
            code_files: List of code files to include

        Returns:
            Path to submission zip file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create submission metadata
        metadata = {
            "team_name": self.team_name,
            "model_name": self.model_name,
            "task": "SemEval-2026 Task 2",
            "description": "Predicting Variation in Emotional Valence and Arousal over Time"
        }

        # Create submission zip
        submission_zip_path = os.path.join(output_dir, f"{self.team_name}_submission.zip")

        with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add predictions
            zipf.write(predictions_path, "predictions.csv")

            # Add metadata
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            zipf.write(metadata_path, "metadata.json")

            # Add system description if provided
            if system_description and os.path.exists(system_description):
                zipf.write(system_description, "system_description.pdf")

            # Add code files if provided
            if code_files:
                code_dir = "code/"
                for code_file in code_files:
                    if os.path.exists(code_file):
                        # Preserve directory structure within code/
                        arcname = code_dir + os.path.relpath(code_file)
                        zipf.write(code_file, arcname)

        logger.info(f"Submission package created: {submission_zip_path}")
        return submission_zip_path

    def validate_submission_package(self, submission_zip_path: str) -> bool:
        """
        Validate submission package contents.

        Args:
            submission_zip_path: Path to submission zip file

        Returns:
            True if valid, False otherwise
        """
        try:
            with zipfile.ZipFile(submission_zip_path, 'r') as zipf:
                file_list = zipf.namelist()

                # Check required files
                if "predictions.csv" not in file_list:
                    logger.error("Missing predictions.csv in submission package")
                    return False

                if "metadata.json" not in file_list:
                    logger.error("Missing metadata.json in submission package")
                    return False

                # Validate predictions file
                zipf.extract("predictions.csv", "/tmp/")
                formatter = SubmissionFormatter("csv")
                if not formatter.validate_submission("/tmp/predictions.csv"):
                    return False

                # Validate metadata
                zipf.extract("metadata.json", "/tmp/")
                with open("/tmp/metadata.json", 'r') as f:
                    metadata = json.load(f)

                required_metadata = ["team_name", "model_name", "task"]
                missing_metadata = [key for key in required_metadata if key not in metadata]

                if missing_metadata:
                    logger.error(f"Missing metadata fields: {missing_metadata}")
                    return False

                logger.info("Submission package validation passed")
                return True

        except Exception as e:
            logger.error(f"Package validation failed: {e}")
            return False

    def create_system_description_template(self, output_path: str):
        """
        Create a template for the system description document.

        Args:
            output_path: Path to save the template
        """
        template = f"""# System Description: {self.model_name}

## Team Information
- **Team Name**: {self.team_name}
- **Task**: SemEval-2026 Task 2 - Predicting Variation in Emotional Valence and Arousal over Time
- **Contact**: [Your Email]

## System Overview
[Provide a brief overview of your system approach]

## Model Architecture
[Describe your model architecture in detail]

### Base Model
- Model type: [e.g., BERT, RoBERTa, Custom]
- Pre-trained weights: [e.g., bert-base-uncased]
- Fine-tuning approach: [Describe your fine-tuning strategy]

### Temporal Modeling
- Temporal architecture: [e.g., LSTM, GRU, Transformer]
- Sequence length: [Length of temporal sequences]
- Attention mechanism: [If applicable]

## Data Processing
[Describe your data preprocessing pipeline]

### Text Preprocessing
- Cleaning steps
- Tokenization approach
- Feature engineering

### Temporal Processing
- Time window handling
- Sequence creation
- Data augmentation (if any)

## Training Details
[Provide training configuration details]

### Hyperparameters
- Learning rate: [value]
- Batch size: [value]
- Number of epochs: [value]
- Optimizer: [e.g., AdamW]
- Loss function: [e.g., MSE, MAE]

### Regularization
- Dropout rate: [value]
- Weight decay: [value]
- Early stopping: [if used]

## Evaluation Results
[Present your experimental results]

### Performance Metrics
- Validation MSE: [value]
- Validation MAE: [value]
- Pearson Correlation: [value]
- Other relevant metrics

### Ablation Studies
[If conducted, describe ablation studies]

## Discussion
[Discuss your findings, challenges, and insights]

### Key Contributions
- [List your main contributions]

### Limitations
- [Discuss limitations of your approach]

### Future Work
- [Suggest improvements for future work]

## References
[Include relevant references]

## Appendix
[Additional technical details, code snippets, etc.]
"""

        with open(output_path, 'w') as f:
            f.write(template)

        logger.info(f"System description template created: {output_path}")


def main():
    """Example usage of submission formatting."""
    # Example predictions
    predictions = np.random.uniform(-1, 1, (100, 2))  # 100 samples, valence and arousal
    ids = list(range(1, 101))

    # Format and save predictions
    formatter = SubmissionFormatter("csv")
    output_path = "submissions/example_predictions.csv"
    formatter.save_predictions(predictions, output_path, ids)

    # Validate submission
    is_valid = formatter.validate_submission(output_path)
    print(f"Submission valid: {is_valid}")

    # Create Codabench submission package
    codabench = CodebenchSubmission("TeamName", "EmotionPredictor v1.0")
    submission_zip = codabench.create_submission_package(
        output_path,
        "submissions/",
        system_description=None,  # Add path to system description if available
        code_files=["src/models/", "scripts/train.py"]  # Add relevant code files
    )

    # Validate package
    package_valid = codabench.validate_submission_package(submission_zip)
    print(f"Package valid: {package_valid}")


if __name__ == "__main__":
    main()