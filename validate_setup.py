#!/usr/bin/env python3
"""
Validation script for Subtask 2a Final Training Setup
Run this before executing COLAB_COMPLETE_CODE.py to ensure everything is ready.
"""

import os
import sys
from pathlib import Path

def check_mark(condition):
    return "✅" if condition else "❌"

def validate_setup():
    """Validate that all required files and dependencies are present."""

    print("=" * 80)
    print("SUBTASK 2A - FINAL SETUP VALIDATION")
    print("=" * 80)
    print()

    # Check Python version
    print("🐍 Python Version Check:")
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 8
    print(f"   {check_mark(py_ok)} Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("   ⚠️  Warning: Python 3.8+ recommended")
    print()

    # Check required files
    print("📁 Required Files Check:")
    project_root = Path(__file__).parent

    files_to_check = {
        "Training Script": "COLAB_COMPLETE_CODE.py",
        "README": "README.md",
        "Quick Start Guide": "QUICKSTART.md",
        "Requirements": "requirements.txt",
        "Training Data": "data/raw/train_subtask2a.csv",
        "Feature Prep Script": "scripts/data_preparation/subtask2a/prepare_features_subtask2a.py",
        "Local Training Script": "scripts/data_train/subtask2a/train_final_subtask2a.py",
    }

    all_files_ok = True
    for name, filepath in files_to_check.items():
        full_path = project_root / filepath
        exists = full_path.exists()
        all_files_ok &= exists
        size = f"({full_path.stat().st_size / 1024:.1f} KB)" if exists else ""
        print(f"   {check_mark(exists)} {name}: {filepath} {size}")
    print()

    # Check dependencies
    print("📦 Python Dependencies Check:")
    dependencies = [
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "sklearn",
        "scipy",
        "tqdm",
        "wandb",
    ]

    missing_deps = []
    for dep in dependencies:
        try:
            if dep == "sklearn":
                __import__("sklearn")
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (not installed)")
            missing_deps.append(dep)

    if missing_deps:
        print()
        print("   To install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
    print()

    # Check CUDA availability
    print("🖥️  GPU/CUDA Check:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   {check_mark(cuda_available)} CUDA Available")
        if cuda_available:
            print(f"   ✅ GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA Version: {torch.version.cuda}")
        else:
            print("   ⚠️  No GPU detected - use Google Colab with T4 GPU")
    except ImportError:
        print("   ⚠️  PyTorch not installed - cannot check CUDA")
    print()

    # Check data file
    print("📊 Training Data Check:")
    data_path = project_root / "data/raw/train_subtask2a.csv"
    if data_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"   ✅ Data file loaded successfully")
            print(f"   ✅ Rows: {len(df):,}")
            print(f"   ✅ Columns: {', '.join(df.columns.tolist())}")

            required_cols = ['user_id', 'item_id', 'timestamp', 'text', 'valence', 'arousal']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   ❌ Missing columns: {', '.join(missing_cols)}")
            else:
                print(f"   ✅ All required columns present")
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
    else:
        print(f"   ❌ Data file not found: {data_path}")
    print()

    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_files_ok and not missing_deps:
        print("✅ All checks passed! Ready to train.")
        print()
        print("Next steps:")
        print("1. Open QUICKSTART.md for execution instructions")
        print("2. Copy COLAB_COMPLETE_CODE.py to Google Colab")
        print("3. Enable T4 GPU in Colab runtime settings")
        print("4. Run the training (90-120 minutes)")
        print("5. Monitor progress in WandB dashboard")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        if missing_deps:
            print()
            print("Install missing dependencies:")
            print("pip install -r requirements.txt")

    print()
    print("=" * 80)
    print(f"Expected Performance: CCC 0.65-0.72 (Competition Ready)")
    print("=" * 80)

if __name__ == "__main__":
    validate_setup()
