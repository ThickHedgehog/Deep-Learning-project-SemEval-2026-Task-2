# Archive - Trial and Error Process

This folder contains code and documentation from the iterative development process that led to the final v3.0 ensemble solution.

## Contents

### Training Scripts (Trial Versions)
- **COLAB_OPTIMIZED_v3.1.py** - v3.1 attempt (not tested in practice)
- **COLAB_FINAL_PERFECT_v3.2.py** - v3.2 catastrophic failure (CCC 0.2883)
- **COLAB_FINAL_v3.3_MINIMAL.py** - v3.3 tested but below target (CCC 0.5053)

### Documentation
- **V3.1_IMPROVEMENTS.md** - v3.1 planned improvements documentation
- **V3.3_SUMMARY.md** - v3.3 summary and expectations
- **V3.3_QUICKSTART.md** - v3.3 quick start guide
- **V3.3_COMPLETION_SUMMARY.md** - v3.3 completion summary
- **FINAL_VERSION_COMPARISON.md** - Version comparison (superseded by FINAL_COMPREHENSIVE_ANALYSIS.md)
- **EXECUTION_CHECKLIST.md** - Old execution checklist

## Why These Files Are Archived

These files represent the learning process and experimentation that led to understanding:
- What works: User embeddings (64 dim), LSTM (256 hidden), Dropout (0.3), Arousal CCC (70%)
- What doesn't work: Removing user embeddings (-0.226 CCC), Arousal CCC 75% (backfires)

## Final Solution

The final solution is in the main directory:
- **ENSEMBLE_v3.0_COMPLETE.py** - Production-ready ensemble training code
- **ENSEMBLE_PREDICTION.py** - Ensemble prediction code
- **ENSEMBLE_GUIDE.md** - Complete execution guide
- **FINAL_COMPREHENSIVE_ANALYSIS.md** - Complete analysis of all versions

**Status**: v3.0 Ensemble is the recommended final solution (Expected CCC: 0.530-0.550, 85% success probability)

---

**Last Updated**: 2025-11-14
**Status**: Archived for reference
