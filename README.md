# Medical Image Classification (Chest X-Ray): Pneumonia Detection

A recruiter-ready deep learning project that classifies chest X-ray images into **NORMAL** vs **PNEUMONIA** using TensorFlow/Keras.

This repository demonstrates:
- end-to-end ML workflow (data loading, training, evaluation, experiment tracking)
- model comparison and regularization analysis
- practical CPU-first training setup
- reproducible outputs (metrics JSON, plots, saved models)

## Project Overview

- **Goal:** Binary classification of chest X-ray images (`NORMAL`, `PNEUMONIA`)
- **Framework:** TensorFlow / Keras
- **Primary Setup:** CPU training (no GPU required)
- **Input Size:** `160x160`
- **Default Batch Size:** `16`
- **Default LR:** `0.0005` (Adam)
- **Early Stopping:** `patience=5`

## Dataset

This project uses public Kaggle datasets:
- Chest X-Ray Pneumonia (Mooney): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Chest X-Ray COVID19/Pneumonia: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

Current binary experiments are run on `dataset/chest_xray` with structure:

```text
dataset/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

### Split counts (current run)
- **Train:** 5216 (NORMAL: 1341, PNEUMONIA: 3875)
- **Val:** 16 (NORMAL: 8, PNEUMONIA: 8)
- **Test:** 624 (NORMAL: 234, PNEUMONIA: 390)

## Results (Custom CNN Experiments)

| Run | Augmentation | Dropout | L2 | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `custom_baseline` | No | 0.0 | 0 | **0.8542** | 0.8637 | **0.9103** | **0.8864** | **0.9163** |
| `custom_aug_only` | Yes | 0.0 | 0 | 0.7804 | 0.8754 | 0.7564 | 0.8116 | 0.8736 |
| `custom_aug_reg` | Yes | 0.6 | 1e-4 | 0.8045 | 0.8918 | 0.7821 | 0.8333 | 0.8854 |
| `custom_aug_reg_d04` | Yes | 0.4 | 1e-4 | 0.7869 | **0.9327** | 0.7103 | 0.8064 | 0.8871 |

### Key takeaway
For this dataset/setup, the baseline custom CNN currently provides the strongest test-set balance and highest AUC among tested custom variants.

## Visual Outputs

Baseline outputs are available in `outputs/custom_baseline/`, including:
- `train_val_accuracy.png`
- `roc_curve.png`
- `confusion_matrix.png`
- `training_curves.png`

## Repository Structure

```text
train_custom.py        # Custom CNN training experiments
train_mobilenet.py     # MobileNetV2 transfer learning pipeline
evaluate.py            # Evaluate saved model on test split
utils.py               # Shared data/model/train/eval helpers
download_datasets.py   # KaggleHub dataset download helper
outputs/               # Saved models, metrics, plots per run
```

## Quick Start

### 1) Create and activate environment (Windows)

```powershell
python -m venv pneumonia_env
.\pneumonia_env\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install tensorflow numpy matplotlib scikit-learn pandas kagglehub
```

### 3) Train custom CNN

```powershell
python train_custom.py --dataset-dir dataset/chest_xray --output-dir outputs
```

### 4) Train MobileNetV2

```powershell
python train_mobilenet.py --dataset-dir dataset/chest_xray --output-dir outputs
```

### 5) Evaluate a saved model

```powershell
python evaluate.py --dataset-dir dataset/chest_xray --model-path outputs/custom_baseline/best_model.keras --output-dir outputs --run-name eval_baseline
```

## Recruiter Notes

This project highlights practical ML engineering skills:
- experiment design and ablation thinking (baseline vs augmentation/regularization)
- metric-driven decision making (accuracy, precision, recall, F1, AUC, confusion matrix)
- clean modular code and reproducible artifacts
- ability to optimize for constrained hardware (CPU-only workflow)

## Next Improvements

- expand/stratify validation split for more stable model selection
- threshold tuning for precision-recall tradeoff
- class-balanced training + calibration checks
- full MobileNetV2 benchmark and side-by-side comparison report
