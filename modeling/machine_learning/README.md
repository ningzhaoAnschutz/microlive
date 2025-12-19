# Machine Learning for Spot Detection

This folder contains the CNN-based spot detection system used for colocalization analysis in MicroLive.

## Overview

The CNN model classifies image crops as containing a fluorescent spot or not. It is primarily used for:

- **Colocalization analysis**: Detecting if a spot in one channel has a corresponding spot in another channel
- **Spot validation**: Filtering false positive detections from tracking

## Folder Structure

```
machine_learning/
│
├── README.md                          # This file
│
├── spot_detection_cnn.pth             # ⭐ DEPLOYED MODEL (used by MicroLive GUI)
│
├── Generate_Training_Crops.ipynb      # Step 1: Extract crops from raw data
├── Human_Labeling_Tool.ipynb          # Step 2: Label crops interactively
├── ML_Tranining_Validation.ipynb      # Step 3: Train and evaluate models
│
├── training/                          # Training data and experimental models
│   ├── training_crops_real_data/          # 2000 crops from real microscopy
│   ├── training_crops_simulated_data/     # 6000 synthetic spots
│   ├── training_crops_human_selection/    # 120 consensus-labeled crops
│   ├── human_selection_1.npy              # Annotator 1 labels
│   ├── human_selection_2.npy              # Annotator 2 labels
│   ├── human_selection_3.npy              # Annotator 3 labels
│   ├── particle_detection_cnn_real_data.pth
│   ├── particle_detection_cnn_simulated_data.pth
│   ├── training_losses_*.npy              # Training history
│   └── validation_losses_*.npy
│
├── dev/                               # Legacy/archived files (can be deleted)
│
└── cnn_visualization.png              # Network architecture diagram
```

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Generate_Training_Crops.ipynb                           │
│     • Load raw .lif microscopy data                         │
│     • Run tracking to detect spots                          │
│     • Extract crops around each detected spot               │
│     • Output: training/training_crops_*/                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Human_Labeling_Tool.ipynb                               │
│     • Interactive widget for labeling crops                 │
│     • Multiple annotators label independently               │
│     • Consensus: only use crops where all agree             │
│     • Output: training/human_selection_*.npy                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ML_Tranining_Validation.ipynb                           │
│     • Create consensus labels (all annotators agree)        │
│     • Train CNN on labeled crops                            │
│     • Evaluate: accuracy, precision, recall, F1             │
│     • Output: spot_detection_cnn.pth                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MicroLive GUI                                              │
│     • Loads spot_detection_cnn.pth at startup               │
│     • Uses CNN for colocalization analysis                  │
└─────────────────────────────────────────────────────────────┘
```

## Core Module: `src/ML_SpotDetection.py`

The ML backend is implemented in `/src/ML_SpotDetection.py`. This module contains:

### Classes

| Class | Description |
|-------|-------------|
| `ParticleDetectionCNN` | CNN architecture for spot classification |
| `ParticleDataset` | PyTorch Dataset class for loading training crops |

### Functions

| Function | Description |
|----------|-------------|
| `load_model(model, path)` | Load trained weights from .pth file |
| `save_model(model, path)` | Save model weights to .pth file |
| `predict_crops(model, crops, threshold)` | Run inference on a list of image crops |
| `run_network(image_dir, num_epochs, ...)` | Train the CNN from scratch |
| `validate(model, loader, criterion, device)` | Evaluate model on validation set |

### CNN Architecture

```
Input: 64×64 grayscale image
    │
    ▼
Conv2D (1→32 channels, 3×3 kernel)
    │
MaxPool2D (2×2)
    │
    ▼
Conv2D (32→64 channels, 3×3 kernel)
    │
MaxPool2D (2×2)
    │
    ▼
Flatten (64×16×16 = 16,384)
    │
    ▼
Linear (16,384 → 128)
    │
    ▼
Linear (128 → 1)
    │
    ▼
Sigmoid → Probability [0, 1]
```

## How the GUI Uses the Model

When MicroLive starts, it loads the CNN in `src/imports.py`:

```python
import ML_SpotDetection as ML

# Path to deployed model
ML_folder = src_dir.parents[0] / 'modeling' / 'machine_learning'
model_ML = ML.ParticleDetectionCNN()
model_path = ML_folder / 'spot_detection_cnn.pth'
ML.load_model(model_ML, model_path)
```

During colocalization analysis:

1. User runs spot tracking on a reference channel
2. For each detected spot, a crop is extracted from the test channel
3. The crop is resized to 64×64 and fed to the CNN
4. If `probability > threshold`, the spot is classified as "colocalized"

## Model Performance

The deployed model (`spot_detection_cnn.pth`) was trained on human-consensus labels.

### Training Data

- **Total samples**: 120 consensus-labeled crops
- **Positive (spot)**: 69
- **Negative (no spot)**: 51

### Probability Distribution (Important!)

The model's sigmoid output is **not well-calibrated** - probabilities cluster in a narrow range:

| Sample Type | Mean Probability | Range |
|-------------|------------------|-------|
| **Positive** (spots) | 0.727 | 0.50 - 0.73 |
| **Negative** (no spots) | 0.507 | 0.50 - 0.73 |

Most samples cluster at the extremes:

- **67 of 69 positives** → probabilities ~0.72-0.73
- **49 of 51 negatives** → probabilities exactly ~0.50

### Threshold Sensitivity

Because of this narrow distribution, small threshold changes have dramatic effects on the **test set**:

| Threshold | TP | FP | TN | FN | Accuracy |
|-----------|----|----|----|----|----------|
| 0.50 | 69 | 51 | 0 | 0 | 57.5% |
| **0.51** | 68 | 2 | 49 | 1 | **97.5%** |
| 0.55 | 68 | 2 | 49 | 1 | 97.5% |
| 0.67 | 67 | 1 | 50 | 2 | 97.5% |

The jump from 57.5% to 97.5% at threshold 0.51 is because 49 negative samples score *exactly* 0.50.

### Practical Recommendation

**Default threshold is 0.51** (set in GUI at `gui/micro.py` line 7481).

This provides optimal performance on the test set:

- Accuracy: 97.5%
- Precision: 97.1%
- Recall: 98.6%

### Confusion Matrix (threshold=0.51, default)

|  | Predicted Positive | Predicted Negative |
|--|--------------------|--------------------|
| **Actual Positive** | TP = 68 | FN = 1 |
| **Actual Negative** | FP = 2 | TN = 49 |

This provides excellent balance between catching spots (98.6% recall) and avoiding false positives (97.1% precision).

## Quick Reference

### To evaluate existing models (no training)

```python
# In ML_Tranining_Validation.ipynb
RETRAIN_MODELS = False  # Load pre-trained weights
```

### To retrain from scratch

```python
# In ML_Tranining_Validation.ipynb
RETRAIN_MODELS = True  # Train new model (~30 min)
```

### To use in Python scripts

```python
import sys
sys.path.append('/path/to/microlive/src')
import ML_SpotDetection as ML

# Load model
model = ML.ParticleDetectionCNN()
ML.load_model(model, '/path/to/spot_detection_cnn.pth')

# Predict on crops (list of numpy arrays)
predictions, probabilities = ML.predict_crops(model, crops, threshold=0.5)
```

## Notes

- The model was trained on MPS (Apple Silicon) but is compatible with CUDA and CPU
- Input crops should be grayscale images (any size, will be resized to 64×64)
- Default probability threshold is 0.5; can be adjusted for precision/recall tradeoff
