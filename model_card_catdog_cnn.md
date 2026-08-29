# Model Card — CatDogCNN

**Model type:** Custom Convolutional Neural Network (built from scratch, no pretrained weights)
**Task:** Binary image classification — Cat vs Dog
**Framework:** PyTorch
**Roadmap reference:** Barakode AI Engineering Roadmap, Week 1 — "Develop custom CNN architectures for image tasks"

---

## 1. Overview

`CatDogCNN` is a convolutional neural network designed and trained from scratch (no transfer learning) to classify an input image as either a cat or a dog. The model was built to satisfy the roadmap's requirement of implementing a CNN architecture manually, including a custom training loop, loss function usage, and optimizer configuration in PyTorch.

## 2. Architecture

| Layer | Details | Output shape |
|---|---|---|
| Conv Block 1 | Conv2d(3→16, k=3, pad=1) → ReLU → MaxPool(2) | 16 × 64 × 64 |
| Conv Block 2 | Conv2d(16→32, k=3, pad=1) → ReLU → MaxPool(2) | 32 × 32 × 32 |
| Conv Block 3 | Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool(2) | 64 × 16 × 16 |
| Flatten | — | 16,384 |
| FC1 | Linear(16384 → 128) → ReLU → Dropout(0.5) | 128 |
| FC2 (output) | Linear(128 → 1) | 1 (raw logit) |

**Total parameters:** ~2.1 million
**Input size:** 128 × 128 RGB
**Output:** Single logit → sigmoid → P(Dog); label = Dog if P(Dog) ≥ 0.5, else Cat

## 3. Training Setup

| Item | Value |
|---|---|
| Dataset | Microsoft Cats vs Dogs (Kaggle: `shaunthesheep/microsoft-catsvsdogs-dataset`) |
| Train / Val split | 19,998 / 5,000 images (80/20, corrupt files filtered) |
| Corrupt files skipped | 4 total (2 Cat, 2 Dog) |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | Adam, lr = 0.001 |
| Batch size | 32 |
| Epochs | 10 |
| Augmentation | Random horizontal flip |
| Normalization | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] |
| Hardware | Kaggle GPU (T4) |
| Total training time | ~15.5 minutes (10 epochs, ~93 sec/epoch) |

## 4. Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.599 | 66.96% | 0.514 | 74.06% |
| 3 | 0.453 | 78.85% | 0.452 | 78.06% |
| 5 | 0.374 | 83.85% | 0.433 | 80.76% |
| 7 | 0.311 | 86.63% | 0.330 | 85.66% |
| 9 | 0.273 | 88.46% | 0.309 | **86.58%** |
| 10 | 0.249 | 89.61% | 0.323 | 86.36% |

**Best model:** Epoch 9, val accuracy **86.58%** (this is the checkpoint saved as `cat_dog_cnn_best.pt` and used in the deployed interface).

**Trend:** Train and val accuracy rise together through epoch 9 with a small, steady gap (~2 points at the end) — a mild, healthy sign of the model still generalizing well. Val loss ticked up slightly at epoch 10 while train loss kept falling, an early hint of overfitting starting to creep in, which is why the epoch-9 checkpoint (not the final epoch) is used as the deployed model.

## 5. Intended Use

- Classifying clear, single-subject photos of cats or dogs (e.g. pet photos, adoption listings, casual snapshots).
- Educational / portfolio use — demonstrates a from-scratch CNN training pipeline.

## 6. Limitations

- Not evaluated on images with multiple animals, occluded/partial views, cartoons, or non-photographic images.
- Trained only on cats and dogs — any other input will still be forced into one of these two labels (no "unknown" class).
- Modest 128×128 input resolution — fine detail (e.g. distinguishing similar-looking breeds) is not the focus of this model.
- No hyperparameter search was performed; 10 epochs and default Adam settings were used as a first working baseline.

## 7. How to Load

```python
import torch
from cnn_cat_dog_classifier import CatDogCNN  # or the class defined in app.py

model = CatDogCNN()
model.load_state_dict(torch.load("cat_dog_cnn_best.pt", map_location="cpu"))
model.eval()
```

Deployed via a Flask interface (`PawScan`) with drag-and-drop image upload and a confidence score.
