# UNet Smoke Semantic Segmentation

This repository contains an implementation of a **U-Net architecture built and trained from scratch using PyTorch** for a **semantic segmentation task focused on smoke detection** in images.

The project covers the full pipeline: data loading, augmentation, model definition, training, and evaluation.

---

## Project Overview

- **Task:** Semantic Segmentation  
- **Application:** Smoke detection in images  
- **Model:** U-Net (from scratch)  
- **Framework:** PyTorch  
- **Training:** Supervised learning  
- **Output:** Pixel-wise segmentation masks  

---

## 🧠 Model Architecture

The model follows the standard **U-Net encoder–decoder architecture**, consisting of:

- **Encoder (Contracting Path):**
  - Convolutional blocks
  - Downsampling via max pooling
- **Bottleneck:**
  - Deep feature representation
- **Decoder (Expanding Path):**
  - Upsampling layers
  - Skip connections to recover spatial resolution

Skip connections help preserve spatial information and improve segmentation accuracy.

---

## 📂 Repository Structure

├── UNet_Smoke_Semantic_Segmentation.ipynb
├── requirements.txt
├── README.md

## Data Augmentation

Data augmentation is performed using **Albumentations**, including geometric transformations, color adjustments, and normalization.  
These augmentations help the model generalize better to unseen images.

### Training Transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    # Resize all images to 512x512 (U-Net input size)
    A.Resize(512, 512),

    # Random horizontal flip with 50% probability
    A.HorizontalFlip(p=0.5),

    # Random vertical flip with 50% probability
    A.VerticalFlip(p=0.5),

    # Random 90-degree rotations with 50% probability
    A.RandomRotate90(p=0.5),

    # Random color jitter (brightness, contrast, saturation) with 50% probability
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        p=0.5
    ),

    # Normalize image pixels to range [-1, 1] (mean=0.5, std=0.5)
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # Convert image and mask to PyTorch tensors
    ToTensorV2()
])
