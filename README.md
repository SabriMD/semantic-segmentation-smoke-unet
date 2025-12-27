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
