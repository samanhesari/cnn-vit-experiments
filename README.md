# CNN vs Vision Transformer on CIFAR-10 (PyTorch)

## Overview

This project benchmarks four deep learning architectures for CIFAR-10 image classification using PyTorch, comparing both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

CNNs primarily focus on learning local spatial features such as edges, textures, and shapes through convolution operations, making them highly efficient for smaller image datasets. In contrast, Vision Transformers analyse global relationships between image patches using self-attention mechanisms, enabling stronger contextual understanding and improved feature generalisation.

Models implemented:
- **Custom CNN** — A lightweight convolutional neural network built from scratch using convolution, batch normalisation, pooling, and dropout layers.
- **Fine-tuned VGG16** — An ImageNet pretrained VGG16 model adapted for CIFAR-10 using transfer learning and partial layer fine-tuning.
- **Custom Vision Transformer (ViT)** — A transformer-based architecture developed from scratch using patch embeddings, positional encoding, and self-attention mechanisms.
- **Fine-tuned Pretrained ViT-B/16** — A pretrained Vision Transformer fine-tuned on CIFAR-10 by updating the classification head and final transformer block.


The goal of this experiment is to evaluates:
- CNN vs Transformer-based architectures
- Transfer learning effectiveness
- Training efficiency
- Generalisation performance on CIFAR-10


Fine-tuned Pretrained ViT-B/16 achieved over 95% test accuracy on unseen test data within a single epoch, significantly outperforming all other architectures.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- CUDA / GPU Training

---

## Dataset

CIFAR-10 contains 60,000 colour images across 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Input image size:
32×32 RGB images.

---

## Models Implemented

### 🔹 Simple CNN (From Scratch)

A custom Convolutional Neural Network built as a baseline model.

- 3 convolutional blocks with BatchNorm + ReLU
- MaxPooling for spatial downsampling
- Fully connected classifier with dropout
- Trained entirely from scratch on CIFAR-10
- All parameters are trainable

**Key idea:** Learns visual features directly from dataset without any prior knowledge.

---

### 🔹 VGG16 (Transfer Learning)

A pretrained VGG16 model originally trained on ImageNet, adapted for CIFAR-10.

- Pretrained convolutional feature extractor (ImageNet weights)
- Final classification layer replaced for 10 classes
- Most convolutional layers frozen
- Last convolutional block + classifier fine-tuned

**Key idea:** Reuses learned visual features to improve generalisation and reduce training time.

---

### 🔹 Vision Transformer (ViT – From Scratch)

A transformer-based architecture that processes images as sequences of patches.

- Image split into 4×4 patches
- Patch embeddings passed into transformer encoder
- Self-attention used to model global relationships
- No pretraining used (trained from scratch)
- Fully trainable model

**Key idea:** Learns global dependencies instead of local convolutional features.

---

### 🔹 ViT-B/16 (Pretrained + Fine-Tuned)

A pretrained Vision Transformer fine-tuned for CIFAR-10 classification.

- Pretrained on ImageNet-1K
- Patch-based transformer architecture (16×16 patches)
- Most encoder layers frozen
- Last transformer block + classification head fine-tuned

**Key idea:** Combines large-scale pretraining with task-specific adaptation.

---

## 📊 Model Summary Table

| Model | Architecture Type | Total Parameters | Trainable Parameters | Frozen Parameters | Pretrained | Key Idea |
|------|------------------|------------------|---------------------|------------------|------------|----------|
| Simple CNN | CNN (from scratch) | Low (~1–2M) | 100% | 0% | ❌ | Learns features directly from CIFAR-10 |
| VGG16 (TL) | CNN + Transfer Learning | High (~138M) | ~10–15% | ~85–90% | ✅ ImageNet | Transfers pretrained visual features |
| ViT (from scratch) | Transformer | Medium (~5–7M) | 100% | 0% | ❌ | Learns global relationships via patches |
| ViT-B/16 (TL) | Transformer + Transfer Learning | Very High (~85M+) | ~5–10% | ~90–95% | ✅ ImageNet | Strong transfer learning with attention |

---

## Results

| Model | Train Accuracy | Test Accuracy | Best Epoch |
|------|------|------|------|
| Custom CNN | 90.98% | 80.47% | 20 |
| VGG16 Transfer Learning | 94.81% | 86.97% | 24 |
| Custom ViT | 87.63% | 75.45% | 144 |
| Pretrained ViT-B/16 | 99.72% | 96.13% | 18 |

---

## 📈 Learning Curves (Accuracy vs Epoch)

The following plots show training and validation accuracy across epochs for each model.

### Custom CNN
<img src="results/custom_cnn/accuracy_curve.png" width="600"/>

### VGG16 (Transfer Learning)
<img src="results/pretrained_vgg16/accuracy_curve.png" width="600"/>

### Custom Vision Transformer
<img src="results/custom_vit/accuracy_curve.png" width="600"/>

### ViT-B/16 (Pretrained)
<img src="results/finetuned_vit/accuracy_curve.png" width="600"/>


as it is clear
## Key Findings
- Pretrained Vision Transformers achieved the highest accuracy.
- Pretrained Vision Transformers reached more than 95% accuracy on unseen test data in just one epoch.
- Transfer learning significantly outperformed training from scratch.
- Custom ViT required substantially more training epochs.
- custom CNN and custom ViT showed signs of overfitting, where training accuracy increased substantially while validation/test accuracy lagged behind.
- Transfer learning significantly reduced overfitting compared to training models from scratch.
- CNN architectures remained computationally efficient for smaller datasets.
---

## Project Structure

project/
│
├── models/
├── training/
├── notebooks/
├── results/
├── plots/
├── README.md
└── requirements.txt

---

## Future Improvements

- Data augmentation experiments
- Hyperparameter optimisation
- Mixed precision training
- EfficientNet comparison
- Grad-CAM visualisation

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Jupyter Notebook (if not installed)
pip install notebook

# Or install Jupyter Lab
pip install jupyterlab

# 6. Run Jupyter
jupyter notebook
# or
jupyter lab

## Author

[Your Name]

MSc Artificial Intelligence / Data Science
Liverpool, United Kingdom