
# CIFAR-10 Image Classification using CNNs and Vision Transformers

A deep learning project comparing convolutional neural networks (CNNs) and Vision Transformers (ViTs) on the CIFAR-10 dataset using PyTorch.

This project explores:
- Custom CNN architectures
- Transfer learning with VGG16
- Custom Vision Transformer implementation
- Fine-tuning pretrained ViT models

The goal was to evaluate how traditional CNNs compare against transformer-based architectures for image classification tasks.

---

# Project Overview

In this project, four different deep learning models were trained and evaluated on CIFAR-10:

| Model | Category | Approach |
|---|---|---|
| Simple CNN | CNN | Built from scratch |
| VGG16 | CNN | Transfer Learning |
| Custom Vision Transformer | Transformer | Built from scratch |
| ViT-B/16 | Transformer | Transfer Learning |

---

# Dataset

The project uses the CIFAR-10 dataset containing 60,000 colour images across 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset image size: `32x32 RGB`

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

---

# Models Implemented

## 1. Simple CNN
Custom convolutional neural network with:
- Convolution layers
- Batch Normalisation
- ReLU activations
- Max Pooling
- Dropout regularisation

### Key Learning
Built a lightweight baseline CNN architecture for image classification.

---

## 2. VGG16 Transfer Learning
Fine-tuned pretrained VGG16 from ImageNet.

### Transfer Learning Strategy
- Frozen early layers
- Fine-tuned:
  - Final convolution block
  - Fully connected classifier layers

### Key Learning
Demonstrated how pretrained CNN features improve performance on smaller datasets.

---

## 3. Custom Vision Transformer (ViT)
Implemented a Vision Transformer from scratch using:
- Patch embeddings
- Positional encoding
- Transformer encoder blocks
- Multi-head self-attention

### Key Learning
Explored transformer-based image classification without convolution operations.

---

## 4. ViT-B/16 Transfer Learning
Fine-tuned pretrained Vision Transformer from ImageNet.

### Fine-Tuning Strategy
- Frozen encoder layers
- Fine-tuned:
  - Final transformer block
  - Classification head

### Key Learning
Compared transformer transfer learning performance against CNN-based transfer learning.

---

# Results

| Model | Test Accuracy |
|---|---|
| Simple CNN | XX% |
| VGG16 Transfer Learning | XX% |
| Custom ViT | XX% |
| ViT-B/16 Transfer Learning | XX% |

---

# Sample Training Curves

Add your plots here:

```markdown
![Training Accuracy](results/training_curves.png)























# Deep Learning Benchmark: CNNs vs Vision Transformers on CIFAR-10









Built as part of independent deep learning research and experimentation using PyTorch.

---

## Project Overview

This project benchmarks and compares four deep learning architectures for image classification on the CIFAR-10 dataset using PyTorch.

The goal was to evaluate the differences between traditional Convolutional Neural Networks (CNNs) and Transformer-based architectures in terms of:

- Classification accuracy
- Training efficiency
- Transfer learning performance
- Generalisation capability
- Model complexity

The project includes both custom-built architectures and pretrained transfer learning approaches.

---

## Models Implemented

### 1. Simple CNN
A lightweight convolutional neural network designed from scratch for CIFAR-10 classification.

Features:
- 3 convolutional blocks
- Batch Normalisation
- ReLU activations
- MaxPooling
- Dropout regularisation

---

### 2. VGG16 Transfer Learning
Fine-tuned ImageNet-pretrained VGG16 model.

Transfer learning strategy:
- Frozen early convolutional layers
- Fine-tuned Conv5 block
- Custom classifier head for CIFAR-10

---

### 3. Custom Vision Transformer (ViT)
A Vision Transformer implementation built from scratch.

Components:
- Patch Embedding
- Positional Encoding
- CLS Token
- Multi-Head Self-Attention
- Transformer Encoder Layers

---

### 4. ViT-B/16 Fine-Tuning
Fine-tuned pretrained Vision Transformer using ImageNet weights.

Fine-tuning strategy:
- Frozen encoder layers
- Fine-tuned final transformer block
- Custom classification head

---

# Dataset

## CIFAR-10

The CIFAR-10 dataset contains:
- 60,000 colour images
- 10 image classes
- Image size: 32×32 RGB

Classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Project Structure

```text
project/
│
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
│
├── models/
│   ├── simple_cnn.py
│   ├── vgg16_transfer.py
│   ├── custom_vit.py
│   └── vit_transfer.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│   └── metrics.csv
│
└── saved_models/