# PRACTICAL NO.02: Transfer Learning

## Problem Statement

**Transfer Learning Model Development:** Fine-tune a pre-trained model on a specific dataset by modifying the top layers and optimizing hyperparameters.

---

**Course Name:** Deep Learning (DL)

**Lab Title:** Enhancement of Pre-Trained Deep Learning Models to Improve Brain Tumor Classification

**Student Name:** Uday Sapate (202301040162)

**Student ID:** 202301040001, 202301040033, 202301040162, 202301040176

**Date of Submission:** 18/02/2026

**Group Members:** Kiran Nandi, Anushka Jadhav, Om Panchal

---

## Objectives

1. Study a research paper utilizing a pre-trained model
2. Reproduce the model implementation using the dataset and methodology from the research paper
3. Fine-tune the pre-trained model and optimize hyperparameters
4. Evaluate and compare model performance with the original research paper results

---

## Task 1: Research Paper Selection and Dataset Preparation

### Research Paper
- **DOI:** https://doi.org/10.31449/inf.v47i6.4645
- **Title:** Enhancement of Pre-Trained Deep Learning Models to Improve Brain Tumor Classification

### Dataset
- **Source:** Brain MRI Images with tumor masks
- **Description:** MRI scans labeled as tumor (1) or no-tumor (0) with corresponding segmentation masks
- **Total Samples:** 3929 images from 110 patients

### Preprocessing Steps Implemented
- Image resizing to match model input (224x224 for VGG16, 256x256 for ResNet50)
- Pixel normalization (rescaling to 0-1 range)
- Data augmentation using Keras ImageDataGenerator

### Data Split
- Training: 70%
- Validation: 30%
- Test: 23% held out separately

---

## Task 2: Model Implementation and Fine-Tuning

### Models Implemented

| Model | Base Architecture | Custom Head | Input Shape |
|-------|-------------------|-------------|-------------|
| ResNet50 | Frozen ResNet50 (ImageNet) | Dense(256-256-128) + Softmax | 256x256x3 |
| VGG16 Model-1 | Frozen VGG16 (ImageNet) | Dense(128) + Softmax | 224x224x3 |
| VGG16 Model-2 | Frozen VGG16 (ImageNet) | Dense(512-256-128) + Softmax | 224x224x3 |

### Training Configuration
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Batch Size:** 16
- **Epochs:** Up to 100 (with early stopping)

### Callbacks Used
- EarlyStopping (patience=2)
- ModelCheckpoint (save best weights)
- ReduceLROnPlateau (reduce LR by 5%)
- LearningRateScheduler (5% decay every 3 epochs)

---

## Task 3: Model Evaluation and Performance Comparison

### Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| ResNet50 | 98.88% | 98.88% | 98.88% | 98.88% |
| VGG16 Model-1 | 82.14% | 82.14% | 82.14% | 82.40% |
| VGG16 Model-2 | 83.04% | 83.04% | 83.04% | 83.09% |

### Key Finding
ResNet50 with transfer learning achieved the best performance (98.88% accuracy), outperforming both VGG16 variants.

### Comparison with Research Paper
The implementation successfully reproduces the methodology from the research paper, achieving comparable results using frozen pre-trained weights with custom classification heads.

---

## Reference

**DOI:** https://doi.org/10.31449/inf.v47i6.4645
