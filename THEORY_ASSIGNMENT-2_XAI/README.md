# Explainable AI (XAI) Implementation

## Problem Statement

Implement and apply Explainable AI (XAI) techniques to understand and interpret machine learning and deep learning model decisions. Develop models for both image and tabular data classification, then use multiple XAI methods to analyze feature importance, visualize decision-making processes, and provide interpretable explanations for individual predictions.

## Student Details

- **Student Name:** Uday Patiram Sapate
- **PRN:** 202301040162
- **Batch:** DL3
- **Branch:** Computer Engineering (CE)
- **Subject:** Deep Learning Theory (MDM)
- **GitHub Repository:** [https://github.com/Udaysapate212/DEEP_LEARNING_ASSIGNMENTS/tree/main/THEORY_ASSIGNMENT-2_XAI](https://github.com/Udaysapate212/DEEP_LEARNING_ASSIGNMENTS/tree/main/THEORY_ASSIGNMENT-2_XAI)

---

## Objectives

1. Understand the importance of transparency and interpretability in ML/CNN models
2. Implement and train models on both image and tabular datasets
3. Apply multiple Explainable AI techniques to analyze model behavior
4. Generate global and local explanations for model predictions
5. Visualize and interpret feature importance and decision patterns
6. Evaluate XAI methods and provide insights for model improvement

---

## Provided Solution

This notebook demonstrates a comprehensive implementation of Explainable AI techniques across different model types and datasets.

### Part 1: Dataset Selection and Preprocessing

- **Image Dataset (CIFAR-10):**
  - 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
  - Training: 50,000 images | Test: 10,000 images
  - Preprocessing: Normalization (0-1 range), one-hot encoding, train-validation split

- **Tabular Dataset (Breast Cancer Wisconsin):**
  - 569 samples with 30 features computed from breast mass images
  - Binary classification: Malignant vs Benign
  - Preprocessing: Standardization (StandardScaler), stratified train-test split

### Part 2: Model Implementation

- **CNN Model (CIFAR-10):**
  - Optimized architecture with 2 convolutional blocks
  - Layers: Conv2D → MaxPooling → Dropout → Dense → Softmax
  - Training: Adam optimizer, categorical cross-entropy loss
  - Callbacks: EarlyStopping, ReduceLROnPlateau
  - Performance: ~70-75% accuracy (quick mode) or ~85%+ (full training)

- **Random Forest Classifier (Breast Cancer):**
  - 200 trees with optimized hyperparameters
  - Max depth: 15, min samples split: 5
  - Performance: ~95%+ accuracy on test set

### Part 3: XAI Techniques Applied

#### Global Explanations:
1. **Feature Importance (Random Forest):** Built-in importance based on mean decrease in impurity
2. **Permutation Importance:** Model-agnostic method measuring feature impact by shuffling
3. **SHAP Summary Plots:** Unified measure showing global feature contributions across all predictions

#### Local Explanations:
1. **SHAP Force Plots:** Visualize how features push predictions from base value for individual samples
2. **SHAP Waterfall Plots:** Intuitive breakdown of feature contributions for single predictions
3. **LIME (Tabular):** Local linear approximations explaining individual tabular predictions
4. **LIME (Images):** Highlights image regions contributing to CNN predictions
5. **Grad-CAM:** Gradient-weighted class activation mapping showing important CNN regions

### Part 4: Visualization and Analysis

- Comparative analysis of feature importance methods (RF, Permutation, SHAP)
- Confusion matrices and classification reports
- Training history plots (accuracy and loss curves)
- Heatmaps showing feature correlations
- SHAP summary plots with feature distributions
- Grad-CAM overlays on original images
- LIME explanations with highlighted regions

### Part 5: Results and Insights

- **Model Performance:**
  - CNN achieved strong performance on CIFAR-10 image classification
  - Random Forest showed excellent accuracy on breast cancer diagnosis
  
- **Feature Importance Consistency:**
  - Top features identified by RF, Permutation, and SHAP methods show high agreement
  - Most important features align with medical domain knowledge
  
- **Local Explanations:**
  - LIME and SHAP provide complementary insights for individual predictions
  - Grad-CAM successfully highlights relevant image regions
  - Explanations enable debugging and validation of model decisions
  
- **Practical Implications:**
  - XAI techniques enable model validation and debugging
  - Explanations build trust with stakeholders
  - Feature insights guide data collection and engineering

---

## Quick Mode Feature

**⚡ OPTIMIZED FOR FAST TRAINING:**
- Quick mode enabled by default: `QUICK_MODE = True`
- Uses 20% of training data
- Reduces epochs from 50 to 15
- Training time: 5-10 minutes (vs 30-60 minutes full mode)
- Expected accuracy: 70-75% (sufficient for XAI demonstration)
- To use full training: Set `QUICK_MODE = False`

**Note:** This assignment focuses on XAI techniques, not achieving maximum accuracy. Quick mode is recommended!

---

## Files Included

- `XAI_Complete_Implementation.ipynb` — Complete implementation with all XAI techniques
- `README.md` — This file
- `requirements.txt` — Python dependencies
- `test_tensorflow.py` — TensorFlow installation verification script
- `setup_kernel.bat` — Jupyter kernel setup helper (Windows)

---

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Setup Jupyter Kernel (if needed)
If you encounter TensorFlow import errors:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=python312 --display-name "Python 3.12"
```
Then in Jupyter: Kernel → Change Kernel → Python 3.12

### Step 3: Open and Run Notebook
```bash
jupyter notebook XAI_Complete_Implementation.ipynb
```

### Step 4: Execute All Cells
- Quick mode is enabled by default (5-10 minutes training)
- For full training: Set `QUICK_MODE = False` in the notebook
- Run all cells in order for complete execution

---

## Key Results

### Model Performance
- **CNN (CIFAR-10):** 70-75% accuracy (quick mode) | 85%+ (full mode)
- **Random Forest (Breast Cancer):** 95%+ accuracy

### XAI Insights
- Successfully identified most important features for breast cancer diagnosis
- Grad-CAM visualizations show CNN focuses on relevant image regions
- SHAP and LIME provide consistent explanations across methods
- Feature importance methods (RF, Permutation, SHAP) show high agreement

### Practical Applications
- Model debugging and validation
- Building trust with stakeholders
- Identifying potential biases
- Guiding feature engineering

---

## Troubleshooting

### TensorFlow Import Error in Jupyter
**Problem:** `ModuleNotFoundError: No module named 'tensorflow.python'`

**Solution:** Change Jupyter kernel to Python 3.12
1. Run: `setup_kernel.bat` (or commands in Step 2 above)
2. In Jupyter: Kernel → Change Kernel → Python 3.12
3. Restart kernel and run again

### Training Too Slow
- Quick mode is already enabled by default
- Close other applications to free up resources
- Consider using Google Colab for free GPU access

### Out of Memory
- Restart Jupyter kernel
- Run cells one by one instead of "Run All"
- Reduce batch size if needed

---

All steps are clearly commented in the notebook for easy understanding and reproducibility.
