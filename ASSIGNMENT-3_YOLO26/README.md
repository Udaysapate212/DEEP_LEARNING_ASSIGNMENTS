# YOLOv8 Object Detection Lab

## Problem Statement

Develop and evaluate an object detection, segmentation, and classification pipeline using YOLOv8. Prepare a dataset, train and fine-tune models, analyze results, and demonstrate a real-world application.

## Group & Submission Details

- **Institute:** MIT Academy of Engineering, Pune
- **Department:** Computer Engineering
- **Course:** Deep Learning (2311332L)
- **Lab Date:** 16/02/2026
- **Submission Date:** 25/03/2026
- **Group Members:**
  - Kiran Nandi (202301040001)
  - Anushka Jadhav (202301040033)
  - Uday Sapate (202301040162)
  - Om Panchal (202301040176)
- **Faculty:** Dr. Diptee Ghusse

## Objectives

1. Understand YOLOv8 architecture and its variants (detection, segmentation, classification).
2. Prepare and analyze a dataset (COCO128 or custom YOLO-format).
3. Train, evaluate, and fine-tune YOLOv8 models.
4. Benchmark model performance and compare variants.
5. Demonstrate a real-world application (e.g., traffic monitoring).

## Solution Workflow

- **Environment Setup:**
  - Install dependencies: `ultralytics`, `roboflow`, `supervision`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `opencv-python-headless`, `pillow`.
  - Check GPU availability for faster training.

- **Dataset Preparation:**
  - Use COCO128 (default) or import a custom dataset in YOLO format.
  - Visualize and explore sample images.

- **Model Training & Evaluation:**
  - Train YOLOv8 detection model (e.g., `yolov8n.pt`) for a few epochs.
  - Evaluate using metrics: mAP@50, mAP@50-95, Precision, Recall.
  - Visualize training curves and confusion matrix.

- **Inference & Visualization:**
  - Run detection, segmentation, and classification on test images.
  - Display and save output images with predictions and masks.

- **Benchmarking & Analysis:**
  - Compare YOLOv8 variants (nano, small, medium) for speed and accuracy.
  - Analyze model size and inference time.

- **Fine-Tuning & Hyperparameter Tuning:**
  - Experiment with learning rate, momentum, weight decay, and augmentation.
  - Summarize results in a comparison table.

- **Model Export:**
  - Export best model weights to ONNX and TorchScript formats for deployment.

- **Application Demo:**
  - Example: Intelligent Traffic Monitoring using YOLOv8 for real-time vehicle and pedestrian detection.

## Results & Observations

- Successfully trained and evaluated YOLOv8 models on COCO128.
- Achieved reasonable mAP and accuracy for detection, segmentation, and classification tasks.
- Demonstrated real-world application with traffic images.
- Compared model sizes and inference times for deployment considerations.

## Files Included

- `P3_YOLOv8_Object_Detection_Lab_Completed.ipynb` — Complete code and results
- `README.md` — This file

## How to Run

1. Open the notebook in Google Colab or a local Jupyter environment with GPU support.
2. Run all cells in order for end-to-end execution.
3. Modify dataset or hyperparameters as needed for custom experiments.

---

All steps are clearly commented in the notebook for easy understanding and reproducibility.