# Diabetic Retinopathy Classification using CNNs

This repository contains deep learning experiments for **Diabetic Retinopathy (DR) classification** using convolutional neural networks (CNNs).
The goal is to identify retinal disease severity from fundus images through progressive model benchmarking and optimization.

Dataset: https://www.kaggle.com/datasets/mohamedabdalkader/retinal-disease-detection
## Project Overview

Diabetic Retinopathy (DR) is one of the leading causes of vision loss globally.
Early and accurate detection through automated classification of retinal fundus images can significantly improve patient outcomes.

This project explores a series of experiments involving **CNN architectures**, **image resolutions**, and **imbalance handling techniques** to optimize DR classification performance.

## Features

* Multi-architecture evaluation: **ResNet50**, **DenseNet121**, **EfficientNet-B0**
* Multi-scale input resolutions (128×128, 256×256, 512×512)
* Class imbalance strategies: Inverse weighting & weighted random sampling
* Data augmentation applied selectively to minority classes

## Experimental Summary

| Experiment Folder | Model           | Input Size | Imbalance Handling | Sampler                 | Image Transformation                            | Epochs | Patience | Early Stopping Index | AUC    | F1     | Recall | Precision | Accuracy | Config      |
| :---------------: | :-------------- | :--------- | :----------------- | :---------------------- | :---------------------------------------------- | :----- | :------- | :------------------- | :----- | :----- | :----- | :-------- | :------- | :---------- |
|         1         | ResNet50        | 128×128    | Inverse Weight     | None                    | None                                            | 10     | 5        | 7                    | 0.9223 | 0.7606 | 0.8182 | 0.7105    | 0.8994   | 32 GB + MPS |
|         2         | ResNet50        | 256×256    | Inverse Weight     | None                    | None                                            | 10     | 5        | 9                    | 0.9628 | 0.8657 | 0.8788 | 0.8529    | 0.9467   | 32 GB + MPS |
|         3         | ResNet50        | 512×512    | Inverse Weight     | None                    | None                                            | 10     | 5        | —                    | 0.9562 | 0.7294 | 0.9394 | 0.5962    | 0.8639   | 32 GB + MPS |
|         4         | EfficientNet_B0 | 256×256    | Inverse Weight     | None                    | None                                            | 10     | 5        | 7                    | 0.9667 | 0.8000 | 0.8182 | 0.7826    | 0.9201   | 32 GB + MPS |
|         5         | DenseNet121     | 256×256    | Inverse Weight     | None                    | None                                            | 10     | 5        | 7                    | 0.9589 | 0.8382 | 0.8636 | 0.8143    | 0.9349   | 32 GB + MPS |
|         6         | ResNet50        | 256×256    | None               | Weighted Random Sampler | Only Minority Class [Horizontal Flip, Rotation] | 10     | 5        | —                    | 0.9461 | 0.6465 | 0.4848 | 0.9697    | 0.8964   | 32 GB + MPS |
|         7         | ResNet50        | 256×256    | None               | Weighted Random Sampler | Only Minority Class [Horizontal Flip, Rotation] | 20     | 3        | 4                    | 0.8735 | 0.6296 | 0.7727 | 0.5312    | 0.8225   | 32 GB + MPS |
|         8         | ResNet50        | 256×256    | Inverse Weight     | Weighted Random Sampler | None                                            | 20     | 3        | 10                   | 0.9570 | 0.8310 | 0.8939 | 0.7763    | 0.9290   | 32 GB + MPS |

NOTE: Metal Performance Shaders (MPS), a framework for high-performance, data-parallel computation on Apple hardware.

## Repository Structure

```
diabetic-retinopathy-kaggle/
│
├── data/                      # Dataset (train/test split, metadata, etc.)
├── experiments/               # Experiment folders with logs and best models
│   ├── 001/
│   ├── 002/
│   ├── 003/
│   └── ...
│
├── models/                    # Model definitions (ResNet, DenseNet, etc.)
├── utils/                     # Helper scripts for metrics, training loops, and loaders
├── configs/                   # YAML/JSON configuration files per experiment
├── train.py                   # Main training entry point
├── evaluate.py                # Evaluation and reporting scripts
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/pradhanhitesh/diabetic-retinopathy-kaggle.git
cd diabetic-retinopathy-kaggle
```

### 2. Create and activate environment

```bash
python3.10 -m venv venv
source venv/bin/activate    # (or venv\Scripts\activate on Windows)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train a model

```bash
python main.py --mode train --config config/config.yml --dataset src/data
```

### 5. Evaluate a trained model
If you want to evaluate the latest model,
```bash
python main.py --mode test --config config/config.yml
```
If you want to evaluate any specific model,
```bash
python main.py --mode test --config experiments/001/config.yaml --model_path experiments/001/best_model.pth
```

## Key Observations

* Increasing image size from 128×128 → 256×256 improved both **AUC** and **F1-score**.
* **Inverse weighting** proved more stable than **sampler-based balancing**.
* **EfficientNet_B0** outperformed ResNet50 and DenseNet121 marginally on AUC but showed slower convergence.
* Data augmentation applied **only to minority classes** didn’t outperform global inverse weighting.
* Training beyond 20 epochs yielded diminishing returns — early stopping typically triggered between epochs 7–10.

## Upcoming Experiments

| ID | Objective                                                                 |
| -- | ------------------------------------------------------------------------- |
| 1  | Test on **Windows (6 GB RTX 3090 GPU)**                                |
| 2  | Test on **Linux (48 GB A100 GPU)**                                     |
| 3  | Implement **Neural Architecture Search (NAS)**                         |
| 4  | Extend to **multi-class DR prediction**                                |
| 5  | Develop **clinician comment generation** (vision-language integration) |

---

## 🧑‍💻 Author

**Hitesh Pradhan**
Data Scientist & Machine Learning Researcher | [pradhanhitesh](https://github.com/pradhanhitesh)

## 📄 License

This repository is released under the **MIT License** — feel free to use, modify, and build upon it with proper attribution.

