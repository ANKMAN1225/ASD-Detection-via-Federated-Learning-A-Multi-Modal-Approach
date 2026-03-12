# ASD Detection via Federated Learning: A Multi-Modal Approach

A privacy-preserving machine learning framework for **Autism Spectrum Disorder (ASD) detection** using **Federated Learning**. The project combines facial feature analysis and behavioral video classification in a federated setting, enabling collaborative learning across institutions without sharing raw patient data.

## Problem Description

Autism Spectrum Disorder detection requires analyzing sensitive medical data (facial images, behavioral videos) that cannot be easily centralized due to privacy regulations (HIPAA, GDPR). This project demonstrates how **Federated Learning** enables:

- **Privacy preservation**: Raw data never leaves client devices
- **Multi-institutional collaboration**: Hospitals/clinics can train models jointly
- **Differential Privacy**: Additional noise-based privacy guarantees
- **Fair aggregation**: FedAvg with IID and Non-IID data distributions

## Datasets

### 1. Autistic Children Facial Dataset (Kaggle)

- **Purpose**: Facial feature analysis for ASD detection
- **Structure**: `autistic` and `non_autistic` class folders across `train`, `test`, and `valid` splits
- **Obtain**: [Kaggle - Autistic Children Facial Data Set](https://www.kaggle.com/datasets/cihan063/autistic-children-facial-data-set)
- **Usage**: Place downloaded data in `data/autistic-children-facial-dataset/` with structure:
  ```
  data/autistic-children-facial-dataset/
  ├── train/
  │   ├── autistic/
  │   └── non_autistic/
  ├── test/
  │   ├── autistic/
  │   └── non_autistic/
  └── valid/
      ├── autistic/
      └── non_autistic/
  ```

### 2. SSBD (Self-Stimulatory Behaviour Dataset)

- **Purpose**: Behavioral pattern recognition (arm flapping, head banging, spinning)
- **Classes**: `armflapping`, `headbanging`, `spinning`
- **Obtain**: [SSBD Dataset](https://www.kaggle.com/datasets) or relevant behavioral video datasets
- **Usage**: Place videos in `data/ssbd_data/<class_name>/` with `.mp4` or `.avi` files

> **Important for accurate results**: The reference notebook runs on **Kaggle** with the dataset at `/kaggle/input/autistic-children-facial-data-set`. To achieve the same accuracy locally:
> 1. Download the [Autistic Children Facial Data Set](https://www.kaggle.com/datasets/cihan063/autistic-children-facial-data-set) from Kaggle
> 2. Extract it to `data/autistic-children-facial-dataset/` (or the project will auto-detect it)
> 3. Ensure the structure has `train/`, `test/`, and `valid/` folders with `autistic/` and `non_autistic/` subfolders
>
> If the dataset is not found, dummy random data is used and accuracy will be low (~50%).

## Project Structure

```
asd-federated-learning/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   ├── __init__.py
│   ├── facial_dataset.py      # FacialDataset class
│   └── video_dataset.py       # BehavioralVideoDataset class
├── models/
│   ├── __init__.py
│   ├── mobilenet_model.py     # MobileNetFeatureExtractor
│   └── tcn_model.py           # VideoTCNModel, TCN components, FusedModel
├── federated/
│   ├── __init__.py
│   ├── client.py              # Federated client (local training)
│   ├── server.py              # FedAvg aggregation
│   └── privacy.py             # Differential Privacy, Secure Aggregation
├── training/
│   ├── __init__.py
│   └── trainer.py             # Optimizer, loss setup
├── evaluation/
│   ├── __init__.py
│   └── metrics.py             # Accuracy, F1, AUC-ROC, evaluator
├── visualization/
│   ├── __init__.py
│   └── plots.py               # Plots, comparison tables, research summary
└── experiments/
    ├── __init__.py
    └── experiment_runner.py   # Experiment configs and runner
```

## Setup and Installation

```bash
cd asd-federated-learning
pip install -r requirements.txt
```

## How to Run

```bash
python main.py
```

This runs all configured experiments (IID, Non-IID, Differential Privacy) and generates evaluation reports and visualizations.

## Federated Learning Approach

- **Algorithm**: FedAvg (Federated Averaging)
- **Data Distribution**:
  - **IID**: Data shuffled and split equally across clients
  - **Non-IID**: Dirichlet distribution to simulate heterogeneous client data
- **Privacy**:
  - **Differential Privacy**: Gaussian noise on gradients/parameters
  - **Secure Aggregation**: Placeholder for cryptographic multi-party aggregation
- **Models**:
  - **Facial**: MobileNetV2 feature extractor + classifier (2 classes)
  - **Behavioral**: MobileNetV2 frame encoder + TCN (3 classes)
  - **Fusion**: Late fusion of facial and behavioral features (attention/concat)

## Results Summary

Results are printed to the console and displayed as matplotlib plots. Key metrics include:

- Global accuracy across clients
- Per-client performance and fairness
- Convergence analysis
- Privacy–utility trade-off
- Communication overhead estimation

> Add your actual results here after running experiments.

## Tech Stack

- **Python 3.8+**
- **PyTorch** – Deep learning
- **torchvision** – MobileNetV2, transforms
- **OpenCV** – Video frame extraction
- **scikit-learn** – Metrics (F1, AUC-ROC, etc.)
- **matplotlib, seaborn** – Visualization
- **Differential Privacy** – Noise injection for privacy

## License

Capstone project for academic submission.
