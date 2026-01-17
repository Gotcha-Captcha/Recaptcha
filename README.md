# reCAPTCHA Auto-Solver ML Project

This project builds a machine learning model to automatically solve reCAPTCHA using three Kaggle datasets.

## Datasets

1. [Google reCAPTCHA Dataset](https://www.kaggle.com/datasets/sanjeetsinghnaik/google-recaptcha)
2. [Test Dataset](https://www.kaggle.com/datasets/mikhailma/test-dataset)
3. [Google reCAPTCHA v2 Images](https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images)

## Project Structure

```
Recaptcha/
├── data/              # Dataset storage directory
├── models/            # Trained model storage
├── src/
│   ├── download.py    # Dataset download
│   ├── preprocess.py  # Data preprocessing
│   ├── model.py       # Model definition
│   ├── train.py       # Training code
│   └── evaluate.py    # Evaluation code
├── notebooks/         # Jupyter notebooks
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Kaggle API Setup

To download Kaggle datasets, you need a Kaggle API token:

1. Download API token from https://www.kaggle.com/account
2. Save token to `~/.kaggle/kaggle.json`

## Usage

```bash
# 1. Download datasets
python src/download.py

# 2. Preprocess data
python src/preprocess.py

# 3. Train model
python src/train.py

# 4. Evaluate model
python src/evaluate.py
```
