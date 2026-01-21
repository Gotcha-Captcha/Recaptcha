# reCAPTCHA Auto-Solver ML Project

This project builds a **machine learning** model to automatically solve **reCAPTCHA** using three Kaggle datasets.

## Approach

**ML Pipeline:**
1. Feature Extraction: HOG, Color Histogram, LBP, Gradient, Texture (GLCM)
2. Visualization: Heatmaps and feature analysis
3. Classification: SVM, Random Forest, K-NN, Logistic Regression

## Datasets

1. [Google reCAPTCHA Dataset](https://www.kaggle.com/datasets/sanjeetsinghnaik/google-recaptcha)
2. [Test Dataset](https://www.kaggle.com/datasets/mikhailma/test-dataset)
3. [Google reCAPTCHA v2 Images](https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images)

## Project Structure

```
Recaptcha/
├── data/
│   ├── processed/     # Preprocessed images
│   └── features/       # Extracted features
├── models/            # Trained model storage
├── src/
│   ├── download.py         # Dataset download
│   ├── preprocess.py       # Data preprocessing
│   ├── feature_extractor.py # Feature extraction + visualization
│   └── train_ml.py         # Traditional ML training
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
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

```bash
# 1. Download datasets
python src/download.py

# 2. Preprocess data
python src/preprocess.py

# 3. Extract features with visualization
python src/feature_extractor.py --feature_type combined --num_samples 20

# 4. Train traditional ML model
python src/train_ml.py --classifier svm --feature_type combined
```

## Feature Types

- **HOG**: Histogram of Oriented Gradients (edge orientations)
- **Color Histogram**: RGB color distribution
- **LBP**: Local Binary Patterns (texture patterns)
- **Gradient**: Gradient magnitude and direction
- **Texture (GLCM)**: Gray-Level Co-occurrence Matrix
- **Combined**: All features combined
