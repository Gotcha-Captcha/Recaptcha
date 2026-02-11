# reCAPTCHA Auto-Solver 프로젝트 구조 설계

## 📋 프로젝트 개요

**목표**: Hand-crafted features + ML classifiers로 reCAPTCHA 이미지 분류 자동화

**접근 방식**: 
- 딥러닝 대신 수동 특징 추출 (HOG, Color Histogram, LBP, Gradient, Texture)
- ML 분류기 (SVM, Random Forest, K-NN, Logistic Regression, XGBoost, AdaBoost)
- K-fold Cross-Validation으로 데이터 분할
- Feature selection, Ensemble methods 적용

---

## 📁 디렉토리 구조

```
Recaptcha/
├── data/                          # 데이터 저장소
│   ├── raw/                       # 원본 다운로드 데이터 (Kaggle에서 받은 그대로)
│   │   ├── google-recaptcha/      # 데이터셋 1
│   │   ├── test-dataset/          # 데이터셋 2
│   │   └── google-recaptcha-v2/   # 데이터셋 3
│   ├── processed/                 # 전처리된 이미지
│   │   ├── images/                # 모든 전처리된 이미지 (K-fold용)
│   │   ├── metadata.csv           # 이미지 경로와 레이블 정보
│   │   └── label_mapping.json     # 클래스 이름 ↔ 숫자 매핑
│   └── features/                  # 추출된 특징 벡터
│       ├── hog/                   # HOG 특징만
│       │   ├── train_features.npy
│       │   ├── train_labels.npy
│       │   └── ...
│       ├── color_hist/            # Color Histogram 특징만
│       ├── lbp/                   # LBP 특징만
│       └── combined/              # 모든 특징 결합
│           ├── train_features.npy
│           ├── train_labels.npy
│           ├── visualizations/   # 특징 시각화 이미지들
│           └── statistics.json
│
├── models/                        # 학습된 모델 저장
│   ├── svm_combined_model.pkl
│   ├── svm_combined_scaler.pkl
│   ├── ensemble_model.pkl
│   └── results/                   # 학습 결과
│       ├── kfold_results/         # K-fold별 결과
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── learning_curves.png
│       ├── feature_importance.png
│       ├── metrics.json
│       └── training_log.json
│
├── src/                           # 소스 코드
│   ├── __init__.py
│   ├── download.py                # 데이터셋 다운로드 (API 또는 수동)
│   ├── preprocess.py              # 이미지 전처리
│   ├── feature_extraction.py      # 특징 추출
│   ├── feature_selection.py       # 특징 선택 (PCA, SelectKBest 등)
│   ├── visualization.py           # 특징 시각화
│   ├── train.py                   # ML 모델 학습 (K-fold 포함)
│   ├── ensemble.py                # 앙상블 모델
│   └── evaluate.py                # 모델 평가
│
├── notebooks/                     # Jupyter 노트북 (분석용)
│   └── exploration.ipynb
│
├── config/                        # 설정 파일
│   └── config.yaml               # 하이퍼파라미터 설정
│
├── scripts/                       # 실행 스크립트
│   ├── run_pipeline.sh           # 전체 파이프라인 실행
│   └── extract_features.sh       # 특징 추출만 실행
│
├── requirements.txt               # Python 패키지 의존성
├── README.md                      # 프로젝트 설명
└── PROJECT_STRUCTURE.md           # 이 문서
```

---

## 🔄 데이터 흐름 (Pipeline)

```
1. 데이터 다운로드 (download.py)
   └─> data/raw/ (원본 데이터)

2. 이미지 전처리 (preprocess.py)
   └─> data/processed/ (정규화된 이미지)
       - RGB → HSV 변환
       - Histogram Equalization
       - 크기 통일 (224x224)
       - K-fold 분할 준비 (분할은 train.py에서 수행)

3. 특징 추출 (feature_extraction.py)
   └─> data/features/ (특징 벡터)
       - HOG: 1764차원
       - Color Histogram: 96차원
       - LBP: 26차원
       - Gradient: 64차원
       - Texture (GLCM): 60차원
       - Combined: 2010차원

4. 시각화 (visualization.py)
   └─> data/features/*/visualizations/ (히트맵, 차트 등)

5. 특징 선택 (feature_selection.py) - 선택적
   └─> 차원 축소, 중요 특징 선택

6. 모델 학습 (train.py)
   └─> K-fold Cross-Validation
   └─> 앙상블 모델 생성
   └─> models/ (학습된 모델)

7. 모델 평가 (evaluate.py)
   └─> K-fold 결과 통합
   └─> models/results/ (성능 지표, 혼동 행렬, ROC 곡선 등)
```

---

## 📝 각 모듈 상세 설계

### 1. `src/download.py` - 데이터셋 다운로드

**기능**:
- Kaggle API를 사용하여 3개 데이터셋 다운로드 (선택적)
- 또는 로컬에 이미 다운로드된 데이터 사용
- 데이터셋 구조 탐색 및 검증
- 다운로드 상태 확인 및 에러 처리

**입력**: 
- 옵션 1: Kaggle API 토큰 (`~/.kaggle/kaggle.json`)
- 옵션 2: 로컬 디렉토리 경로 (이미 다운로드된 경우)

**출력**: 
- `data/raw/` 디렉토리에 원본 데이터 저장
- `data/raw/dataset_info.json` (데이터셋 통계)

**주요 함수**:
```python
def download_from_kaggle(dataset_name, output_dir)
def use_local_data(local_path, output_dir)
def validate_dataset_structure(data_dir)
def explore_dataset(data_dir) -> dataset_info
def main()
```

**시각화**:
- 데이터셋 구조 트리 다이어그램
- 클래스별 이미지 개수 바 차트
- 샘플 이미지 그리드

---

### 2. `src/preprocess.py` - 이미지 전처리

**기능**:
- 이미지 색상 공간 변환 (RGB → HSV/Lab)
- Histogram Equalization (CLAHE)
- 이미지 크기 통일 (224x224)
- 데이터 증강 (Data Augmentation) - 회전, 뒤집기, 밝기 조정
- 클래스 불균형 처리 (SMOTE, Undersampling 등)
- Train/Val/Test 분할 (Stratified Split으로 클래스 비율 유지)

**입력**: `data/raw/` 디렉토리

**출력**: 
- `data/processed/images/` (전처리된 이미지)
- `data/processed/metadata.csv` (이미지 경로와 레이블)
- `data/processed/label_mapping.json` (클래스 매핑)
- `data/processed/preprocessing_report.html` (전처리 결과 리포트)

**시각화**:
- 전처리 전/후 이미지 비교
- 색상 공간 변환 시각화
- Histogram Equalization 효과 비교
- 클래스 분포 파이 차트
- `data/processed/preprocessing_report.html` (전처리 통계 리포트)

**주요 함수**:
```python
def preprocess_image(image) -> processed_image
def convert_color_space(image, target_space='HSV')
def apply_histogram_equalization(image)
def augment_image(image) -> augmented_images
def handle_class_imbalance(images, labels) -> balanced_data
def stratified_split(images, labels, ratios) -> train, val, test
def create_metadata(image_paths, labels)
def generate_preprocessing_report(data_stats)
def main()
```

**시각화**:
- 전처리 전/후 이미지 비교
- 클래스별 데이터 분포 (파이 차트, 바 차트)
- 색상 공간 변환 시각화
- Histogram Equalization 효과 비교
- 데이터 증강 샘플 그리드

---

### 3. `src/feature_extraction.py` - 특징 추출

**기능**:
- HOG (Histogram of Oriented Gradients) 특징 추출
- Color Histogram 특징 추출
- LBP (Local Binary Patterns) 특징 추출
- Gradient 특징 추출
- Texture (GLCM) 특징 추출
- 특징 결합 및 저장
- 특징 상관관계 분석
- 특징 중요도 계산

**입력**: `data/processed/` 디렉토리

**출력**: 
- `data/features/*/all_features.npy` (전체 특징 벡터)
- `data/features/*/all_labels.npy` (전체 레이블)
- `data/features/*/statistics.json` (특징 통계)
- `data/features/*/feature_correlation.png` (특징 간 상관관계)

**시각화**:
- 각 특징의 분포 히스토그램
- 특징 간 상관관계 히트맵
- 특징 중요도 (분산 기반)
- `data/features/*/feature_analysis.json` (특징 분석 결과)

**주요 클래스/함수**:
```python
class FeatureExtractor:
    def extract_hog(image) -> features (1764차원)
    def extract_color_histogram(image) -> features (96차원)
    def extract_lbp(image) -> features (26차원)
    def extract_gradient(image) -> features (64차원)
    def extract_texture(image) -> features (60차원)
    def extract_combined(image) -> features (2010차원)

def extract_features_from_dataset(data_dir, feature_type)
def analyze_feature_correlation(features, labels)
def calculate_feature_importance(features, labels)
def save_features(features, labels, output_dir)
def main()
```

**시각화**:
- 특징별 분포 히스토그램
- 특징 간 상관관계 히트맵
- 특징 중요도 바 차트
- 특징 차원별 분산 분석
- 클래스별 특징 분포 비교

---

### 4. `src/visualization.py` - 특징 시각화

**기능**:
- 전처리 과정 시각화 (원본 → HSV → 평활화)
- 각 특징의 히트맵 생성
- 특징 분포 및 통계 시각화
- 샘플 이미지별 상세 분석 리포트 생성

**입력**: 
- `data/processed/` (이미지)
- `data/features/` (특징 벡터)

**출력**: 
- `data/features/*/visualizations/*.png` (20개 패널 시각화)

**주요 함수**:
```python
def visualize_preprocessing(image, processed_image)
def visualize_hog_features(image, hog_features)
def visualize_color_histogram(image, hist_features)
def visualize_lbp_features(image, lbp_features)
def create_comprehensive_report(image_path, features)
def main()
```

**시각화 패널 구성** (20개):
1. 원본 이미지
2. HSV 변환
3. Histogram Equalization 비교
4. 전처리 결과
5. 특징 벡터 구성
6-8. HOG 시각화 + 히트맵 + 원리 설명
9-10. Color Histogram + 원리 설명
11-13. LBP 패턴 + 히스토그램 + 원리 설명
14-15. Gradient 크기/방향
16-20. 통합 분석 (히트맵, 통계, 분포, 중요도, 최종 요약)

---

### 5. `src/train.py` - ML 모델 학습

**기능**:
- 특징 벡터 로드
- 특징 정규화 (StandardScaler, MinMaxScaler, RobustScaler)
- Feature Selection (SelectKBest, RFE, PCA)
- K-fold Cross Validation (기본 5-fold, Stratified K-Fold)
- 여러 분류기 학습 및 비교
- 하이퍼파라미터 최적화 (Grid Search, Random Search, Bayesian Optimization)
- 앙상블 모델 (Voting, Stacking)
- 모델 저장 및 학습 곡선 기록

**입력**: `data/features/*/` (특징 벡터)

**출력**: 
- `models/*_model.pkl` (학습된 모델)
- `models/*_scaler.pkl` (정규화 스케일러)
- `models/*_selector.pkl` (Feature Selector)
- `models/results/*_metrics.json` (성능 지표)
- `models/results/cv_results.json` (K-fold 결과)
- `models/results/training_curves.png` (학습 곡선)

**실행 환경**:
- **맥북에서 실행 가능**: 
  - CPU 기반 ML 학습 (SVM, Random Forest, XGBoost 등)
  - 소규모~중규모 데이터셋 (~10K-50K 이미지) 처리 가능
  - 특징 추출: CPU로 충분 (병렬 처리 활용)
  - 학습 시간: 분류기와 데이터 크기에 따라 수분~수십분
  - 메모리: 8GB RAM 이상 권장 (특징 벡터 메모리 사용)
- **Colab 사용 권장 경우**:
  - 대규모 하이퍼파라미터 튜닝 (GridSearchCV)
  - 여러 모델 동시 학습 및 비교
  - GPU 가속이 필요한 경우 (하지만 전통 ML은 CPU로 충분)

**주요 함수**:
```python
def load_features(feature_dir, split='train')
def normalize_features(X_train, X_val, method='standard')
def select_features(X_train, y_train, method='pca', n_components=100)
def k_fold_cross_validation(X, y, classifier, k=5, stratified=True)
def optimize_hyperparameters(X_train, y_train, classifier_type, method='grid')
def train_model(X_train, y_train, X_val, y_val, classifier)
def create_ensemble(models, method='voting')
def plot_training_curves(cv_results)
def save_model(model, scaler, selector, output_path)
def main()
```

**지원하는 분류기**:
- SVM (Support Vector Machine) - Linear, RBF, Polynomial kernels
- Random Forest
- XGBoost (Gradient Boosting)
- LightGBM (Light Gradient Boosting)
- K-NN (K-Nearest Neighbors)
- Logistic Regression
- Naive Bayes (Gaussian, Multinomial)
- Decision Tree
- AdaBoost
- Gradient Boosting

**성능 향상 기법**:
- **K-fold Cross Validation**: Stratified K-fold로 클래스 비율 유지하며 일반화 성능 향상
- **Feature Selection**: SelectKBest, RFE, PCA로 불필요한 특징 제거 및 차원 축소
- **하이퍼파라미터 최적화**: GridSearchCV, RandomSearchCV, Bayesian Optimization
- **앙상블**: Voting Classifier, Stacking Classifier로 여러 모델 결합
- **클래스 가중치 조정**: 불균형 데이터 처리 (class_weight='balanced')
- **Feature Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **SMOTE**: 소수 클래스 오버샘플링 (선택적)
- **Early Stopping**: XGBoost, LightGBM에서 과적합 방지

**시각화**:
- K-fold CV 결과 박스플롯
- 하이퍼파라미터 최적화 히트맵
- 학습 곡선 (Accuracy, Loss)
- Feature Importance 차트
- 모델 성능 비교 바 차트
- 학습 시간 비교

---

### 6. `src/evaluate.py` - 모델 평가

**기능**:
- 테스트 데이터로 모델 평가
- 정확도, Precision, Recall, F1-Score, AUC 계산
- 혼동 행렬(Confusion Matrix) 생성
- 클래스별 성능 분석
- 오분류 샘플 분석
- 모델 비교 (여러 모델 성능 비교)
- ROC Curve, Precision-Recall Curve 생성

**입력**: 
- `models/*_model.pkl` (학습된 모델)
- `data/features/*/test_*.npy` (테스트 특징)

**출력**: 
- `models/results/confusion_matrix.png`
- `models/results/classification_report.txt`
- `models/results/metrics.json`
- `models/results/roc_curve.png`
- `models/results/pr_curve.png`
- `models/results/model_comparison.png`
- `models/results/misclassified_samples/` (오분류 이미지)

**주요 함수**:
```python
def load_model(model_path)
def evaluate_model(model, X_test, y_test)
def plot_confusion_matrix(y_true, y_pred, class_names)
def plot_roc_curve(y_true, y_proba, class_names)
def plot_pr_curve(y_true, y_proba, class_names)
def analyze_misclassifications(model, X_test, y_test, image_paths)
def compare_models(model_results)
def generate_report(metrics, confusion_matrix)
def main()
```

**시각화**:
- 혼동 행렬 히트맵
- ROC Curve (각 클래스별)
- Precision-Recall Curve
- 모델 성능 비교 바 차트
- 클래스별 성능 비교
- 오분류 샘플 이미지 그리드
- 예측 확률 분포

---

## ⚙️ 설정 파일 (`config/config.yaml`)

```yaml
# 데이터 설정
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  features_dir: "data/features"
  
  # 데이터셋 목록
  datasets:
    - name: "sanjeetsinghnaik/google-recaptcha"
      output: "google-recaptcha"
    - name: "mikhailma/test-dataset"
      output: "test-dataset"
    - name: "cry2003/google-recaptcha-v2-images"
      output: "google-recaptcha-v2"

# 전처리 설정
preprocessing:
  target_size: [224, 224]
  color_space: "HSV"  # HSV or Lab
  apply_equalization: true
  equalization_method: "CLAHE"  # CLAHE or Histogram
  # K-fold는 train.py에서 수행하므로 여기서는 분할하지 않음

# 특징 추출 설정
features:
  hog:
    cell_size: [8, 8]
    block_size: [16, 16]
    nbins: 9
    enabled: true
  
  color_histogram:
    bins: 32
    enabled: true
  
  lbp:
    num_points: 24
    radius: 3
    enabled: true
  
  gradient:
    enabled: true
  
  texture:
    enabled: true

# 모델 학습 설정
training:
  # K-fold Cross Validation
  use_kfold: true
  k_fold: 5
  stratified: true
  
  # Feature Selection
  feature_selection:
    enabled: true
    method: "pca"  # pca, selectkbest, rfe
    n_components: 100  # PCA 사용 시
  
  # 분류기 선택 (여러 개 선택 가능)
  classifiers: ["svm", "random_forest", "xgboost", "lightgbm", "knn"]
  feature_type: "combined"  # hog, color_hist, lbp, combined
  
  # 하이퍼파라미터 최적화
  hyperparameter_tuning:
    enabled: true
    method: "grid"  # grid, random, bayesian
    cv: 3
  
  svm:
    kernel: "rbf"
    C: [0.1, 1.0, 10.0, 100.0]
    gamma: ["scale", "auto", 0.001, 0.01]
  
  random_forest:
    n_estimators: [50, 100, 200]
    max_depth: [None, 10, 20, 30]
    min_samples_split: [2, 5, 10]
  
  xgboost:
    n_estimators: 100
    max_depth: [3, 5, 7]
    learning_rate: [0.01, 0.1, 0.3]
  
  lightgbm:
    n_estimators: 100
    max_depth: [3, 5, 7]
    learning_rate: [0.01, 0.1, 0.3]
  
  knn:
    n_neighbors: [3, 5, 7, 9]
    weights: ["uniform", "distance"]
  
  logistic_regression:
    max_iter: 1000
    solver: "lbfgs"
    C: [0.1, 1.0, 10.0]
  
  # 앙상블
  ensemble:
    enabled: true
    method: "voting"  # voting, stacking
    classifiers: ["svm", "random_forest", "xgboost"]

# 시각화 설정
visualization:
  num_samples: 20
  dpi: 150
  figure_size: [24, 16]
```

---

## 🚀 실행 스크립트

### `scripts/run_pipeline.sh` - 전체 파이프라인

```bash
#!/bin/bash
# 전체 파이프라인 실행

# 1. 데이터 다운로드 (API 또는 수동 검증)
python src/download.py --mode auto  # API 사용
# 또는
python src/download.py --mode manual  # 수동 다운로드 검증

# 2. 전처리
python src/preprocess.py

# 3. 특징 추출
python src/feature_extraction.py --feature_type combined

# 4. 특징 선택 (선택적)
python src/feature_selection.py --method pca --n_components 0.95

# 5. 시각화
python src/visualization.py --num_samples 20

# 6. 모델 학습 (K-fold 포함)
python src/train.py --classifier svm --feature_type combined --kfold 5

# 7. 앙상블 (선택적)
python src/ensemble.py --models svm random_forest xgboost

# 8. 평가
python src/evaluate.py --model models/svm_combined_model.pkl
```

---

## 📊 예상 데이터 크기

- **원본 데이터**: ~500MB - 2GB (압축 해제 후)
- **전처리된 이미지**: ~1-3GB (JPEG, 224x224)
- **특징 벡터**: 
  - HOG만: ~140MB (9404 samples × 1764 dims × 8 bytes)
  - Combined: ~150MB (9404 samples × 2010 dims × 8 bytes)
- **학습된 모델**: ~10-50MB (분류기 종류에 따라)

---

## 🔍 주요 설계 결정사항

1. **모듈화**: Clean Code 원칙 준수 - 단일 책임, 명확한 함수명, DRY
2. **K-fold Cross-Validation**: 미리 분할하지 않고 학습 시 동적 분할로 일반화 성능 향상
3. **설정 파일**: YAML로 하이퍼파라미터 중앙 관리
4. **시각화**: 모든 단계에서 판단 가능한 시각 자료 생성
5. **성능 최적화**: Feature selection, Ensemble, Hyperparameter tuning 적용
6. **확장성**: 새로운 특징이나 분류기 추가 용이
7. **재현성**: 설정 파일과 시드 값으로 재현 가능
8. **플랫폼**: 맥북에서 실행 가능 (Colab 불필요, 단 GPU 가속 없음)

## 💻 실행 환경

**맥북에서 실행 가능**: 
- CPU 기반 ML 학습 (SVM, Random Forest 등)
- 소규모 데이터셋 (~10K 이미지) 처리 가능
- 특징 추출은 CPU로 충분히 가능
- 학습 시간: 분류기와 데이터 크기에 따라 수분~수십분

**Colab 사용 권장 경우**:
- 대규모 하이퍼파라미터 튜닝
- 여러 모델 동시 학습
- GPU 가속이 필요한 경우 (하지만 전통 ML은 CPU로 충분)

## 📐 Clean Code 6원칙

1. **Meaningful Names**: 변수/함수명이 의도를 명확히 표현
2. **Functions**: 작고 단일 책임, 한 가지 일만 수행
3. **Comments**: 코드로 설명되지 않는 부분만 주석
4. **Formatting**: 일관된 코드 스타일 (PEP 8)
5. **Error Handling**: 명확한 예외 처리 및 에러 메시지
6. **DRY (Don't Repeat Yourself)**: 중복 코드 제거, 재사용 가능한 함수 작성

---

## ✅ 체크리스트

이 구조로 진행하기 전에 확인할 사항:

- [ ] 디렉토리 구조가 명확한가?
- [ ] 각 모듈의 역할이 분명한가?
- [ ] 데이터 흐름이 논리적인가?
- [ ] 필요한 모든 기능이 포함되어 있는가?
- [ ] 확장 가능한 구조인가?

---

**이 구조로 진행할까요? 수정할 부분이 있으면 알려주세요!**
