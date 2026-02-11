# 🚨 치명적인 설계 포인트 (Critical Design Points)

## 📊 데이터 현황
- **총 이미지 수**: 약 55,000장
- **환경**: M1 MacBook 16GB RAM
- **데이터 구조**: 폴더별 클래스 구조 확인됨
  - `google-recaptcha/data/train/Bicycle/`, `Bus/`, `Car/` 등
  - `google-recaptcha-v2/images/Bicycle/`, `Bus/` 등

---

## 🚨 포인트 1: Data Leakage (데이터 오염) 방지

### 문제점
- **중복 이미지**: 동일한 사물을 각도만 살짝 바꾼 이미지가 train/test에 섞이면 모델이 '암기'하게 됨
- **결과**: 높은 점수지만 실제 캡차에서는 실패

### 해결 전략

#### 1. 중복 이미지 검출
```python
def detect_duplicate_images(image_paths: List[Path], 
                            similarity_threshold: float = 0.95) -> Dict:
    """
    이미지 해시 기반 중복 검출
    
    Args:
        image_paths: 이미지 경로 리스트
        similarity_threshold: 유사도 임계값 (0.95 = 95% 이상 유사)
    
    Returns:
        중복 그룹 딕셔너리
    """
    from PIL import Image
    import imagehash
    
    # Perceptual hash 계산
    image_hashes = {}
    for img_path in tqdm(image_paths, desc="Computing image hashes"):
        try:
            img = Image.open(img_path)
            phash = imagehash.phash(img)
            image_hashes[img_path] = phash
        except Exception as e:
            continue
    
    # 중복 그룹 찾기
    duplicate_groups = {}
    processed = set()
    
    for img_path, hash_val in image_hashes.items():
        if img_path in processed:
            continue
        
        duplicates = [img_path]
        for other_path, other_hash in image_hashes.items():
            if img_path != other_path and other_path not in processed:
                # 해시 거리 계산
                hamming_distance = hash_val - other_hash
                if hamming_distance <= 5:  # 임계값 조정 가능
                    duplicates.append(other_path)
                    processed.add(other_path)
        
        if len(duplicates) > 1:
            duplicate_groups[img_path] = duplicates
            processed.add(img_path)
    
    return duplicate_groups
```

#### 2. 엄격한 데이터 분할
```python
def strict_train_test_split(image_paths: List[Path], 
                           labels: List[str],
                           test_size: float = 0.2,
                           random_state: int = 42,
                           ensure_no_duplicates: bool = True) -> Tuple:
    """
    중복 이미지가 train/test에 섞이지 않도록 엄격한 분할
    
    Args:
        image_paths: 이미지 경로 리스트
        labels: 레이블 리스트
        test_size: 테스트 세트 비율
        random_state: 랜덤 시드
        ensure_no_duplicates: 중복 체크 여부
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    if ensure_no_duplicates:
        # 중복 그룹 확인
        duplicate_groups = detect_duplicate_images(image_paths)
        
        # 중복 그룹의 대표 이미지만 사용
        representative_images = set()
        for group in duplicate_groups.values():
            # 각 그룹에서 첫 번째 이미지만 선택
            representative_images.add(group[0])
        
        # 중복되지 않은 이미지만 필터링
        filtered_paths = []
        filtered_labels = []
        for path, label in zip(image_paths, labels):
            if path in representative_images or path not in duplicate_groups:
                filtered_paths.append(path)
                filtered_labels.append(label)
        
        image_paths = filtered_paths
        labels = filtered_labels
    
    # 클래스별로 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # 클래스 비율 유지
    )
    
    return X_train, X_test, y_train, y_test
```

#### 3. 파일명 기반 중복 체크
```python
def check_filename_duplicates(image_paths: List[Path]) -> Dict:
    """파일명 기반 중복 체크 (간단한 방법)"""
    filename_groups = {}
    
    for img_path in image_paths:
        filename = img_path.name
        if filename not in filename_groups:
            filename_groups[filename] = []
        filename_groups[filename].append(img_path)
    
    # 중복이 있는 파일명만 반환
    duplicates = {k: v for k, v in filename_groups.items() if len(v) > 1}
    return duplicates
```

### 구현 위치
- `src/preprocess/analyze.py`: 중복 이미지 검출 함수 추가
- `src/preprocess/main.py`: 전처리 파이프라인에 중복 체크 단계 추가
- `src/train.py`: 데이터 분할 시 중복 방지 적용

---

## 🚨 포인트 2: Feature Scaling (특징 스케일 통일)

### 문제점
- **HOG**: 0~0.2 사이의 작은 값
- **Color Histogram**: 수백~수천 단위의 큰 값
- **LBP**: 0~1 사이의 정규화된 값
- **결과**: 모델이 큰 값만 중요하게 여기고 HOG의 형태 정보 무시

### 해결 전략

#### 1. 특징별 개별 정규화 후 결합
```python
class FeatureScaler:
    """특징별로 적절한 스케일링 적용"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_ranges = {
            'hog': (0, 0.2),  # HOG는 작은 값
            'color_hist': (0, 10000),  # Color Histogram은 큰 값
            'lbp': (0, 1),  # LBP는 이미 정규화됨
            'gradient': (0, 255),  # Gradient는 픽셀 값 범위
            'texture': (0, 1)  # Texture는 정규화됨
        }
    
    def fit_transform(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        특징별로 스케일링 후 결합
        
        Args:
            features_dict: {'hog': array, 'color_hist': array, ...}
        
        Returns:
            스케일링된 결합 특징 벡터
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        scaled_features = []
        
        for feat_name, feat_array in features_dict.items():
            if feat_name == 'hog':
                # HOG는 MinMaxScaler로 0-1 범위로 정규화
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(feat_array.reshape(-1, 1)).flatten()
            elif feat_name == 'color_hist':
                # Color Histogram은 StandardScaler로 표준화
                scaler = StandardScaler()
                scaled = scaler.fit_transform(feat_array.reshape(-1, 1)).flatten()
            elif feat_name in ['lbp', 'texture']:
                # 이미 정규화된 특징은 그대로 사용
                scaled = feat_array
            else:
                # 기본적으로 StandardScaler 사용
                scaler = StandardScaler()
                scaled = scaler.fit_transform(feat_array.reshape(-1, 1)).flatten()
            
            self.scalers[feat_name] = scaler
            scaled_features.append(scaled)
        
        return np.concatenate(scaled_features)
```

#### 2. 결합 후 전체 스케일링 (권장)
```python
def scale_combined_features(features: np.ndarray, 
                          method: str = 'standard') -> Tuple[np.ndarray, object]:
    """
    결합된 특징 벡터에 전체 스케일링 적용
    
    Args:
        features: 결합된 특징 벡터 (N, 2000)
        method: 'standard', 'minmax', 'robust'
    
    Returns:
        (scaled_features, scaler_object)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()  # 이상치에 강함
    else:
        scaler = StandardScaler()
    
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, scaler
```

### 구현 위치
- `src/feature_extraction/combined_extractor.py`: 특징 결합 시 스케일링
- `src/train.py`: 학습 전 특징 스케일링 (config에서 설정)
- **중요**: Scaler 객체를 저장하여 추론 시 동일하게 적용

### 저장 구조
```
models/
├── svm_combined_model.pkl
├── svm_combined_scaler.pkl  # ⚠️ 필수! 추론 시 사용
└── ...
```

---

## 🚨 포인트 3: PCA 차원 축소 (Explained Variance 기반)

### 문제점
- **임의 차원 축소**: "50차원으로 줄여줘" → 정보 손실
- **과도한 축소**: 중요한 특징 제거
- **부족한 축소**: PCA 의미 없음

### 해결 전략

#### 1. Explained Variance Ratio 분석
```python
def find_optimal_pca_dimensions(features: np.ndarray, 
                               variance_threshold: float = 0.95) -> Dict:
    """
    Explained Variance Ratio를 기반으로 최적 PCA 차원 수 찾기
    
    Args:
        features: 특징 벡터 (N, D)
        variance_threshold: 보존할 분산 비율 (0.95 = 95%)
    
    Returns:
        {'optimal_dim': int, 'variance_ratio': array, 'plot_data': dict}
    """
    from sklearn.decomposition import PCA
    
    # 전체 차원으로 PCA 수행
    pca_full = PCA()
    pca_full.fit(features)
    
    # 누적 설명 분산 계산
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # 임계값을 만족하는 최소 차원 찾기
    optimal_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # 시각화 데이터
    plot_data = {
        'components': range(1, min(100, len(cumulative_variance)) + 1),  # 최대 100개까지만
        'variance_ratio': pca_full.explained_variance_ratio_[:100],
        'cumulative_variance': cumulative_variance[:100],
        'optimal_dim': optimal_dim,
        'variance_at_optimal': cumulative_variance[optimal_dim - 1]
    }
    
    return {
        'optimal_dim': optimal_dim,
        'variance_ratio': pca_full.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance,
        'plot_data': plot_data
    }
```

#### 2. PCA 차원 결정 시각화
```python
def visualize_pca_analysis(pca_results: Dict, output_path: Path):
    """PCA 분석 결과 시각화"""
    import matplotlib.pyplot as plt
    
    plot_data = pca_results['plot_data']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Explained Variance Ratio
    ax = axes[0]
    ax.plot(plot_data['components'], plot_data['variance_ratio'][:len(plot_data['components'])], 
            'b-o', markersize=3)
    ax.axvline(plot_data['optimal_dim'], color='r', linestyle='--', 
               label=f"Optimal: {plot_data['optimal_dim']} dims")
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance Ratio by Component')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative Explained Variance
    ax = axes[1]
    ax.plot(plot_data['components'], plot_data['cumulative_variance'][:len(plot_data['components'])], 
            'g-o', markersize=3)
    ax.axhline(0.95, color='r', linestyle='--', label='95% Threshold')
    ax.axvline(plot_data['optimal_dim'], color='r', linestyle='--', 
               label=f"Optimal: {plot_data['optimal_dim']} dims")
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title(f'Cumulative Explained Variance (Optimal: {plot_data["optimal_dim"]} dims, '
                 f'{plot_data["variance_at_optimal"]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

#### 3. PCA 적용
```python
def apply_pca_with_optimal_dim(features: np.ndarray, 
                               variance_threshold: float = 0.95) -> Tuple:
    """
    최적 차원으로 PCA 적용
    
    Returns:
        (transformed_features, pca_object, optimal_dim)
    """
    from sklearn.decomposition import PCA
    
    # 최적 차원 찾기
    pca_analysis = find_optimal_pca_dimensions(features, variance_threshold)
    optimal_dim = pca_analysis['optimal_dim']
    
    print(f"Optimal PCA dimensions: {optimal_dim} (preserves "
          f"{pca_analysis['cumulative_variance'][optimal_dim - 1]:.2%} variance)")
    
    # PCA 적용
    pca = PCA(n_components=optimal_dim)
    transformed_features = pca.fit_transform(features)
    
    return transformed_features, pca, optimal_dim
```

### 구현 위치
- `src/feature_selection/pca_analysis.py`: PCA 분석 모듈
- `src/feature_selection/main.py`: PCA 적용 파이프라인
- `config/config.yaml`: `variance_threshold` 설정 추가

---

## ✅ 체크리스트

### 1. 데이터 구조 파악 ✅
- [x] 폴더별 클래스 구조 확인됨
  - `google-recaptcha/data/train/Bicycle/`, `Bus/`, `Car/` 등
  - `google-recaptcha-v2/images/Bicycle/`, `Bus/` 등

### 2. 클래스 비율 확인 ⚠️ (구현 필요)
```python
def analyze_class_distribution(data_dir: Path) -> Dict:
    """클래스별 이미지 개수 및 비율 분석"""
    class_counts = {}
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            class_counts[class_dir.name] = len(images)
    
    total = sum(class_counts.values())
    class_ratios = {k: v/total for k, v in class_counts.items()}
    
    return {
        'counts': class_counts,
        'ratios': class_ratios,
        'total': total,
        'is_balanced': max(class_ratios.values()) / min(class_ratios.values()) < 2.0
    }
```

### 3. HOG 파라미터 결정 ⚠️ (112x112 이미지 기준)
```python
# 112x112 이미지에 최적화된 HOG 파라미터
HOG_CONFIG_112x112 = {
    'cell_size': (8, 8),      # 112 / 8 = 14 cells per dimension
    'block_size': (2, 2),     # 2x2 cells per block
    'block_stride': (1, 1),   # 50% overlap
    'nbins': 9,               # 9 orientation bins
    'win_size': (112, 112),   # 전체 이미지 크기
    
    # 예상 차원: (14-1) * (14-1) * 2 * 2 * 9 = 13 * 13 * 36 = 6,084
    # 실제로는 약 1,764 차원 (OpenCV HOG 구현에 따라 다름)
}
```

---

## 📝 구현 우선순위

### Phase 1: 즉시 구현 (치명적 포인트)
1. ✅ 중복 이미지 검출 모듈
2. ✅ 엄격한 데이터 분할 함수
3. ✅ Feature Scaling 적용
4. ✅ PCA 분석 모듈

### Phase 2: 검증 및 시각화
1. 클래스 비율 분석 및 시각화
2. HOG 파라미터 최적화
3. 스케일링 전/후 비교 시각화

### Phase 3: 통합
1. 전처리 파이프라인에 중복 체크 통합
2. 특징 추출 시 자동 스케일링
3. PCA 분석 결과 리포트 생성

---

## 🎯 최종 권장사항

1. **데이터 분할 전**: 반드시 중복 이미지 검출 및 제거
2. **특징 추출 후**: 반드시 Feature Scaling 적용 (StandardScaler 권장)
3. **PCA 적용 전**: 반드시 Explained Variance Ratio 분석
4. **모델 저장 시**: Scaler와 PCA 객체도 함께 저장
5. **추론 시**: 학습 시와 동일한 Scaler/PCA 적용
