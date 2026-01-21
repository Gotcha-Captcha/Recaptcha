# 특징 추출 모듈 설계 (메모리 효율성 중심)

## 📊 메모리 및 성능 분석

### 메모리 사용량 계산
- **특징 벡터 크기**: 2,000차원 (float32)
- **이미지 수**: 55,000장
- **특징 벡터 메모리**: 55,000 × 2,000 × 4바이트 ≈ **440MB** ✅ (16GB RAM에서 충분)

### 성능 분석
- **특징 추출 시간**: 장당 0.01초
- **전체 소요 시간**: 55,000 × 0.01초 ≈ **550초 (약 9분)**

### 핵심 전략
1. **Generator 패턴**: 이미지를 하나씩 로드 → 처리 → 해제
2. **캐싱**: 추출한 특징을 `.npy` 또는 `.joblib`로 저장
3. **재사용**: 모델 학습 시에는 저장된 특징만 로드

---

## 🏗️ 모듈 구조

```
src/feature_extraction/
├── __init__.py
├── utils.py              # 공통 유틸리티 (Generator, 캐싱 헬퍼)
├── hog_extractor.py      # HOG 특징 추출
├── color_extractor.py    # Color Histogram 특징 추출
├── lbp_extractor.py      # LBP 특징 추출
├── gradient_extractor.py # Gradient 특징 추출
├── texture_extractor.py   # Texture (GLCM) 특징 추출
├── combined_extractor.py # 모든 특징 결합
└── main.py               # 메인 파이프라인
```

---

## 🔄 데이터 흐름 (Generator 패턴)

### 1. 이미지 로딩 (Generator)
```python
def image_generator(image_paths: List[Path]):
    """이미지를 하나씩 로드하는 Generator"""
    for img_path in image_paths:
        img = load_image(img_path)  # 하나씩 로드
        if img is not None:
            yield img, img_path
        # 메모리에서 자동 해제
```

### 2. 특징 추출 (하나씩 처리)
```python
def extract_features_generator(image_generator, extractors):
    """Generator로 특징 추출"""
    for img, img_path in image_generator:
        features = {}
        for extractor in extractors:
            features[extractor.name] = extractor.extract(img)
        yield features, img_path
```

### 3. 캐싱 (배치 저장)
```python
def save_features_batch(features_batch, labels_batch, output_path):
    """배치 단위로 특징 저장 (메모리 효율)"""
    # 1000개씩 모아서 저장
    features_array = np.array(features_batch, dtype=np.float32)
    labels_array = np.array(labels_batch, dtype=np.int32)
    
    np.save(output_path / "features.npy", features_array)
    np.save(output_path / "labels.npy", labels_array)
```

---

## 💾 캐싱 전략

### 저장 형식
1. **`.npy` 파일** (권장)
   - NumPy 배열 직접 저장
   - 빠른 로딩 속도
   - 메모리 매핑 가능 (`mmap_mode='r'`)

2. **`.joblib` 파일** (대용량 데이터)
   - 압축 지원
   - 큰 배열에 유리

### 저장 구조
```
data/features/
├── combined/
│   ├── train_features.npy      # (N, 2000) float32
│   ├── train_labels.npy        # (N,) int32
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│   └── metadata.json            # 특징 정보 (차원, 통계 등)
├── hog/
│   ├── train_features.npy       # (N, 1764) float32
│   └── ...
└── ...
```

### 캐시 검증
```python
def check_cache_exists(output_dir: Path, feature_type: str) -> bool:
    """캐시 파일 존재 여부 확인"""
    features_path = output_dir / f"{feature_type}_features.npy"
    labels_path = output_dir / f"{feature_type}_labels.npy"
    return features_path.exists() and labels_path.exists()

def load_cached_features(output_dir: Path, feature_type: str):
    """캐시된 특징 로드 (메모리 매핑)"""
    features = np.load(output_dir / f"{feature_type}_features.npy", 
                       mmap_mode='r')  # 메모리 매핑으로 로드
    labels = np.load(output_dir / f"{feature_type}_labels.npy")
    return features, labels
```

---

## 🎯 특징 추출 모듈 설계

### 1. HOG Extractor
```python
class HOGExtractor:
    def __init__(self, cell_size=(8, 8), block_size=(16, 16), nbins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
        self.feature_dim = 1764  # 계산된 차원
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """HOG 특징 추출 (하나의 이미지)"""
        # Grayscale 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # HOG 계산
        hog = cv2.HOGDescriptor(
            _winSize=(gray.shape[1], gray.shape[0]),
            _blockSize=(self.block_size[0] * self.cell_size[0], 
                       self.block_size[1] * self.cell_size[1]),
            _blockStride=(self.cell_size[0], self.cell_size[1]),
            _cellSize=self.cell_size,
            _nbins=self.nbins
        )
        features = hog.compute(gray)
        return features.flatten().astype(np.float32)
```

### 2. Color Histogram Extractor
```python
class ColorHistogramExtractor:
    def __init__(self, bins=32, color_space='hsv'):
        self.bins = bins
        self.color_space = color_space
        self.feature_dim = bins * 3  # RGB/HSV 각 채널
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Color Histogram 특징 추출"""
        if self.color_space == 'hsv':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            img = image
        
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [self.bins], [0, 256])
            hist_features.append(hist.flatten())
        
        return np.concatenate(hist_features).astype(np.float32)
```

### 3. LBP Extractor
```python
class LBPExtractor:
    def __init__(self, num_points=24, radius=3):
        self.num_points = num_points
        self.radius = radius
        self.feature_dim = 26  # 히스토그램 빈 수
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """LBP 특징 추출"""
        from skimage import feature
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        lbp = feature.local_binary_pattern(gray, self.num_points, 
                                           self.radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=self.num_points + 2, 
                              range=(0, self.num_points + 2))
        return (hist / hist.sum()).astype(np.float32)  # 정규화
```

### 4. Combined Extractor
```python
class CombinedExtractor:
    def __init__(self, config):
        self.extractors = {
            'hog': HOGExtractor(**config['hog']),
            'color_hist': ColorHistogramExtractor(**config['color_histogram']),
            'lbp': LBPExtractor(**config['lbp']),
            'gradient': GradientExtractor(**config['gradient']),
            'texture': TextureExtractor(**config['texture'])
        }
        self.feature_dim = sum(e.feature_dim for e in self.extractors.values())
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """모든 특징 결합"""
        features = []
        for name, extractor in self.extractors.items():
            if config['features'][name]['enabled']:
                feat = extractor.extract(image)
                features.append(feat)
        return np.concatenate(features).astype(np.float32)
```

---

## 🚀 메인 파이프라인

### 특징 추출 (Generator + 캐싱)
```python
def extract_features_pipeline(config, split='train', 
                              use_cache=True, batch_size=1000):
    """
    특징 추출 파이프라인
    
    Args:
        config: 설정 딕셔너리
        split: 데이터 분할 (train/val/test)
        use_cache: 캐시 사용 여부
        batch_size: 배치 저장 크기
    """
    # 경로 설정
    processed_dir = Path(config['data']['processed_dir'])
    features_dir = Path(config['data']['features_dir'])
    feature_type = config['training']['feature_type']
    
    output_dir = features_dir / feature_type / split
    cache_path = output_dir / f"{split}_features.npy"
    
    # 캐시 확인
    if use_cache and cache_path.exists():
        print(f"✓ Loading cached features from {cache_path}")
        return load_cached_features(output_dir, split)
    
    # 이미지 경로 수집
    image_paths, labels = collect_image_paths(processed_dir, split)
    
    # Generator 생성
    image_gen = image_generator(image_paths)
    
    # 특징 추출기 초기화
    extractor = CombinedExtractor(config['features'])
    
    # 배치 처리
    features_batch = []
    labels_batch = []
    
    print(f"Extracting features for {split} split...")
    for idx, (img, img_path) in enumerate(tqdm(image_gen, total=len(image_paths))):
        # 특징 추출
        features = extractor.extract(img)
        label = labels[idx]
        
        features_batch.append(features)
        labels_batch.append(label)
        
        # 배치 저장
        if len(features_batch) >= batch_size:
            save_features_batch(features_batch, labels_batch, output_dir, append=True)
            features_batch = []
            labels_batch = []
    
    # 남은 배치 저장
    if features_batch:
        save_features_batch(features_batch, labels_batch, output_dir, append=True)
    
    # 최종 통합 및 저장
    final_features, final_labels = load_and_merge_batches(output_dir)
    np.save(cache_path, final_features)
    np.save(output_dir / f"{split}_labels.npy", final_labels)
    
    return final_features, final_labels
```

---

## 📈 메모리 효율성 체크리스트

### ✅ 현재 구현 확인
- [x] Generator 패턴으로 이미지 하나씩 로드
- [x] 처리 후 즉시 메모리 해제
- [ ] 특징 추출 시 배치 저장
- [ ] 캐시 검증 및 재사용
- [ ] 메모리 매핑으로 대용량 특징 로드

### 🔧 개선 사항
1. **배치 저장**: 1000개씩 모아서 저장 (메모리 효율)
2. **메모리 매핑**: `mmap_mode='r'`로 특징 로드 (대용량 데이터)
3. **캐시 검증**: 해시 기반 캐시 무효화 (설정 변경 시)
4. **진행 상황 저장**: 중단 시 재개 가능

---

## 🎨 시각화 전략

### 특징 시각화 (샘플링)
- 전체 특징을 메모리에 올리지 않고 **샘플링**하여 시각화
- 예: 1000개 샘플만 로드하여 히트맵 생성

```python
def visualize_features_sampled(features_path, num_samples=1000):
    """샘플링하여 특징 시각화"""
    # 메모리 매핑으로 로드
    features = np.load(features_path, mmap_mode='r')
    
    # 샘플링
    indices = np.random.choice(len(features), num_samples, replace=False)
    sampled_features = features[indices]
    
    # 시각화
    visualize_feature_heatmap(sampled_features)
```

---

## 📝 구현 우선순위

### Phase 1: 기본 구현
1. Generator 패턴 이미지 로더
2. 각 특징 추출기 구현
3. 배치 저장 기능

### Phase 2: 캐싱 및 최적화
1. 캐시 검증 및 재사용
2. 메모리 매핑 로더
3. 진행 상황 저장

### Phase 3: 고급 기능
1. 특징 선택 (PCA, SelectKBest)
2. 특징 정규화
3. 특징 시각화
