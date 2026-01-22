# 전처리 파이프라인 상세 설명

## 📊 현재 상황

### 데이터 품질 검사 결과
- **총 이미지**: 55,729장
- **중복 이미지**: 25,415장 (45.6%)
- **유효 이미지**: 약 30,314장 (중복 제거 후)
- **클래스 수**: 15개 (실제로는 14개, "visualizations" 제외)
- **클래스 불균형**: 심각 (Balance Ratio: 1,791.5)

---

## 🔄 전처리 파이프라인 전체 흐름

```
원본 데이터 (data/raw/)
    ↓
[Step 0] 데이터 품질 검사 ✅ (완료)
    ├─ 중복 이미지 검출
    ├─ 클래스 분포 분석
    └─ 결과: quality_checks/
    ↓
[Step 1] 데이터셋 구조 분석
    ├─ 이미지 크기 분포
    ├─ 밝기 분포
    ├─ 노이즈 레벨 측정
    └─ 결과: analysis/
    ↓
[Step 2] 이미지 리사이징 (112x112)
    ├─ 모든 이미지를 112x112로 통일
    ├─ 메모리 효율을 위한 크기 조정
    └─ 결과: resized/
    ↓
[Step 3] 노이즈 제거 (조건부)
    ├─ 노이즈 레벨이 높은 이미지만 처리
    ├─ 가우시안 블러 또는 Bilateral Filter
    └─ 결과: denoised/
    ↓
[Step 4] CLAHE (대비 향상)
    ├─ 어두운 이미지의 대비 개선
    ├─ 적응적 파라미터 조정
    └─ 결과: clahe/
    ↓
[Step 5] 색상 공간 변환
    ├─ HSV 변환 (Color Histogram용)
    ├─ Grayscale 변환 (HOG, LBP용)
    └─ 결과: color_spaces/
    ↓
[Step 6] 감마 교정 (조건부)
    ├─ 어두운 이미지만 밝게 조정
    └─ 결과: gamma_corrected/
    ↓
[Step 7] 정규화
    ├─ 픽셀 값을 0~1 범위로 변환
    ├─ float32 형식으로 저장 (.npy)
    └─ 결과: normalized/
    ↓
최종 전처리된 이미지 준비 완료!
```

---

## 📝 각 단계 상세 설명

### **Step 0: 데이터 품질 검사** ✅ (완료)

**목적**: 데이터셋의 품질 문제를 사전에 발견

**수행 작업**:
1. **중복 이미지 검출**
   - Perceptual Hash로 동일/유사 이미지 찾기
   - 결과: 13,366개 그룹, 25,415장 중복 발견

2. **클래스 분포 분석**
   - 각 클래스별 이미지 개수 확인
   - 불균형 정도 측정 (Balance Ratio: 1,791.5)

**결과물**:
- `quality_checks/class_distribution.json`: 클래스별 통계
- `quality_checks/duplicate_groups.json`: 중복 그룹 목록
- `quality_checks/class_distribution.png`: 시각화

**다음 액션**:
- 중복 이미지 제거 (25,415장 삭제)
- "visualizations" 폴더 제외
- Mountain 클래스 처리 결정 (19장은 너무 적음)

---

### **Step 1: 데이터셋 구조 분석**

**목적**: 전처리 파라미터를 최적화하기 위한 데이터 특성 파악

**수행 작업**:
1. **이미지 크기 분포**
   - 각 이미지의 width, height 측정
   - 평균 크기, 최소/최대 크기 확인

2. **밝기 분포**
   - 각 이미지의 평균 밝기 계산
   - 어두운 이미지 비율 확인

3. **노이즈 레벨 측정**
   - 각 이미지의 표준편차 계산
   - 노이즈가 많은 이미지 비율 확인

**결과물**:
- `analysis/dataset_analysis.json`: 통계 데이터
- `analysis/dataset_analysis.png`: 시각화
- **권장사항**:
  - 노이즈 제거 임계값 제안
  - CLAHE 파라미터 제안
  - 감마 교정 필요 여부 제안

**예상 결과 예시**:
```json
{
  "image_sizes": {
    "mean_width": 112,
    "mean_height": 112,
    "min_width": 64,
    "max_width": 256
  },
  "brightness": {
    "mean": 120.5,
    "std": 35.2,
    "dark_images_ratio": 0.15
  },
  "noise_levels": {
    "mean": 18.3,
    "high_noise_ratio": 0.25
  },
  "recommendations": {
    "noise_reduction_needed": true,
    "clahe_needed": true,
    "gamma_correction_needed": false
  }
}
```

---

### **Step 2: 이미지 리사이징 (112x112)**

**목적**: 모든 이미지를 동일한 크기로 통일

**왜 필요한가?**
- HOG 특징 추출 시 일관된 차원 보장
- 메모리 효율성 (112x112 = 12,544 픽셀)
- 처리 속도 향상

**수행 작업**:
1. 모든 이미지를 112x112로 리사이징
2. Bilinear 보간법 사용 (품질 유지)
3. 원본 비율 유지 또는 자르기 (설정 가능)

**결과물**:
- `resized/`: 모든 이미지가 112x112 크기
- `resized/visualizations/resize_comparison.png`: 전/후 비교

**메모리 사용량**:
- 원본 (다양한 크기): 평균 50KB × 30,314장 ≈ 1.5GB
- 리사이징 후 (112x112): 12.5KB × 30,314장 ≈ 380MB
- **절약**: 약 1.1GB

---

### **Step 3: 노이즈 제거 (조건부)**

**목적**: 디지털 노이즈 제거로 특징 추출 품질 향상

**언제 적용?**
- Step 1 분석 결과 노이즈 레벨이 높은 경우
- 표준편차가 15.0 이상인 이미지만 처리

**수행 작업**:
1. 각 이미지의 노이즈 레벨 측정
2. 임계값 이상인 이미지만 처리
3. 가우시안 블러 (3x3 커널) 또는 Bilateral Filter 적용

**결과물**:
- `denoised/`: 노이즈가 제거된 이미지
- `denoised/visualizations/noise_reduction_comparison.png`: 전/후 비교

**예상 처리량**:
- 전체 이미지: 30,314장
- 노이즈 제거 필요: 약 25% (7,578장)
- 처리 시간: 약 2-3분

---

### **Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)**

**목적**: 어두운 이미지의 대비 개선

**왜 필요한가?**
- 어두운 이미지에서 HOG 특징 추출이 어려움
- 엣지 정보를 더 명확하게 만들기 위함

**수행 작업**:
1. 각 이미지의 평균 밝기 측정
2. 어두운 이미지(밝기 < 100)는 clip_limit 증가
3. 작은 이미지는 tile_grid_size 조정

**결과물**:
- `clahe/`: 대비가 개선된 이미지
- `clahe/visualizations/clahe_comparison.png`: 전/후 비교

**적응적 파라미터**:
- 밝은 이미지: clip_limit=2.0, tile=(8,8)
- 어두운 이미지: clip_limit=3.0, tile=(8,8)
- 작은 이미지: tile=(4,4)

---

### **Step 5: 색상 공간 변환**

**목적**: 특징 추출을 위한 다양한 색상 표현 준비

**수행 작업**:
1. **HSV 변환**
   - Color Histogram 특징 추출용
   - 조명 변화에 강한 색상 정보

2. **Grayscale 변환**
   - HOG, LBP, Texture 특징 추출용
   - 형태 정보에 집중

3. **Lab 변환** (선택적)
   - 더 정확한 색상 인식

**결과물**:
- `color_spaces/hsv/`: HSV 변환 이미지
- `color_spaces/grayscale/`: Grayscale 이미지
- `color_spaces/visualizations/color_space_conversions.png`: 비교 시각화

**저장 구조**:
```
color_spaces/
├── hsv/
│   ├── Bicycle/
│   │   └── image1.png
│   └── Bus/
│       └── image2.png
└── grayscale/
    ├── Bicycle/
    │   └── image1.png
    └── Bus/
        └── image2.png
```

---

### **Step 6: 감마 교정 (조건부)**

**목적**: 어두운 부분의 디테일 강조

**언제 적용?**
- 평균 밝기가 100 이하인 이미지만
- Step 1 분석 결과 필요하다고 판단된 경우

**수행 작업**:
1. 각 이미지의 평균 밝기 측정
2. 어두운 이미지에 감마 교정 적용 (γ=0.8)
3. 밝은 이미지는 그대로 유지

**결과물**:
- `gamma_corrected/`: 감마 교정된 이미지
- `gamma_corrected/visualizations/gamma_correction_comparison.png`: 전/후 비교

**감마 값 의미**:
- γ < 1: 어두운 부분을 밝게 (0.8 권장)
- γ > 1: 밝은 부분을 어둡게
- γ = 1: 변화 없음

---

### **Step 7: 정규화**

**목적**: 픽셀 값을 0~1 범위로 통일하여 모델 학습 안정화

**수행 작업**:
1. **Min-Max 정규화**
   - 픽셀 값: 0~255 → 0.0~1.0
   - 공식: `(pixel - min) / (max - min)`

2. **float32 형식으로 저장**
   - `.npy` 파일로 저장 (NumPy 배열)
   - 메모리 효율적 로딩 가능

**결과물**:
- `normalized/`: 정규화된 이미지 (.npy 파일)
- `normalized/visualizations/normalization_comparison.png`: 전/후 비교

**저장 형식**:
```
normalized/
├── Bicycle/
│   └── image1.npy  (112, 112, 3) float32
└── Bus/
    └── image2.npy  (112, 112, 3) float32
```

**메모리 사용량**:
- float32: 4바이트/픽셀
- 112×112×3 = 37,632 픽셀
- 37,632 × 4 = 150KB/이미지
- 총: 150KB × 30,314장 ≈ 4.5GB (하지만 디스크에만 저장)

---

## 🎯 전처리 후 최종 결과

### 데이터 구조
```
data/processed/
├── quality_checks/          # Step 0 결과
├── analysis/                # Step 1 결과
├── resized/                 # Step 2 결과
├── denoised/                # Step 3 결과
├── clahe/                   # Step 4 결과
├── color_spaces/            # Step 5 결과
│   ├── hsv/
│   └── grayscale/
├── gamma_corrected/         # Step 6 결과
└── normalized/              # Step 7 결과 (최종)
```

### 최종 데이터 특성
- **이미지 수**: 약 30,314장 (중복 제거 후)
- **크기**: 모두 112×112 픽셀
- **형식**: float32 NumPy 배열 (.npy)
- **값 범위**: 0.0 ~ 1.0
- **색상 공간**: HSV, Grayscale 준비 완료

---

## ⚠️ 주의사항

### 1. 중복 제거 적용
현재는 중복을 **검출만** 했습니다. 실제로 제거하려면:
- `duplicate_groups.json`을 읽어서
- 각 그룹에서 첫 번째 이미지만 남기고
- 나머지는 삭제하거나 별도 폴더로 이동

### 2. 클래스 불균형 처리
전처리 후에 별도로 처리:
- **Mountain (19장)**: 제외 또는 증강
- **"visualizations" 폴더**: 제외
- **SMOTE/Undersampling**: 특징 추출 후 적용

### 3. 메모리 관리
- 각 단계는 Generator 패턴으로 하나씩 처리
- 중간 결과는 디스크에 저장
- 최종 결과만 메모리에 로드

---

## 🚀 실행 방법

### 전체 파이프라인 실행
```bash
python src/preprocess/main.py
```

### 단계별 실행
```bash
# Step 1: 구조 분석
python src/preprocess/main.py --step analyze

# Step 2: 리사이징
python src/preprocess/main.py --step resize

# Step 3: 노이즈 제거
python src/preprocess/main.py --step noise_reduction

# ... (각 단계별로 실행 가능)
```

### 예상 소요 시간
- Step 1 (분석): 약 5분
- Step 2 (리사이징): 약 3분
- Step 3 (노이즈 제거): 약 2분
- Step 4 (CLAHE): 약 3분
- Step 5 (색상 변환): 약 2분
- Step 6 (감마 교정): 약 1분
- Step 7 (정규화): 약 2분
- **총 예상 시간**: 약 18분 (30,314장 기준)

---

## 📊 전처리 전/후 비교

| 항목 | 전처리 전 | 전처리 후 |
|------|----------|----------|
| 이미지 수 | 55,729장 | 30,314장 (중복 제거) |
| 크기 | 다양함 | 모두 112×112 |
| 형식 | PNG/JPG | float32 .npy |
| 값 범위 | 0~255 | 0.0~1.0 |
| 노이즈 | 있음 | 제거됨 |
| 대비 | 불균일 | 개선됨 |
| 색상 공간 | RGB만 | HSV, Grayscale 준비 |

---

## ✅ 다음 단계

전처리가 완료되면:
1. **특징 추출**: HOG, Color Histogram, LBP 등
2. **특징 스케일링**: StandardScaler 적용
3. **PCA 분석**: 최적 차원 찾기
4. **모델 학습**: K-fold Cross-Validation
