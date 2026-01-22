#!/bin/bash
# CLAHE → Gaussian 파이프라인 실행 스크립트
# 기존 resize 결과를 사용하여 새로운 파이프라인 적용

echo "============================================================"
echo "CLAHE → Gaussian 파이프라인 실행"
echo "============================================================"
echo ""

# Step 1: CLAHE 적용 (resize 결과 사용)
echo "[Step 1] CLAHE 적용..."
python src/preprocess/main.py --step clahe

if [ $? -ne 0 ]; then
    echo "❌ CLAHE 단계 실패"
    exit 1
fi

echo ""
echo "✓ CLAHE 완료"
echo ""

# Step 2: Gaussian Blur 적용 (CLAHE 결과 사용)
echo "[Step 2] Gaussian Blur 적용..."
python src/preprocess/main.py --step noise_reduction

if [ $? -ne 0 ]; then
    echo "❌ Noise Reduction 단계 실패"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ 파이프라인 완료! (v2: CLAHE → Gaussian)"
echo "============================================================"
echo ""
echo "결과 위치 (v2):"
echo "  - CLAHE: data/processed/clahe_v2/"
echo "  - Denoised: data/processed/denoised_v2/"
echo "  - 시각화: data/visualization/clahe_v2/, data/visualization/noise_reduction_v2/"
echo ""
echo "기존 결과 (유지됨):"
echo "  - CLAHE: data/processed/clahe/"
echo "  - Denoised: data/processed/denoised/"
echo "  - 시각화: data/visualization/clahe/, data/visualization/noise_reduction/"
