"""
Main preprocessing pipeline.
Orchestrates all preprocessing steps sequentially.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocess.utils import load_config, ensure_dir
from src.preprocess.data_quality import (
    detect_duplicate_images, 
    check_filename_duplicates,
    analyze_class_distribution,
    visualize_class_distribution,
    get_representative_images
)
from src.preprocess.analyze import analyze_dataset_structure
from src.preprocess.resize import resize_dataset
from src.preprocess.noise_reduction import reduce_noise_dataset
from src.preprocess.clahe import apply_clahe_dataset
from src.preprocess.color_space import convert_color_spaces_dataset
from src.preprocess.gamma_correction import apply_gamma_correction_dataset
from src.preprocess.normalize import normalize_dataset


def run_preprocessing_pipeline(config_path: Path, 
                               step: Optional[str] = None,
                               skip_analysis: bool = False) -> dict:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config_path: Path to config.yaml
        step: Specific step to run (None = all steps)
        skip_analysis: Whether to skip dataset analysis
        
    Returns:
        Dictionary with pipeline results
    """
    # Load configuration
    config = load_config(config_path)
    preprocess_config = config['preprocessing']
    vis_config = config['visualization']
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / config['data']['raw_dir']
    processed_dir = project_root / config['data']['processed_dir']
    
    print("\n" + "="*60)
    print("reCAPTCHA Preprocessing Pipeline")
    print("="*60)
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Processed data directory: {processed_dir}")
    print("="*60)
    
    results = {}
    current_dir = raw_data_dir
    
    # Step 0: Data Quality Checks (🚨 Critical!)
    if (step is None or step == "data_quality") and config.get('data_quality', {}).get('duplicate_detection', {}).get('enabled', True):
        print("\n[Step 0] Data Quality Checks")
        quality_output = processed_dir / "quality_checks"
        ensure_dir(quality_output)
        
        # Find all images
        import os
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        all_image_paths = []
        for root, dirs, files in os.walk(raw_data_dir):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    all_image_paths.append(Path(root) / f)
        
        print(f"Found {len(all_image_paths)} images for quality checks")
        
        # 1. Class Distribution Analysis
        print("\n--- Analyzing Class Distribution ---")
        class_dist = analyze_class_distribution(raw_data_dir)
        visualize_class_distribution(class_dist, quality_output)
        results['class_distribution'] = class_dist
        
        # Save class distribution
        import json
        with open(quality_output / "class_distribution.json", 'w') as f:
            json.dump(class_dist, f, indent=2, default=str)
        
        # 2. Duplicate Detection
        quality_config = config.get('data_quality', {})
        dup_config = quality_config.get('duplicate_detection', {})
        
        if dup_config.get('enabled', True):
            print("\n--- Detecting Duplicate Images ---")
            
            # Filename-based duplicate check (fast)
            filename_duplicates = check_filename_duplicates(all_image_paths)
            if filename_duplicates:
                print(f"Found {len(filename_duplicates)} files with duplicate filenames")
                with open(quality_output / "filename_duplicates.json", 'w') as f:
                    json.dump({k: [str(p) for p in v] for k, v in filename_duplicates.items()}, 
                             f, indent=2)
            
            # Perceptual hash-based duplicate check
            duplicate_groups = detect_duplicate_images(
                all_image_paths,
                similarity_threshold=dup_config.get('similarity_threshold', 0.95),
                method=dup_config.get('method', 'hash')
            )
            
            if duplicate_groups:
                print(f"Found {len(duplicate_groups)} duplicate groups")
                # Save duplicate groups
                with open(quality_output / "duplicate_groups.json", 'w') as f:
                    json.dump({str(k): [str(p) for p in v] 
                              for k, v in duplicate_groups.items()}, f, indent=2)
                
                # Get representative images (keep first, remove others)
                if dup_config.get('remove_duplicates', True):
                    representative_images = get_representative_images(duplicate_groups)
                    print(f"Keeping {len(representative_images)} representative images")
                    results['duplicates'] = {
                        'total_groups': len(duplicate_groups),
                        'total_duplicates': sum(len(v) - 1 for v in duplicate_groups.values()),
                        'representative_images': len(representative_images)
                    }
                else:
                    results['duplicates'] = {
                        'total_groups': len(duplicate_groups),
                        'total_duplicates': sum(len(v) - 1 for v in duplicate_groups.values())
                    }
            else:
                print("✓ No duplicate images found")
                results['duplicates'] = {'total_groups': 0, 'total_duplicates': 0}
        
        # Save quality check results
        with open(quality_output / "quality_checks.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Data quality checks complete. Results saved to {quality_output}")
    
    # Step 1: Dataset Structure Analysis
    if not skip_analysis and (step is None or step == "analyze"):
        print("\n[Step 0] Dataset Structure Analysis")
        analysis_output = processed_dir / "analysis"
        analysis_result = analyze_dataset_structure(
            raw_data_dir, 
            analysis_output,
            num_samples=1000
        )
        results['analysis'] = analysis_result
        
        # Update recommendations in config if needed
        recommendations = analysis_result.get('recommendations', {})
        if recommendations.get('suggested_noise_threshold'):
            preprocess_config['noise_reduction']['threshold'] = recommendations['suggested_noise_threshold']
        if recommendations.get('suggested_brightness_threshold'):
            preprocess_config['gamma_correction']['threshold'] = recommendations['suggested_brightness_threshold']
    
    # Step 1: Resize
    if step is None or step == "resize":
        print("\n[Step 1] Image Resizing")
        resize_output = processed_dir / "resized"
        resize_result = resize_dataset(
            current_dir,
            resize_output,
            target_size=tuple(preprocess_config['target_size']),
            method=preprocess_config['resize_method'],
            num_samples=vis_config['num_samples']
        )
        results['resize'] = resize_result
        current_dir = resize_output
    
    # Step 2: Noise Reduction
    if (step is None or step == "noise_reduction") and preprocess_config['noise_reduction']['enabled']:
        print("\n[Step 2] Noise Reduction")
        noise_output = processed_dir / "denoised"
        noise_result = reduce_noise_dataset(
            current_dir,
            noise_output,
            method=preprocess_config['noise_reduction']['method'],
            threshold=preprocess_config['noise_reduction']['threshold'],
            gaussian_kernel=tuple(preprocess_config['noise_reduction']['gaussian_kernel']),
            bilateral_params={
                'd': preprocess_config['noise_reduction']['bilateral_d'],
                'sigma_color': preprocess_config['noise_reduction']['bilateral_sigma_color'],
                'sigma_space': preprocess_config['noise_reduction']['bilateral_sigma_space']
            },
            num_samples=vis_config['num_samples']
        )
        results['noise_reduction'] = noise_result
        current_dir = noise_output
    
    # Step 3: CLAHE
    if (step is None or step == "clahe") and preprocess_config['clahe']['enabled']:
        print("\n[Step 3] CLAHE")
        clahe_output = processed_dir / "clahe"
        clahe_result = apply_clahe_dataset(
            current_dir,
            clahe_output,
            clip_limit=preprocess_config['clahe']['clip_limit'],
            tile_grid_size=tuple(preprocess_config['clahe']['tile_grid_size']),
            adaptive=True,
            brightness_threshold=preprocess_config['clahe']['brightness_threshold'],
            num_samples=vis_config['num_samples']
        )
        results['clahe'] = clahe_result
        current_dir = clahe_output
    
    # Step 4: Color Space Conversion
    if step is None or step == "color_space":
        print("\n[Step 4] Color Space Conversion")
        color_output = processed_dir / "color_spaces"
        color_result = convert_color_spaces_dataset(
            current_dir,
            color_output,
            convert_hsv=preprocess_config['color_space']['hsv'],
            convert_lab=preprocess_config['color_space']['lab'],
            convert_grayscale=preprocess_config['color_space']['grayscale'],
            num_samples=vis_config['num_samples']
        )
        results['color_space'] = color_result
        # Use grayscale for next steps if available
        if preprocess_config['color_space']['grayscale']:
            current_dir = color_output / "grayscale"
    
    # Step 5: Gamma Correction
    if (step is None or step == "gamma") and preprocess_config['gamma_correction']['enabled']:
        print("\n[Step 5] Gamma Correction")
        gamma_output = processed_dir / "gamma_corrected"
        gamma_result = apply_gamma_correction_dataset(
            current_dir,
            gamma_output,
            gamma=preprocess_config['gamma_correction']['gamma'],
            threshold=preprocess_config['gamma_correction']['threshold'],
            num_samples=vis_config['num_samples']
        )
        results['gamma_correction'] = gamma_result
        current_dir = gamma_output
    
    # Step 6: Normalization
    if step is None or step == "normalize":
        print("\n[Step 6] Normalization")
        normalize_output = processed_dir / "normalized"
        normalize_result = normalize_dataset(
            current_dir,
            normalize_output,
            method=preprocess_config['normalization']['method'],
            range_min=preprocess_config['normalization']['range'][0],
            range_max=preprocess_config['normalization']['range'][1],
            save_as_float=True,  # Save as .npy for feature extraction
            num_samples=vis_config['num_samples']
        )
        results['normalization'] = normalize_result
        current_dir = normalize_output
    
    # Save pipeline results
    results_path = processed_dir / "preprocessing_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Preprocessing Pipeline Complete!")
    print("="*60)
    print(f"Results saved to: {results_path}")
    print(f"Final processed images: {current_dir}")
    print("="*60)
    
    return results


def main():
    """Main entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--step', type=str, default=None,
                       choices=['data_quality', 'analyze', 'resize', 'noise_reduction', 'clahe', 
                               'color_space', 'gamma', 'normalize'],
                       help='Run specific step only')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip dataset analysis step')
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent.parent / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_preprocessing_pipeline(config_path, args.step, args.skip_analysis)


if __name__ == "__main__":
    main()
