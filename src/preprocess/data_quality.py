"""
Data quality checks: duplicate detection, class distribution analysis.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_image, ensure_dir


def detect_duplicate_images(image_paths: List[Path], 
                            similarity_threshold: float = 0.95,
                            method: str = 'hash') -> Dict:
    """
    Detect duplicate or near-duplicate images.
    
    Args:
        image_paths: List of image paths
        similarity_threshold: Similarity threshold (0.95 = 95% similar)
        method: 'hash' (fast) or 'feature' (accurate but slow)
    
    Returns:
        Dictionary with duplicate groups
    """
    print(f"\n{'='*60}")
    print("Detecting duplicate images...")
    print(f"{'='*60}")
    
    if method == 'hash':
        return _detect_duplicates_by_hash(image_paths, similarity_threshold)
    else:
        return _detect_duplicates_by_features(image_paths, similarity_threshold)


def _detect_duplicates_by_hash(image_paths: List[Path], 
                               threshold: float) -> Dict:
    """Detect duplicates using perceptual hash (fast)."""
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        print("Warning: imagehash not installed. Install with: pip install imagehash")
        return {}
    
    print("Computing perceptual hashes...")
    image_hashes = {}
    
    for img_path in tqdm(image_paths, desc="Computing hashes"):
        try:
            img = Image.open(img_path)
            phash = imagehash.phash(img)
            image_hashes[img_path] = phash
        except Exception as e:
            continue
    
    # Find duplicate groups
    duplicate_groups = {}
    processed = set()
    
    print("Finding duplicate groups...")
    for img_path, hash_val in tqdm(image_hashes.items(), desc="Comparing hashes"):
        if img_path in processed:
            continue
        
        duplicates = [img_path]
        for other_path, other_hash in image_hashes.items():
            if img_path != other_path and other_path not in processed:
                # Hamming distance
                hamming_distance = hash_val - other_hash
                if hamming_distance <= 5:  # Adjustable threshold
                    duplicates.append(other_path)
                    processed.add(other_path)
        
        if len(duplicates) > 1:
            duplicate_groups[img_path] = duplicates
            processed.add(img_path)
    
    print(f"Found {len(duplicate_groups)} duplicate groups")
    return duplicate_groups


def _detect_duplicates_by_features(image_paths: List[Path], 
                                   threshold: float) -> Dict:
    """Detect duplicates using feature matching (slow but accurate)."""
    # This is more accurate but much slower
    # For 55,000 images, hash method is recommended
    print("Feature-based duplicate detection is slow. Use hash method instead.")
    return {}


def check_filename_duplicates(image_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Check for duplicate filenames (simple method).
    
    Returns:
        Dictionary mapping filename to list of paths with that filename
    """
    filename_groups = {}
    
    for img_path in image_paths:
        filename = img_path.name
        if filename not in filename_groups:
            filename_groups[filename] = []
        filename_groups[filename].append(img_path)
    
    # Return only duplicates
    duplicates = {k: v for k, v in filename_groups.items() if len(v) > 1}
    return duplicates


def analyze_class_distribution(data_dir: Path) -> Dict:
    """
    Analyze class distribution in dataset.
    
    Args:
        data_dir: Root directory containing class folders
    
    Returns:
        Dictionary with class counts, ratios, and balance info
    """
    print(f"\n{'='*60}")
    print("Analyzing class distribution...")
    print(f"{'='*60}")
    
    class_counts = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    # Find all class directories
    for item in data_dir.rglob("*"):
        if item.is_dir():
            # Count images in this directory
            images = []
            for ext in image_extensions:
                images.extend(list(item.glob(f"*{ext}")))
                images.extend(list(item.glob(f"*{ext.upper()}")))
            
            if len(images) > 0:
                class_name = item.name
                class_counts[class_name] = len(images)
    
    if not class_counts:
        # Try alternative structure: look for common class names in path
        for img_path in data_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                # Extract class from parent directory
                class_name = img_path.parent.name
                if class_name not in ['data', 'train', 'test', 'val', 'images']:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total = sum(class_counts.values())
    class_ratios = {k: v/total for k, v in class_counts.items()} if total > 0 else {}
    
    # Check balance
    if class_ratios:
        max_ratio = max(class_ratios.values())
        min_ratio = min(class_ratios.values())
        balance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
        is_balanced = balance_ratio < 2.0
    else:
        balance_ratio = float('inf')
        is_balanced = False
    
    result = {
        'counts': class_counts,
        'ratios': class_ratios,
        'total': total,
        'num_classes': len(class_counts),
        'balance_ratio': balance_ratio,
        'is_balanced': is_balanced,
        'max_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None,
        'min_class': min(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
    }
    
    print(f"Total images: {total:,}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Balance ratio: {balance_ratio:.2f} {'(balanced)' if is_balanced else '(imbalanced)'}")
    print(f"Max class: {result['max_class']} ({class_counts.get(result['max_class'], 0):,} images)")
    print(f"Min class: {result['min_class']} ({class_counts.get(result['min_class'], 0):,} images)")
    
    return result


def visualize_class_distribution(class_dist: Dict, output_dir: Path):
    """Visualize class distribution."""
    ensure_dir(output_dir / "visualizations")
    
    counts = class_dist['counts']
    ratios = class_dist['ratios']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Bar chart of class counts
    ax = axes[0]
    classes = list(counts.keys())
    values = list(counts.values())
    
    bars = ax.barh(classes, values, color='steelblue')
    ax.set_xlabel('Number of Images')
    ax.set_title('Class Distribution (Counts)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f' {val:,}', va='center', fontsize=9)
    
    # 2. Pie chart of class ratios
    ax = axes[1]
    if len(classes) <= 20:  # Only show pie chart if not too many classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        wedges, texts, autotexts = ax.pie(values, labels=classes, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax.set_title('Class Distribution (Ratios)')
    else:
        ax.axis('off')
        # Show as text instead
        text_content = "Class Ratios:\n\n"
        for cls, ratio in sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:20]:
            text_content += f"{cls}: {ratio:.2%}\n"
        ax.text(0.5, 0.5, text_content, ha='center', va='center',
               fontsize=10, family='monospace', transform=ax.transAxes)
        ax.set_title('Top 20 Classes by Ratio')
    
    plt.tight_layout()
    plt.savefig(output_dir / "visualizations" / "class_distribution.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved class distribution visualization to {output_dir / 'visualizations' / 'class_distribution.png'}")


def get_representative_images(duplicate_groups: Dict) -> Set[Path]:
    """
    Get representative images from duplicate groups (keep first, remove others).
    
    Args:
        duplicate_groups: Dictionary of duplicate groups
    
    Returns:
        Set of representative image paths to keep
    """
    representative = set()
    
    for group in duplicate_groups.values():
        # Keep first image, mark others for removal
        if len(group) > 0:
            representative.add(group[0])
    
    return representative
