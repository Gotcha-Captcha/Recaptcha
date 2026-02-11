"""
기존 시각화 파일들을 v2로 백업
"""
import shutil
from pathlib import Path


def backup_visualizations_v2():
    """기존 visualization 파일들을 v2로 백업"""
    project_root = Path(__file__).parent.parent
    vis_dir = project_root / "data" / "visualization"
    
    if not vis_dir.exists():
        print(f"Visualization directory not found: {vis_dir}")
        return
    
    print("=" * 60)
    print("Backing up existing visualizations to v2...")
    print("=" * 60)
    
    # 백업할 디렉토리 목록 (v2는 새 파이프라인 결과용이므로 _backup으로 백업)
    dirs_to_backup = [
        "clahe",
        "noise_reduction",
        "resize",
        "analysis"
    ]
    
    for dir_name in dirs_to_backup:
        source_dir = vis_dir / dir_name
        target_dir = vis_dir / f"{dir_name}_backup"  # v2 대신 _backup 사용
        
        if source_dir.exists():
            if target_dir.exists():
                print(f"⚠️  {target_dir} already exists. Skipping...")
            else:
                print(f"📁 Copying {dir_name} → {dir_name}_backup...")
                shutil.copytree(source_dir, target_dir)
                print(f"   ✓ Copied {len(list(source_dir.rglob('*')))} files")
        else:
            print(f"⚠️  {source_dir} does not exist. Skipping...")
    
    # 기존 v2 폴더가 백업으로 생성된 경우 정리 안내
    print("\n" + "=" * 60)
    print("⚠️  주의: 기존 v2 폴더가 백업으로 생성된 경우 수동으로 정리하세요")
    print("   - noise_reduction_v2 (백업) → noise_reduction_backup으로 이동 권장")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("✓ Backup complete!")
    print("=" * 60)


if __name__ == "__main__":
    backup_visualizations_v2()
