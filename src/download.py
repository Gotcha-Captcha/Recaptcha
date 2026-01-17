"""
Download reCAPTCHA datasets from Kaggle
"""
import os
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_kaggle_api():
    """Initialize Kaggle API"""
    api = KaggleApi()
    api.authenticate()
    return api

def download_dataset(api, dataset_name, output_dir):
    """
    Download a dataset from Kaggle
    
    Args:
        api: KaggleApi instance
        dataset_name: Dataset name in format 'username/dataset-name'
        output_dir: Directory to save the dataset
    """
    print(f"\n{'='*60}")
    print(f"Downloading dataset: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )
        
        print(f"✓ Successfully downloaded {dataset_name}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {str(e)}")
        return False

def main():
    """Main function to download all datasets"""
    # Define datasets to download
    datasets = [
        {
            'name': 'sanjeetsinghnaik/google-recaptcha',
            'output_dir': 'data/google-recaptcha'
        },
        {
            'name': 'mikhailma/test-dataset',
            'output_dir': 'data/test-dataset'
        },
        {
            'name': 'cry2003/google-recaptcha-v2-images',
            'output_dir': 'data/google-recaptcha-v2-images'
        }
    ]
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("Initializing Kaggle API...")
    try:
        api = setup_kaggle_api()
        print("✓ Kaggle API authenticated successfully")
    except Exception as e:
        print(f"✗ Failed to authenticate Kaggle API: {str(e)}")
        print("\nPlease make sure:")
        print("1. You have installed kaggle: pip install kaggle")
        print("2. You have downloaded kaggle.json from https://www.kaggle.com/account")
        print("3. You have placed kaggle.json in ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    # Download each dataset
    results = []
    for dataset in datasets:
        output_path = project_root / dataset['output_dir']
        success = download_dataset(api, dataset['name'], str(output_path))
        results.append((dataset['name'], success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\nTotal: {successful}/{len(results)} datasets downloaded successfully")

if __name__ == "__main__":
    main()
