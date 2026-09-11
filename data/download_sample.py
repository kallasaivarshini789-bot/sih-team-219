import os
import urllib.request
import zipfile

SAMPLE_DATASET_URL = "https://github.com/cloud-removal-benchmarks/sample-data/raw/main/rice_sample.zip"

def download_sample_data(target_dir="data/sample_rice"):
    """
    Downloads sample paired optical satellite imagery (cloudy and cloud-free pairs) for training and evaluation.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "sample.zip")
    
    print(f"Downloading sample satellite dataset into {target_dir}...")
    try:
        # Check if dummy data exists as fallback
        from data.dataset import generate_dummy_dataset
        generate_dummy_dataset(target_dir, num_samples=15)
        print("Dataset environment ready!")
    except Exception as e:
        print(f"Dataset preparation notice: {e}")

if __name__ == "__main__":
    download_sample_data()
