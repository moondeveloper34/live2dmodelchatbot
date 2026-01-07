import os
import requests
from tqdm import tqdm

def download_file(url, filepath):
    # Check if file exists and has reasonable size (e.g. > 1KB) to avoid skipping corrupted 15-byte files
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 1024:
            print(f"File already exists and seems valid: {filepath}")
            return
        else:
            print(f"File exists but is too small ({os.path.getsize(filepath)} bytes). Re-downloading: {filepath}")
            os.remove(filepath)

    print(f"Downloading {filepath}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise error for 404/500
        
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filepath,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
                
        print(f"Successfully downloaded: {filepath}")
        
    except Exception as e:
        print(f"Error downloading {filepath}: {str(e)}")
        # Clean up partial/failed file
        if os.path.exists(filepath):
            os.remove(filepath)
        raise e

def main():
    # Using 'resolve/main' with '?download=true' ensures we get the LFS file
    BASE_URL = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main"
    
    weights = [
        # Base Models
        ("pretrained_weights/base_models/appearance_feature_extractor.pth", f"{BASE_URL}/pretrained_weights/base_models/appearance_feature_extractor.pth?download=true"),
        ("pretrained_weights/base_models/motion_extractor.pth", f"{BASE_URL}/pretrained_weights/base_models/motion_extractor.pth?download=true"),
        ("pretrained_weights/base_models/spade_generator.pth", f"{BASE_URL}/pretrained_weights/base_models/spade_generator.pth?download=true"),
        ("pretrained_weights/base_models/warping_module.pth", f"{BASE_URL}/pretrained_weights/base_models/warping_module.pth?download=true"),
        
        # Landmark Model
        ("pretrained_weights/landmark_model/landmarker.onnx", f"{BASE_URL}/pretrained_weights/landmark_model/landmarker.onnx?download=true"),
    ]

    # InsightFace models are usually loaded via library, but we ensure directory exists
    os.makedirs("pretrained_weights/insightface", exist_ok=True)

    print("Starting model weights download...")
    for relative_path, url in weights:
        try:
            download_file(url, relative_path)
        except Exception as e:
            print(f"Failed to download {relative_path}. Deployment may fail.")

if __name__ == "__main__":
    main()
