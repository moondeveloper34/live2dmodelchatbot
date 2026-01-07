import os
import requests
from tqdm import tqdm

def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    # Base URL for model weights (using official HuggingFace Repo or similar)
    BASE_URL = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main"
    
    weights = [
        # Base Models
        ("pretrained_weights/base_models/appearance_feature_extractor.pth", f"{BASE_URL}/pretrained_weights/base_models/appearance_feature_extractor.pth"),
        ("pretrained_weights/base_models/motion_extractor.pth", f"{BASE_URL}/pretrained_weights/base_models/motion_extractor.pth"),
        ("pretrained_weights/base_models/spade_generator.pth", f"{BASE_URL}/pretrained_weights/base_models/spade_generator.pth"),
        ("pretrained_weights/base_models/warping_module.pth", f"{BASE_URL}/pretrained_weights/base_models/warping_module.pth"),
        
        # Landmark Model
        ("pretrained_weights/landmark_model/landmarker.onnx", f"{BASE_URL}/pretrained_weights/landmark_model/landmarker.onnx"), # Placeholder URL, check exact path if fails
    ]

    # InsightFace models are usually handled by the library automatically, 
    # ensuring directory exists
    os.makedirs("pretrained_weights/insightface", exist_ok=True)

    for relative_path, url in weights:
        download_file(url, relative_path)

if __name__ == "__main__":
    main()
