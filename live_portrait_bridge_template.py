import fastapi
import uvicorn
import cv2
import json
import numpy as np
import base64
import os
import torch
import requests
from io import BytesIO
from PIL import Image
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

# Auto-download weights if missing
import os
if not os.path.exists("pretrained_weights/base_models/appearance_feature_extractor.pth"):
    print("Downloading missing model weights...")
    import download_weights
    download_weights.main()

from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.io import load_image_rgb
from src.utils.camera import get_rotation_matrix
import asyncio

from src.utils.crop import prepare_paste_back, paste_back

app = fastapi.FastAPI()

# Create cache directory for downloaded images
CACHE_DIR = os.path.join(os.getcwd(), 'character_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def download_image_from_url(url, char_id):
    """Download image from URL and save to cache"""
    cache_path = os.path.join(CACHE_DIR, f"{char_id}.jpg")
    
    # Check if already cached
    if os.path.exists(cache_path):
        return cache_path
    
    try:
        if url.startswith('/'):
            url = f"http://localhost:3000{url}"
            
        print(f"Downloading image for character {char_id} from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert to RGB and save
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(cache_path, 'JPEG', quality=95)
        
        print(f"Image cached at {cache_path}")
        return cache_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Initialize LivePortrait Pipeline
root_dir = os.getcwd()
inference_cfg = InferenceConfig(
    flag_force_cpu=True, 
    flag_use_half_precision=False,
    flag_stitching=True,  # Enable stitching
    checkpoint_F=os.path.join(root_dir, 'pretrained_weights', 'base_models', 'appearance_feature_extractor.pth'),
    checkpoint_M=os.path.join(root_dir, 'pretrained_weights', 'base_models', 'motion_extractor.pth'),
    checkpoint_G=os.path.join(root_dir, 'pretrained_weights', 'base_models', 'spade_generator.pth'),
    checkpoint_W=os.path.join(root_dir, 'pretrained_weights', 'base_models', 'warping_module.pth'),
    checkpoint_S=os.path.join(root_dir, 'pretrained_weights', 'retargeting_models', 'stitching_retargeting_module.pth'),
)
crop_cfg = CropConfig(
    landmark_ckpt_path=os.path.join(root_dir, 'pretrained_weights', 'landmark.onnx'),
    flag_force_cpu=True
)
pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)

# Mapping Character IDs to local source images
CHARACTER_MAP = {
    "default": "assets/examples/source/s9.jpg",
    "sakura": "assets/examples/source/s11.jpg",
}

# Cache for preprocessed character data
character_cache = {}

def preprocess_character(char_id, image_url=None):
    """Preprocess character image for animation"""
    if char_id in character_cache:
        return character_cache[char_id]
    
    # Download image from URL if provided
    if image_url:
        source_path = download_image_from_url(image_url, char_id)
        if source_path is None:
            print(f"[WARNING] Failed to download image for {char_id}, using default.")
            source_path = CHARACTER_MAP.get("default")
    else:
        # Fallback to local file
        source_path = CHARACTER_MAP.get(char_id, CHARACTER_MAP.get("default"))
    
    # Load source image
    img_rgb = load_image_rgb(source_path)
    
    # Crop and prepare
    crop_info = pipeline.cropper.crop_source_image(img_rgb, crop_cfg)
    if crop_info is None:
        raise Exception(f"No face detected in character {char_id}")
    
    img_crop_256x256 = crop_info['img_crop_256x256']
    source_lmk = crop_info['lmk_crop']
    
    # Prepare for inference
    I_s = pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = pipeline.live_portrait_wrapper.get_kp_info(I_s)
    f_s = pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    
    # Prepare mask for stitching/paste-back
    mask_ori_float = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
    
    character_cache[char_id] = {
        'I_s': I_s,
        'x_s_info': x_s_info,
        'f_s': f_s,
        'x_s': x_s,
        'R_s': R_s,
        'source_lmk': source_lmk,
        'img_crop_256x256': img_crop_256x256,
        'img_rgb': img_rgb,           # Original full image
        'crop_info': crop_info,       # Crop info for reverse transform
        'mask_ori_float': mask_ori_float # Mask for paste back
    }
    
    return character_cache[char_id]

def generate_blink_frame(current_char_data):
    """Generate a single frame with eyes closed (blinking)"""
    try:
        # Create base keypoints from source
        x_d_i_new = current_char_data['x_s'].clone()
        
        # MANUALLY CLOSE EYES
        # LivePortrait Eye Indices: [11, 13, 15, 16, 18]
        # We assume adding positive Y (down) to top lids and negative Y (up) to bottom lids
        # But for simplicity in keypoint space, we often justSquash the Y distance.
        # Let's try adding a strong offset to "close" them.
        
        # Approximate blink by moving upper eyelid points DOWN
        # Indices are roughly: 
        # Left Eye: 11 (Top), 13 (Bottom)? 
        # Right Eye: 15 (Top), 16 (Bottom)?
        # (This varies by model, but we'll apply a generic 'close' offset)
        
        blink_offset = 0.015
        
        # Closing Left Eye
        x_d_i_new[:, 11, 1] += blink_offset 
        x_d_i_new[:, 13, 1] -= blink_offset
        
        # Closing Right Eye
        x_d_i_new[:, 15, 1] += blink_offset
        x_d_i_new[:, 16, 1] -= blink_offset
        
        # Also index 18?
        x_d_i_new[:, 18, 1] += blink_offset

        # Perform Stitching
        x_d_i_new = pipeline.live_portrait_wrapper.stitching(
            current_char_data['x_s'], 
            x_d_i_new
        )
        
        # Generate frame
        out = pipeline.live_portrait_wrapper.warp_decode(
            current_char_data['f_s'],
            current_char_data['x_s'],
            x_d_i_new
        )
        
        # Get cropped result
        I_p = pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
        
        # PASTE BACK to original image
        I_p_pstbk = paste_back(
            I_p, 
            current_char_data['crop_info']['M_c2o'], 
            current_char_data['img_rgb'], 
            current_char_data['mask_ori_float']
        )
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', I_p_pstbk[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buffer
    except Exception as e:
        print(f"[ERROR] Blink frame gen failed: {e}")
        return None

@app.websocket("/stream")
async def live_portal_stream(websocket: fastapi.WebSocket):
    await websocket.accept()
    print("React Client Connected to LivePortrait Engine")
    
    current_char_data = None
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            char_id = msg.get('character_id', 'default')
            char_image_url = msg.get('character_image')
            
            # Note: We ignore 'isTalking' for streaming now.
            # We just want to prepare the blink frame.
            
            if char_id:
                print(f"[DEBUG] Processing character {char_id} for hybrid animation")
                
                if char_id not in character_cache:
                    await asyncio.to_thread(preprocess_character, char_id, char_image_url)
                
                current_char_data = character_cache[char_id]
                
                # Generate Blink Frame ONCE
                print(f"[DEBUG] Generating Blink Frame...")
                blink_buffer = await asyncio.to_thread(generate_blink_frame, current_char_data)
                
                if blink_buffer is not None:
                    # Send Binary Frame (This is the blink state)
                    await websocket.send_bytes(blink_buffer.tobytes())
                    print(f"[DEBUG] ✅ Sent Blink Frame ({len(blink_buffer)} bytes)")
                    
                    # Also send a text status saying "BlinkReady"
                    await websocket.send_text(json.dumps({
                        "type": "blink_ready",
                        "data": "Blink frame sent"
                    }))
                else:
                     await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": "Failed to generate blink frame"
                    }))

    except Exception as e:
        print(f"[ERROR] Stream interrupted: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
