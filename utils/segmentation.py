import requests
import io
from PIL import Image, ImageOps
import numpy as np
import base64
import os
import zipfile

# Use a default model that supports instance segmentation
# Using the new router URL to avoid 410 errors
MODEL_ID = "facebook/mask2former-swin-large-coco-instance"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

import mimetypes

def segment_image(image_path, api_token=None):
    """
    Sends image to Hugging Face API for instance segmentation.
    Returns a list of masks/labels.
    """
    headers = {
        "x-wait-for-model": "true"
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Determine content type
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    
    headers["Content-Type"] = mime_type
    
    with open(image_path, "rb") as f:
        data = f.read()
    
    response = requests.post(API_URL, headers=headers, data=data)
    
    if response.status_code != 200:
        error_msg = f"API Error: {response.status_code} - {response.text}"
        if response.status_code == 401:
             raise Exception("Authentication failed. Please check your HF_API_TOKEN in .env.")
        raise Exception(error_msg)
        
    results = response.json()
    
    # Standardize output for our processing function
    # API request using requests returns list of dicts: [{'score': float, 'label': str, 'mask': 'base64string'}]
    # NOTE: mask2former usually returns mask as base64 string when using raw API
    
    standardized_results = []
    if isinstance(results, list):
        for item in results:
            standardized_results.append({
                'score': item.get('score', 0),
                'label': item.get('label', 'object'),
                'mask': item.get('mask') # Pass raw base64 to next step
            })
    else:
         # Some errors return a dict
         if 'error' in results:
             raise Exception(f"API Error: {results['error']}")
         raise Exception(f"Unexpected API response type: {type(results)}")

    return standardized_results

def extract_objects(image_path, segmentation_results, output_dir):
    """
    Extracts objects from the image based on segmentation results.
    Saves each object as a transparent PNG.
    Returns a list of generated file paths.
    """
    original_image = Image.open(image_path).convert("RGBA")
    width, height = original_image.size
    
    extracted_files = []
    
    # Sort results by score or area if needed, but for now just process all
    # The API returns a list of objects, each with a 'mask' (base64 encoded) or polygon
    # Note: mask2former returns a list of dictionaries. 
    # Let's handle the specific format returned by the API.
    # Usually it returns [{'score': ..., 'label': ..., 'mask': 'base64str'}, ...] for some models
    # However, the standard Inference API for image-segmentation often returns:
    # [{'score': 0.99, 'label': 'person', 'mask': '...'}] where mask is base64 encoded grayscale image
    
    # We need to verify the response format. If it's the standard pipeline, it returns masks.
    
    if not isinstance(segmentation_results, list):
         raise Exception("Unexpected API response format.")

    for i, obj in enumerate(segmentation_results):
        
        label = obj.get('label', 'object')
        score = obj.get('score', 0.0)
        
        mask_image = None
        if 'mask' in obj and isinstance(obj['mask'], str):
             # Base64 string from raw API
            try:
                mask_bytes = base64.b64decode(obj['mask'])
                mask_image = Image.open(io.BytesIO(mask_bytes))
            except Exception as e:
                print(f"Failed to decode mask for {label}: {e}")
                continue
        elif 'mask_obj' in obj:
             # PIL Object from InferenceClient (fallback if we switched back)
             mask_image = obj['mask_obj']
        
        if not mask_image:
            continue
            
        mask_image = mask_image.convert("L")
        
        # Resize mask to match original image if needed
        # Use LANCZOS for high-quality downsampling/upsampling logic to avoid jagged edges
        if mask_image.size != original_image.size:
            mask_image = mask_image.resize(original_image.size, Image.LANCZOS)
        
        # Clean up the mask: Threshold to remove faint background noise but keep anti-aliasing
        # This helps "just the object" be masked out by removing low-confidence pixels
        mask_arr = np.array(mask_image)
        # Binarize slightly to remove halo? Or just keep it soft?
        # User said "accuracy is low", implying bad edges.
        # Let's keep it soft but clamp the bottom range to clear background.
        
        # Simple thresholding to remove noise (anything < 10/255 becomes 0)
        # But keep the upper range soft for anti-aliasing
        mask_arr = np.where(mask_arr < 10, 0, mask_arr)
        mask_image = Image.fromarray(mask_arr)

        # Create composite: apply mask to alpha channel
        object_img = original_image.copy()
        
        # Ensure mask is L mode (8-bit pixels, black and white)
        if mask_image.mode != 'L':
             mask_image = mask_image.convert('L')
             
        object_img.putalpha(mask_image)
        
        # Crop to bounding box of the non-transparent area
        bbox = object_img.getbbox()
        if bbox:
            object_img = object_img.crop(bbox)
            filename = f"{label}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            object_img.save(filepath, format="PNG")
            extracted_files.append(filepath)
            
    return extracted_files

def create_zip(file_paths, zip_path):
    """
    Creates a zip file containing the specified files.
    """
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_paths:
            zipf.write(file, os.path.basename(file))
    return zip_path
