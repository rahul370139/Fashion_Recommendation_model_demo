import os
from PIL import Image
import numpy as np
import torch

def apply_mask(img_path, mask_path):
    """Apply segmentation mask to image, matching the working code approach."""
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(img.size, Image.NEAREST)
    mask_np = np.array(mask)

    # Binary mask where clothing pixels > 0
    binary_mask = (mask_np > 0).astype(np.uint8)

    # White out non-clothing pixels (set to 255 instead of 0)
    img_np[binary_mask == 0] = 255

    masked_img = Image.fromarray(img_np)
    return masked_img

def preprocess_image(img, preprocess_fn):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    return preprocess_fn(img).unsqueeze(0)

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)
