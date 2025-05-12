from PIL import Image
import numpy as np
import torch

def apply_mask(img_path, mask_path):
    img  = Image.open(img_path).convert("RGB")
    if not os.path.exists(mask_path):
        return img
    mask = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)
    m_np = np.array(mask) > 0
    img_np = np.array(img)
    img_np[~m_np] = 0
    return Image.fromarray(img_np)

def preprocess_image(img, preprocess_fn):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    return preprocess_fn(img).unsqueeze(0)

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)
