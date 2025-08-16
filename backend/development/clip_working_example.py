#!/usr/bin/env python3
"""
Working CLIP example based on the provided working code
This demonstrates the key fixes made to the retrieval system
"""

import os
import torch
import numpy as np
from PIL import Image
import clip
from sklearn.metrics.pairwise import cosine_similarity

def setup_clip():
    """Setup CLIP model with proper device detection"""
    # Updated device detection to support Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model (ViT-B/32 as in working code)
    model_name = "ViT-B/32"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    
    return model, preprocess, device

def apply_mask_working(original_img_path, segm_mask_path):
    """Apply segmentation mask (from working code)"""
    img = Image.open(original_img_path).convert('RGB')
    img_np = np.array(img)
    
    mask = Image.open(segm_mask_path).convert('L')
    mask = mask.resize(img.size, Image.NEAREST)
    mask_np = np.array(mask)

    # Binary mask where clothing pixels > 0
    binary_mask = (mask_np > 0).astype(np.uint8)

    # Black out non-clothing pixels
    img_np[binary_mask == 0] = 0

    masked_img = Image.fromarray(img_np)
    return masked_img

def get_image_embedding(img_path, model, preprocess, device):
    """Get image embedding (from working code)"""
    img = Image.open(img_path).convert('RGB')
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    return image_feature.cpu().numpy()

def get_text_embedding(user_text, model, device):
    """Get text embedding (from working code)"""
    text_tokens = clip.tokenize([user_text]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_tokens)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    return text_feature.cpu().numpy()

def combine_embeddings(image_embed, text_embed, alpha):
    """Combine image and text embeddings (from working code)"""
    combined = (alpha * text_embed) + ((1 - alpha) * image_embed)
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    return combined

def find_similar_images(query_embedding, dataset_embeddings, dataset_paths, top_k):
    """Find similar images using cosine similarity (from working code)"""
    similarities = cosine_similarity(query_embedding, dataset_embeddings)
    similarities = similarities.flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_k]
    results = [(dataset_paths[i], similarities[i]) for i in top_indices]
    return results

def main():
    """Main function demonstrating the working CLIP implementation"""
    print("Setting up CLIP model...")
    model, preprocess, device = setup_clip()
    
    print("\nKey fixes implemented:")
    print("1. ✓ Device detection supports Apple Silicon (MPS)")
    print("2. ✓ CLIP model changed to ViT-B/32")
    print("3. ✓ Proper mask application logic")
    print("4. ✓ Correct embedding normalization")
    print("5. ✓ Cosine similarity instead of FAISS")
    
    print("\nThe system is now ready to use with the corrected implementation!")
    print("You can run: python main.py prep --image_dir <path> --mask_dir <path>")
    print("Or: python main.py query --query_image <path> --query_text <text>")

if __name__ == "__main__":
    main() 