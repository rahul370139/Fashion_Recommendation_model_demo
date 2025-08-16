import os
import numpy as np
import faiss
import clip
import torch
from .utils import apply_mask, preprocess_image, cosine_sim
from PIL import Image

# Updated device detection to support Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Lazy loader for CLIP model
_clip_cache = None

def _get_clip():
    global _clip_cache
    if _clip_cache is None:
        m, p = clip.load("ViT-B/32", device=device)
        m.eval()
        _clip_cache = (m, p)
    return _clip_cache

def encode_query(img_path, text):
    """Helper function to encode a query (image + text) into a blended vector."""
    model, preprocess = _get_clip()
    
    # Encode image
    img = Image.open(img_path).convert('RGB')
    image_input = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_emb = model.encode_image(image_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    # Encode text if provided
    if text:
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            txt_emb = model.encode_text(text_tokens)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    else:
        txt_emb = torch.zeros_like(img_emb)

    # Blend image and text embeddings with adaptive alpha
    alpha = min(0.4 + 0.02 * len(text.split()), 0.6) if text else 0.0
    query_emb = (1 - alpha) * img_emb + alpha * txt_emb
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    return query_emb.cpu()

def build_index(image_dir, mask_dir, out_emb="embeddings.npy", out_idx="index_paths.txt"):
    model, preprocess = _get_clip()
    embeddings, paths = [], []
    
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        
        # Construct mask path
        base_name, ext = os.path.splitext(img_name)
        segm_name = base_name + "_segm.png"
        mask_path = os.path.join(mask_dir, segm_name)
        
        # Apply mask if it exists, otherwise use original image
        if os.path.exists(mask_path):
            img = apply_mask(img_path, mask_path)
        else:
            img = Image.open(img_path).convert('RGB')
        
        # Preprocess and encode
        image_input = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model.encode_image(image_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy())
        paths.append(img_path)

    embeddings = np.vstack(embeddings).astype("float32")
    np.save(out_emb, embeddings)
    with open(out_idx, "w") as f:
        f.write("\n".join(paths))
    print(f"✔ Saved {len(paths)} embeddings → {out_emb}")
    print(f"✔ Saved paths → {out_idx}")

def load_index(emb_file, idx_file):
    """Load a pre-built FAISS index and paths."""
    vecs = np.load(emb_file)
    ix = faiss.IndexFlatIP(vecs.shape[1])
    ix.add(vecs)
    paths = open(idx_file).read().splitlines()
    return ix, paths

def search(query_image, query_text, top_k, emb_file, idx_file):
    """Search for similar images using FAISS index."""
    # Load index
    ix, paths = load_index(emb_file, idx_file)

    # Encode query
    query_emb = encode_query(query_image, query_text)
    query_np = query_emb.cpu().numpy()

    # Search using FAISS
    D, I = ix.search(query_np, top_k)
    
    print(f"Top {top_k} matches:")
    for score, idx in zip(D[0], I[0]):
        print(f"{paths[idx]} — sim={score:.4f}")
    
    return D[0], I[0], paths
