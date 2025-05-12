import os
import numpy as np
import faiss
import clip
import torch
from src.utils import apply_mask, preprocess_image, cosine_sim

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def build_index(image_dir, mask_dir, out_emb="embeddings.npy", out_idx="index_paths.txt"):
    embeddings, paths = [], []
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path  = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + "_segm.png")
        img       = apply_mask(img_path, mask_path)
        inp       = preprocess_image(img, preprocess).to(device)

        with torch.no_grad():
            emb = model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy())
        paths.append(img_path)

    embeddings = np.vstack(embeddings)
    np.save(out_emb, embeddings)
    with open(out_idx, "w") as f:
        f.write("\n".join(paths))
    print(f"✔ Saved {len(paths)} embeddings → {out_emb}")
    print(f"✔ Saved paths → {out_idx}")

def search(query_image, query_text, top_k, emb_file, idx_file):
    # Load index
    db     = np.load(emb_file)
    paths  = open(idx_file).read().splitlines()
    index  = faiss.IndexFlatIP(db.shape[1])
    index.add(db)

    # Encode query
    img_emb  = model.encode_image(preprocess_image(query_image, preprocess).to(device))
    txt_emb  = model.encode_text(clip.tokenize([query_text]).to(device)) \
                   if query_text else torch.zeros_like(img_emb)
    with torch.no_grad():
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    alpha     = min(0.4 + 0.02 * len(query_text.split()), 0.6)
    query_emb = (1 - alpha) * img_emb + alpha * txt_emb
    query_np  = query_emb.cpu().numpy()

    # Search
    D, I = index.search(query_np, top_k)
    for score, idx in zip(D[0], I[0]):
        print(f"{paths[idx]} — sim={score:.4f}")
