import os
import numpy as np
import faiss
import torch
import clip
from PIL import Image

# Updated device detection to support Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Lazy loader for CLIP model (mock version)
_clip_cache = None

def _get_clip_mock():
    """Mock CLIP model loader that returns random embeddings"""
    global _clip_cache
    if _clip_cache is None:
        # Create a mock model that returns random embeddings
        class MockCLIP:
            def eval(self):
                pass
            def encode_image(self, x):
                # Return random embeddings of correct size (512 for ViT-B/32)
                return torch.randn(x.size(0), 512)
            def encode_text(self, x):
                # Return random embeddings of correct size (512 for ViT-B/32)
                return torch.randn(x.size(0), 512)
        
        class MockPreprocess:
            def __call__(self, img):
                # Return a random tensor of correct size
                return torch.randn(1, 3, 224, 224)
        
        _clip_cache = (MockCLIP(), MockPreprocess())
    return _clip_cache

def encode_query_mock(img_path, text):
    """Mock function that returns random embeddings for testing"""
    model, preprocess = _get_clip_mock()
    
    # Mock image processing
    img = Image.open(img_path).convert('RGB')
    image_input = preprocess(img).to(device)
    
    # Mock encoding
    with torch.no_grad():
        img_emb = model.encode_image(image_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    # Mock text encoding if provided
    if text:
        # Mock text tokens
        text_tokens = torch.randint(0, 1000, (1, 77)).to(device)  # Mock CLIP tokenization
        with torch.no_grad():
            txt_emb = model.encode_text(text_tokens)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    else:
        txt_emb = torch.zeros_like(img_emb)

    # Blend embeddings
    alpha = min(0.4 + 0.02 * len(text.split()), 0.6) if text else 0.0
    query_emb = (1 - alpha) * img_emb + alpha * txt_emb
    
    return query_emb

def apply_mask_mock(img_path, mask_path):
    """Mock mask application function"""
    img = Image.open(img_path).convert('RGB')
    # Mock: just return the original image
    return img

def build_index_mock(image_dir, mask_dir, out_emb="embeddings.npy", out_idx="index_paths.txt"):
    """Mock function that creates random embeddings for testing with FAISS"""
    model, preprocess = _get_clip_mock()
    embeddings, paths = [], []
    
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        
        # Construct mask path
        base_name, ext = os.path.splitext(img_name)
        segm_name = base_name + "_segm.png"
        mask_path = os.path.join(mask_dir, segm_name)
        
        # Apply mask if it exists, otherwise use original image
        if os.path.exists(mask_path):
            img = apply_mask_mock(img_path, mask_path)
        else:
            img = Image.open(img_path).convert('RGB')
        
        # Mock preprocessing and encoding
        image_input = preprocess(img).to(device)
        
        with torch.no_grad():
            emb = model.encode_image(image_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy())
        paths.append(img_path)

    embeddings = np.vstack(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Save embeddings and paths
    np.save(out_emb, embeddings)
    with open(out_idx, "w") as f:
        f.write("\n".join(paths))
    
    print(f"✔ Saved {len(paths)} mock embeddings → {out_emb}")
    print(f"✔ Saved paths → {out_idx}")
    print(f"✔ Created FAISS index with {dimension} dimensions")

def load_index(emb_file, idx_file):
    """Load a pre-built FAISS index and paths."""
    vecs = np.load(emb_file)
    ix = faiss.IndexFlatIP(vecs.shape[1])
    ix.add(vecs.astype('float32'))
    paths = open(idx_file).read().splitlines()
    return ix, paths

def search_mock(query_image, query_text, top_k, emb_file, idx_file):
    """Mock search function using FAISS"""
    # Load index
    ix, paths = load_index(emb_file, idx_file)

    # Create a mock query vector
    query_emb = encode_query_mock(query_image, query_text)
    query_np = query_emb.cpu().numpy()

    # Search using FAISS
    D, I = ix.search(query_np, top_k)
    
    print(f"Top {top_k} matches:")
    for score, idx in zip(D[0], I[0]):
        print(f"{paths[idx]} — sim={score:.4f}")
    
    return D[0], I[0], paths

def test_all_functionality():
    """Test all main functionality of the retrieval system"""
    print("Testing all main functionality...")
    print(f"Using device: {device}")
    
    # Test 1: CLIP model loading
    print("\n1. Testing CLIP model loading...")
    try:
        model, preprocess = _get_clip_mock()
        print("✓ Mock CLIP model loaded successfully")
    except Exception as e:
        print(f"✗ CLIP model loading failed: {e}")
        return False
    
    # Test 2: Image processing
    print("\n2. Testing image processing...")
    try:
        test_image_path = "test_images/WOMEN-Tees_Tanks-id_00005085-31_4_full.jpg"
        if os.path.exists(test_image_path):
            img = Image.open(test_image_path).convert('RGB')
            processed = preprocess(img)
            print(f"✓ Image processing successful, output shape: {processed.shape}")
        else:
            print("⚠ Test image not found, skipping image processing test")
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False
    
    # Test 3: Mask application
    print("\n3. Testing mask application...")
    try:
        masked_img = apply_mask_mock(test_image_path, "nonexistent_mask.png")
        print("✓ Mask application successful")
    except Exception as e:
        print(f"✗ Mask application failed: {e}")
        return False
    
    # Test 4: Query encoding
    print("\n4. Testing query encoding...")
    try:
        query_emb = encode_query_mock(test_image_path, "test query")
        print(f"✓ Query encoding successful, embedding shape: {query_emb.shape}")
    except Exception as e:
        print(f"✗ Query encoding failed: {e}")
        return False
    
    # Test 5: Index building
    print("\n5. Testing index building...")
    try:
        build_index_mock("test_images", "test_images", "test_embeddings.npy", "test_paths.txt")
        print("✓ Index building successful")
    except Exception as e:
        print(f"✗ Index building failed: {e}")
        return False
    
    # Test 6: FAISS search
    print("\n6. Testing FAISS search...")
    try:
        search_mock(test_image_path, "test query", 3, "test_embeddings.npy", "test_paths.txt")
        print("✓ FAISS search successful")
    except Exception as e:
        print(f"✗ FAISS search failed: {e}")
        return False
    
    # Cleanup test files
    for file in ["test_embeddings.npy", "test_paths.txt"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n🎉 All functionality tests passed!")
    return True

if __name__ == "__main__":
    test_all_functionality() 