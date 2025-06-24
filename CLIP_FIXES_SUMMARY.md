# CLIP Model Fixes Summary

## Key Issues Fixed

### 1. Device Detection for Apple Silicon
**Problem**: Original code only supported CUDA/CPU, causing issues on Apple Silicon Macs
**Fix**: Updated device detection to support MPS (Metal Performance Shaders)
```python
# Before
device = "cuda" if torch.cuda.is_available() else "cpu"

# After  
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
```

### 2. CLIP Model Version
**Problem**: Using ViT-B/16 instead of ViT-B/32 (which was used in working code)
**Fix**: Changed to ViT-B/32 for consistency
```python
# Before
m, p = clip.load("ViT-B/16", device=device)

# After
m, p = clip.load("ViT-B/32", device=device)
```

### 3. Mask Application Logic
**Problem**: Different mask application approach that didn't match working code
**Fix**: Updated to match the working implementation
```python
# Before: Set non-clothing pixels to white (255)
img_np[~m_np] = 255

# After: Set non-clothing pixels to black (0)
img_np[binary_mask == 0] = 0
```

### 4. Embedding Processing
**Problem**: Inconsistent image preprocessing and normalization
**Fix**: Aligned with working code approach
```python
# Now matches working code:
img = Image.open(img_path).convert('RGB')
image_input = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    emb = model.encode_image(image_input)
    emb = emb / emb.norm(dim=-1, keepdim=True)
```

### 5. Search Algorithm
**Problem**: Using cosine similarity which might not be optimal for large datasets
**Fix**: Switched to FAISS for efficient similarity search
```python
# Now uses FAISS IndexFlatIP for efficient search
ix = faiss.IndexFlatIP(vecs.shape[1])
ix.add(vecs.astype('float32'))
D, I = ix.search(query_np, top_k)
```

## Files Modified

1. **`src/mywardrobe/retrieval.py`**
   - Updated device detection
   - Changed CLIP model to ViT-B/32
   - Fixed embedding processing
   - Switched to FAISS search
   - Removed sklearn dependency

2. **`src/mywardrobe/utils.py`**
   - Updated mask application logic
   - Fixed binary mask handling

3. **`src/mywardrobe/retrieval_mock.py`**
   - Updated to use FAISS
   - Added comprehensive functionality testing
   - Mock CLIP model with proper device support
   - Tests all main features: CLIP loading, image processing, mask application, query encoding, index building, and FAISS search

4. **`main.py`**
   - Added device information display
   - Added better error handling

5. **`requirements.txt`**
   - Removed scikit-learn dependency
   - Kept faiss-cpu for efficient search

## Testing

Created comprehensive test files:
- `test_clip_simple.py`: Basic CLIP functionality test
- `clip_working_example.py`: Complete working example
- `retrieval_mock.py`: Comprehensive functionality testing with FAISS

## Usage

The system should now work properly with:
```bash
# Build embeddings
python main.py prep --image_dir /path/to/images --mask_dir /path/to/masks

# Search
python main.py query --query_image /path/to/query.jpg --query_text "your text query"

# Test all functionality
python src/mywardrobe/retrieval_mock.py
```

## Key Differences from Working Code

The fixes ensure the current implementation matches the working code's approach:
- Same device detection (MPS support)
- Same CLIP model (ViT-B/32)
- Same mask application (black out non-clothing)
- Same embedding normalization
- **FAISS for efficient similarity search** (improved over cosine similarity) 