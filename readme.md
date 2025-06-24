# 👗 MyWardrobe — Multimodal Fashion Retrieval & Wardrobe Assistant  
From dataset-only prototype → zero-cost, cloud-hosted MVP

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/) 
[![Streamlit Cloud Ready](https://img.shields.io/badge/Streamlit-Cloud_ready-orange)](https://streamlit.io/)  
Licensed under the MIT License.

---

## 🎯 Project Summary

A production‐ready multimodal recommendation engine for **fashion e-commerce**, combining both image and text input using:
- **Segmentation masks** to isolate garments  
- **CLIP** for unified vision & language embeddings  
- **FAISS** for lightning-fast vector search  

---

## 1 . Why this project matters
Fashion shoppers rarely search with *only* keywords or *only* images.  
MyWardrobe blends **segmentation-cleaned images + natural-language text** in a single CLIP embedding space, stores 44k+ products in FAISS, and serves results in ±250 ms on a free CPU dyno.  

The codebase is intentionally lean so that we can run it end-to-end on free credits:

| Layer | Tech | Zero-cash tier we use |
|-------|------|----------------------|
| Vector embeddings | CLIP - ViT-B/32 | Google Colab T4 (free) |
| ANN search | FAISS `IndexFlatIP` | Render / GitHub Codespaces CPU |
| API | FastAPI | Render free web-service |
| Front-end | Streamlit | Streamlit Cloud community |
| Wardrobe DB | Supabase Postgres | Hobby 500 MB |
| Auth | Firebase | 10 k MAU free |

---

## 2 . Quick start (local)

### 2-A. Environment Setup

```bash
# Clone the repository
git clone https://github.com/rahul370139/Fashion_Recommendation_model_demo.git
cd Fashion_Recommendation_model_demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (with proper NumPy version)
pip install numpy==1.26.4
pip install -r requirements.txt

# Set environment variable for FAISS compatibility
export KMP_DUPLICATE_LIB_OK=TRUE  # On Windows: set KMP_DUPLICATE_LIB_OK=TRUE
```

**Alternative: Use the automated setup script**
```bash
chmod +x setup.sh
./setup.sh
```

### 2-B. Test Installation

```bash
# Test basic functionality
python test_basic.py

# Test CLIP with mock data
python test_mock.py

# Test with small dataset
python test_small.py
```

### 2-C. Build the Index

**Prerequisites:**
- Raw images in `/data/images/` (or your image directory)
- Mask PNGs ending with `_segm.png` in `/data/masks/` (or your mask directory)

```bash
# Build embeddings from your dataset
python main.py prep \
  --image_dir data/images \
  --mask_dir  data/masks \
  --out_emb   embeddings.npy \
  --out_idx   index_paths.txt
```

This runs `build_index()` which:
- Loads CLIP lazily once
- Applies segmentation masks (blacking out non-clothing pixels)
- ℓ₂-normalizes each embedding
- Writes the matrix + path list (≈ 43MB for 44k × 512-d vectors)

### 2-D. Search and Query

```bash
# Text-only search
python main.py query \
  --query_text "red floral dress" \
  --top_k 10 \
  --emb_file embeddings.npy \
  --idx_file index_paths.txt

# Image-only search
python main.py query \
  --query_image sample.jpg \
  --top_k 10 \
  --emb_file embeddings.npy \
  --idx_file index_paths.txt

# Multimodal search (image + text)
python main.py query \
  --query_image sample.jpg \
  --query_text "same in navy blue" \
  --top_k 10 \
  --emb_file embeddings.npy \
  --idx_file index_paths.txt
```

### 2-E. Run the Full Demo Stack

**Start the API:**
```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Launch Streamlit UI (in another terminal):**
```bash
streamlit run ui/streamlit_app.py
```

Then:
- Drag-drop or camera-capture an image
- Type optional text ("same in navy blue")
- Click Search

---

## 3 . What We've Accomplished ✅

### Core System
- ✅ **CLIP Integration**: Fixed device detection for Apple Silicon (MPS), CUDA, and CPU
- ✅ **Model Optimization**: Switched to ViT-B/32 for consistency and performance
- ✅ **Mask Processing**: Implemented proper segmentation mask application
- ✅ **FAISS Search**: Replaced cosine similarity with efficient FAISS IndexFlatIP
- ✅ **Full Dataset Indexing**: Successfully indexed 44,096 DeepFashion images
- ✅ **Multimodal Queries**: Support for image-only, text-only, and combined queries

### Environment & Dependencies
- ✅ **Dependency Management**: Fixed NumPy version conflicts (1.26.4)
- ✅ **OpenMP Compatibility**: Resolved FAISS runtime issues with `KMP_DUPLICATE_LIB_OK=TRUE`
- ✅ **Automated Setup**: Created `setup.sh` for one-click environment setup
- ✅ **Configuration System**: Centralized config with validation scripts

### Testing & Validation
- ✅ **Comprehensive Tests**: Created test suite for all components
- ✅ **Mock Data**: Built-in test data for development and validation
- ✅ **Error Handling**: Robust error handling and debugging information

### Production Readiness
- ✅ **API Endpoints**: FastAPI with proper error handling
- ✅ **Configuration Management**: Environment-based config with validation
- ✅ **Automation Scripts**: `run_search.py` for automated workflows
- ✅ **Documentation**: Complete setup and usage instructions

---

## 4 . Roadmap to the zero-cost MVP

| Week | Milestone | Status | Key Files |
|------|-----------|--------|-----------|
| 0 | Repo hardening | ✅ **DONE** | All core files implemented |
| 1 | Scraping ShopStyle RSS → data/raw/ | 🔄 **IN PROGRESS** | `/scraper/rss_spider.py` |
| 2 | Colab embedding job for scraped 10k | 📋 **PLANNED** | `notebook colab_embed.ipynb` |
| 2 | Deploy FastAPI on Render | 📋 **PLANNED** | `render.yaml` |
| 2 | Streamlit Cloud front-end | 📋 **PLANNED** | `streamlit_app.py` |
| 3 | Wardrobe CRUD via Supabase | 📋 **PLANNED** | `db.py` (config ready) |
| 4 | 90-s Loom demo | 📋 **PLANNED** | — |

**After week 4 we iterate on:**
- LLM "AI stylist" RAG agent (calls /search under the hood)
- Virtual try-on micro-service (TryOnDiffusion on Segmind)
- Sustainability-score scraper (LLM information extraction)

---

## 5 . Code Health Checklist

| Area | Status | Notes |
|------|--------|-------|
| Package import path | ✅ **Good** | `pip install -e .` works |
| Device fallback | ✅ **Good** | Auto-detects M-series / CUDA / CPU |
| Index reuse | ✅ **Good** | `load_index()` cheap, API optimized |
| Tests | ✅ **Good** | Comprehensive test suite |
| Fine-tune script | 🔄 **Partial** | Basic scaffold ready, needs CLI args |

---

## 6 . Troubleshooting

### Common Issues

**1. NumPy Version Conflicts**
```bash
pip install numpy==1.26.4 --force-reinstall
```

**2. FAISS Runtime Error**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**3. CLIP Installation Issues**
```bash
pip install openai-clip --no-deps
pip install torch torchvision
```

**4. Device Detection Issues**
```bash
# Check available devices
python -c "import torch; print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

### Validation Scripts

```bash
# Validate complete setup
python validate_setup.py

# Interactive configuration
python setup_config.py
```

---

## 7 . Contributing
Fork → feature branch → PR.

Run `pytest -q` to ensure all tests pass.

All PRs must keep the test suite green.

---

## 8 . Citing

```bibtex
@software{Sharma_2025_MyWardrobe,
  author       = {Rahul Sharma},
  title        = {MyWardrobe: Multimodal Clothing Retrieval and Wardrobe Assistant},
  year         = 2025,
  note         = {GitHub repository},
  url          = {https://github.com/rahul370139/Fashion_Recommendation_model_demo}
}
```

---

## 9 . Contact
Created by Rahul Sharma (MS Data Science, U Maryland).
Let's talk internships or collaborations → rahul.sharma@umd.edu.