# 👗 Multimodal Fashion Retrieval – End-to-End Demo  
CLIP + Segmentation + FAISS for image-and-text product search  
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![MIT license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🎯 Project Summary

A production‐ready multimodal recommendation engine for **fashion e-commerce**, combining both image and text input using:
- **Segmentation masks** to isolate garments  
- **CLIP** for unified vision & language embeddings  
- **FAISS** for lightning-fast vector search  

---

## ✨ Features

- **Mask & preprocess** images via segmentation  
- **Generate** and **store** CLIP embeddings for your catalog  
- **Query** by image, text, or hybrid  
- **Adaptive reranking** (tune image/text weight)  
- Minimal dependencies, ready for scale

---

## 🧠 Architecture

```text
1️⃣ Segmentation Preprocessing
   • Apply PNG masks → garment-only images

2️⃣ Embedding Generation
   • CLIP image encoder → ℓ₂-normalized vectors
   • Save `.npy` + path index

3️⃣ Retrieval & Reranking
   • Query: image ⊕ text (alpha weight)
   • FAISS + cosine similarity
   • Top-k nearest neighbours
