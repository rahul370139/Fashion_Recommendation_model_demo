# 👗 Multimodal Recommendation System Using Image and Text Analysis for E-Commerce

Welcome to the **Multimodal Recommendation System** — a smart fashion search tool that combines the powers of **image + text** to deliver **accurate, personalized product recommendations**.

Created by **Rahul Sharma** for research in AI-driven fashion applications.  
Built using **CLIP**, **cosine similarity**, and **two-stage multimodal reranking**.

---

## 🚀 Overview

**Objective:**  
Help users find clothing items that match both **visual style** and **textual preferences** (e.g. *“floral red dress with puffed sleeves”*).

**Why It Matters:**  
Traditional search lacks nuance — a shirt that *looks right* might not *feel right*. This system enables rich, multimodal search, improving customer satisfaction and boosting conversions in fashion e-commerce platforms.

---

## 🧠 Core Techniques

- **CLIP Model (by OpenAI)**: Encodes both images and text into a shared semantic space
- **Unsupervised Retrieval Task**: No labels needed — everything is similarity-based
- **Reranking Strategy**: Prioritizes textual cues post-retrieval for better match quality
- **Optional Segmentation**: Use Mask R-CNN to crop clothing from backgrounds

---

## 🔍 Features

| Feature | Description |
|--------|-------------|
| 🎨 Multimodal Input | Accepts image, text, or both |
| 🧮 CLIP Embeddings | Transforms all inputs into vector space |
| 🧠 Cosine Similarity Search | Retrieves nearest fashion items |
| 🔁 Two-Stage Reranking | Ranks results by combining vision + language |
| 🧩 Segmentation (optional) | Isolates garments to avoid background bias |

---

## 📊 Dataset: DeepFashion1

- 📦 ~44,000 high-res fashion images
- 📐 Image resolution: 512×1024 & 750×1101
- 🏷️ Labels: Fabric, Texture (not used directly)
- 🔗 [Dataset Link (Google Drive)](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok)

---

## ⚙️ Architecture

![architecture](assets/architecture.png)

---

## 🧪 Evaluation

Manual user survey via Google Form. Participants rated:
- 🎯 Match Accuracy (style/pattern alignment)
- 📈 Overall Satisfaction

📄 [Evaluation Form (Google Docs)](https://docs.google.com/forms/d/e/1FAIpQLSenbdz3Fg_p7ssT8ArugoBNaK9sFnyUaHR-yogiEnGCXBeSsQ/viewform?pli=1)

---

## 📁 Example Usage

```bash
# Encode dataset
python src/clip_retrieval.py --encode_dataset

# Query with image only
python src/inference.py --query_image data/sample_queries/user_query.jpg

# Query with image + text
python src/inference.py --query_image data/sample_queries/user_query.jpg --query_text "red floral dress with puffed sleeves"
