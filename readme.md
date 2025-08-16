# 👗 MyWardrobe — AI-Powered Fashion Recommendation & Virtual Wardrobe

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/) 
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-orange)](https://openai.com/research/clip)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-red)](https://github.com/facebookresearch/faiss)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL%20%2B%20Storage-purple)](https://supabase.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Ready-yellow)](https://streamlit.io/)

**A production-ready multimodal AI system that revolutionizes fashion discovery through intelligent image-text understanding and personalized wardrobe management.**

---

## 🚀 **What We Built**

MyWardrobe is a cutting-edge **multimodal recommendation engine** that combines computer vision, natural language processing, and vector search to deliver intelligent fashion recommendations. Unlike traditional systems that rely on either images OR text, our AI understands both simultaneously, creating a seamless shopping experience.

### **🎯 Core Innovation**
- **Multimodal Fusion**: Seamlessly combines image and text queries using CLIP embeddings
- **Intelligent Weighting**: Dynamic alpha blending (0.0-1.0) between visual and textual features
- **Real-time Search**: Sub-250ms response time using FAISS vector similarity
- **Virtual Wardrobe**: AI-powered personal clothing collection management

---

## 🧠 **Technical Architecture**

### **AI/ML Stack**
- **CLIP (ViT-B/32)**: OpenAI's state-of-the-art vision-language model
- **FAISS IndexFlatIP**: High-performance vector similarity search
- **PyTorch**: Deep learning framework with MPS/CUDA/CPU optimization
- **Segmentation Masks**: Precise garment isolation for clean embeddings

### **Backend Infrastructure**
- **FastAPI**: High-performance async API with automatic OpenAPI docs
- **Supabase**: PostgreSQL database + object storage for scalability
- **LangChain + Ollama**: Local LLM integration for AI stylist chat
- **Docker**: Containerized deployment for consistency

### **Frontend & UX**
- **Streamlit**: Interactive web interface with drag-and-drop uploads
- **Real-time Search**: Instant visual feedback and results
- **Responsive Design**: Mobile-friendly interface for on-the-go use

---

## 🔍 **How It Works**

### **1. Multimodal Query Processing**
```
User Input: Image + "casual summer dress"
↓
CLIP Encoding: Image → 512D vector, Text → 512D vector
↓
Intelligent Fusion: (1-α) × Image + α × Text
↓
Normalization: L2 normalization for optimal similarity
↓
FAISS Search: Find closest matches in 44k+ product database
```

### **2. Advanced Weighting System**
- **α = 0.0**: Pure image-based search
- **α = 0.5**: Balanced image-text fusion (default)
- **α = 1.0**: Pure text-based search
- **Dynamic α**: Adjustable based on user preference

### **3. Real-time Vector Search**
- **Database**: 44,096 DeepFashion products indexed
- **Embedding Dimension**: 512D CLIP vectors
- **Search Algorithm**: FAISS IndexFlatIP with cosine similarity
- **Performance**: <250ms response time on CPU

---

## 🎨 **Key Features**

### **🔍 Intelligent Fashion Search**
- **Image Upload**: Drag-and-drop or camera capture
- **Text Queries**: Natural language descriptions
- **Combined Search**: Image + text for precise results
- **Real-time Results**: Instant visual feedback

### **👔 Virtual Wardrobe Management**
- **Personal Collections**: User-specific clothing databases
- **Image Storage**: Supabase object storage integration
- **Smart Organization**: AI-powered categorization
- **Easy Access**: Quick retrieval of saved items

### **💬 AI Stylist Assistant**
- **Natural Conversations**: Chat with fashion AI using Ollama
- **Style Advice**: Personalized recommendations
- **Outfit Planning**: Coordination suggestions
- **Trend Insights**: Fashion knowledge and tips

### **⚡ Performance Optimizations**
- **Lazy Loading**: CLIP model loads only when needed
- **Vector Caching**: Pre-computed embeddings for speed
- **Async Processing**: Non-blocking API responses
- **Memory Efficient**: Optimized for production deployment

---

## 🛠️ **Technical Implementation**

### **Core Algorithms**
```python
# Multimodal embedding fusion
def encode_query(image_path: str, text: str, alpha: float = 0.5):
    img_emb = clip_model.encode_image(preprocess(image))
    txt_emb = clip_model.encode_text(tokenize(text))
    
    # Intelligent fusion with user-defined weight
    query_emb = (1 - alpha) * img_emb + alpha * txt_emb
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    
    return query_emb.cpu()
```

### **Vector Search Engine**
```python
# FAISS-based similarity search
def search_similar(query_embedding, top_k=12):
    D, I = faiss_index.search(query_embedding, top_k)
    return [(paths[i], float(d)) for d, i in zip(D[0], I[0])]
```

### **Database Architecture**
- **Wardrobe Table**: User-specific clothing collections
- **Storage Buckets**: Scalable image storage
- **Row-Level Security**: Secure user data isolation
- **Real-time Sync**: Instant updates across devices

---

## 📊 **Performance Metrics**

| Metric | Value | Significance |
|--------|-------|--------------|
| **Search Speed** | <250ms | Production-ready performance |
| **Database Size** | 44k+ products | Enterprise-scale dataset |
| **Embedding Dim** | 512D | Optimal CLIP representation |
| **Accuracy** | High | CLIP's proven vision-language understanding |
| **Scalability** | Horizontal | FAISS + Supabase architecture |

---

## 🚀 **Deployment & Production**

### **Cloud Infrastructure**
- **Backend**: Railway (FastAPI + ML models)
- **Frontend**: Vercel (Streamlit interface)
- **Database**: Supabase (PostgreSQL + Storage)
- **AI Models**: Local Ollama + CLIP integration

### **Environment Management**
- **Docker**: Consistent deployment across environments
- **Environment Variables**: Secure configuration management
- **Health Checks**: Automated monitoring and recovery
- **CI/CD**: GitHub Actions for automated testing

---

## 🔬 **Research & Innovation**

### **Academic Contributions**
- **Multimodal Fusion**: Novel approach to fashion recommendation
- **Weighted Embeddings**: Dynamic alpha blending methodology
- **Real-time Search**: Optimized vector similarity algorithms
- **User Experience**: Human-centered AI interaction design

### **Technical Challenges Solved**
- **Device Compatibility**: MPS/CUDA/CPU auto-detection
- **Memory Optimization**: Efficient embedding storage and retrieval
- **Real-time Processing**: Sub-second response times
- **Scalable Architecture**: Production-ready deployment

---

## 🎯 **Use Cases & Applications**

### **E-commerce Platforms**
- **Visual Search**: Find similar products instantly
- **Style Matching**: Discover complementary items
- **Personalization**: User-specific recommendations
- **Mobile Shopping**: On-the-go fashion discovery

### **Fashion Brands**
- **Inventory Search**: Quick product identification
- **Style Analysis**: Trend and pattern recognition
- **Customer Insights**: Understanding user preferences
- **Marketing**: Targeted product recommendations

### **Personal Use**
- **Wardrobe Organization**: Digital clothing catalog
- **Outfit Planning**: Coordinate looks efficiently
- **Shopping Lists**: Find similar items to favorites
- **Style Development**: Discover new fashion directions

---

## 🛠️ **Getting Started**

### **Quick Setup**
```bash
# Clone and setup
git clone <repository>
cd mywardrobe
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the system
uvicorn api.app:app --reload  # Backend
streamlit run ui/app.py        # Frontend
```

### **Configuration**
```bash
# Environment variables
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_api_key
OLLAMA_HOST=localhost:11434
```

---

## 🏆 **Why This Project Stands Out**

### **Technical Excellence**
- **State-of-the-art AI**: CLIP + FAISS for cutting-edge performance
- **Production Ready**: Docker, CI/CD, monitoring, and scaling
- **Performance Optimized**: Sub-250ms response times
- **Scalable Architecture**: Horizontal scaling with cloud services

### **Innovation**
- **Multimodal Fusion**: Unique approach to fashion recommendation
- **Intelligent Weighting**: Dynamic user preference adaptation
- **Real-time AI**: Instant intelligent responses
- **User Experience**: Intuitive, responsive interface

### **Business Value**
- **E-commerce Ready**: Immediate commercial application
- **Scalable Model**: Handles enterprise-level datasets
- **Cost Effective**: Free-tier cloud deployment
- **Market Potential**: High-demand fashion tech solution

---

## 🤝 **Contributing & Collaboration**

We welcome contributions from researchers, developers, and fashion enthusiasts! Areas of interest:
- **Model Optimization**: CLIP fine-tuning and performance
- **UI/UX Enhancement**: Better user experience design
- **Feature Expansion**: Additional AI capabilities
- **Performance Tuning**: Speed and accuracy improvements

---

## 📚 **Technical References**

- **CLIP Paper**: [Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
- **FAISS Documentation**: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **FastAPI**: [Modern Python Web Framework](https://fastapi.tiangolo.com/)
- **Supabase**: [Open Source Firebase Alternative](https://supabase.com/)

---

## 👨‍💻 **About the Developer**

**Rahul Sharma** - MS Data Science, University of Maryland

Passionate about building production-ready AI systems that solve real-world problems. This project demonstrates expertise in:
- **Machine Learning**: CLIP, FAISS, PyTorch
- **Full-Stack Development**: FastAPI, Streamlit, Supabase
- **DevOps**: Docker, CI/CD, cloud deployment
- **AI/ML Engineering**: End-to-end system development

**Contact**: rahul.sharma@umd.edu  
**GitHub**: [rahul370139](https://github.com/rahul370139)

---

## 📄 **License**

MIT License - Feel free to use this project for research, commercial applications, or learning purposes.

---

*Built with ❤️ and ☕ for the future of fashion technology*