#!/bin/bash

echo "🚀 Setting up Multimodal Fashion Retrieval System..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "📦 Installing dependencies..."

# First, ensure NumPy 1.26.4 is installed (prevents conflicts)
echo "🔧 Installing NumPy 1.26.4 first..."
pip3 install numpy==1.26.4 --break-system-packages --force-reinstall

# Install all other requirements
echo "📚 Installing all other dependencies..."
pip3 install -r requirements.txt --break-system-packages

# Set environment variable for FAISS
echo "🔧 Setting up environment variables..."
export KMP_DUPLICATE_LIB_OK=TRUE

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import clip
import numpy as np
import faiss
from PIL import Image
print('✅ All core dependencies installed successfully!')
print(f'✅ NumPy version: {np.__version__}')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Using device: {\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\"}')
"

if [ $? -eq 0 ]; then
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Build index: python3 main.py prep --image_dir /path/to/images --mask_dir /path/to/masks"
    echo "2. Search: export KMP_DUPLICATE_LIB_OK=TRUE && python3 main.py query --query_image /path/to/image --top_k 10"
    echo ""
    echo "💡 Remember to set KMP_DUPLICATE_LIB_OK=TRUE before running queries!"
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi 