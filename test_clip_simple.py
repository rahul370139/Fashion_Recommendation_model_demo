#!/usr/bin/env python3
"""
Simple test script to verify CLIP model functionality
This script can be run without installing packages in the current environment
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_clip_import():
    """Test if CLIP can be imported and basic functionality works"""
    try:
        import clip
        import torch
        print("✓ CLIP imported successfully")
        
        # Test device detection
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        # Test model loading
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print("✓ CLIP model loaded successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ CLIP import failed: {e}")
        print("You may need to install CLIP: pip install openai-clip")
        return False
    except Exception as e:
        print(f"✗ CLIP test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CLIP model functionality...")
    success = test_clip_import()
    if success:
        print("\n✓ CLIP model is ready to use!")
    else:
        print("\n✗ CLIP model needs to be installed or configured") 