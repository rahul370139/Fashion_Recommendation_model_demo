#!/usr/bin/env python3
"""
Test script using mock retrieval functions to avoid CLIP issues
"""

import os
import shutil
from src.mywardrobe.retrieval_mock import build_index_mock, search_mock

def main():
    # Create a small test directory with just a few images
    test_dir = "test_images"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Copy first 5 images from the dataset
    source_dir = '/Users/rahul/Downloads/deepfashion1_data/images'
    count = 0
    for fname in os.listdir(source_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')) and count < 5:
            src_path = os.path.join(source_dir, fname)
            dst_path = os.path.join(test_dir, fname)
            shutil.copy2(src_path, dst_path)
            print(f"📁 Copied: {fname}")
            count += 1
    
    print(f"✅ Copied {count} images to test directory")
    
    # Build index with mock embeddings
    print("\n🔍 Building mock index...")
    try:
        build_index_mock(test_dir, test_dir, 'embeddings_mock.npy', 'index_paths_mock.txt')
        print("✅ Mock index built successfully!")
    except Exception as e:
        print(f"❌ Error building mock index: {e}")
        return
    
    # Test search
    print("\n🔍 Testing mock search...")
    test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
    print(f"📸 Testing with: {os.path.basename(test_image)}")
    
    try:
        D, I, paths = search_mock(test_image, "fashion", 3, 'embeddings_mock.npy', 'index_paths_mock.txt')
        print("✅ Mock search completed!")
        
        print("\n📋 Results:")
        for i, (score, idx) in enumerate(zip(D, I)):
            print(f"{i+1}. {os.path.basename(paths[idx])} (score: {score:.4f})")
            
    except Exception as e:
        print(f"❌ Mock search error: {e}")

if __name__ == "__main__":
    main() 