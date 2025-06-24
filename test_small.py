#!/usr/bin/env python3
"""
Test script with limited images to avoid memory issues
"""

import os
import shutil
from mywardrobe import build_index, search

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
    
    # Build index with small dataset
    print("\n🔍 Building index with small dataset...")
    try:
        build_index(test_dir, test_dir, 'embeddings_small.npy', 'index_paths_small.txt')
        print("✅ Small index built successfully!")
    except Exception as e:
        print(f"❌ Error building small index: {e}")
        return
    
    # Test search
    print("\n🔍 Testing search...")
    test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
    print(f"📸 Testing with: {os.path.basename(test_image)}")
    
    try:
        D, I, paths = search(test_image, "fashion", 3, 'embeddings_small.npy', 'index_paths_small.txt')
        print("✅ Search completed!")
        
        print("\n📋 Results:")
        for i, (score, idx) in enumerate(zip(D, I)):
            print(f"{i+1}. {os.path.basename(paths[idx])} (score: {score:.4f})")
            
    except Exception as e:
        print(f"❌ Search error: {e}")

if __name__ == "__main__":
    main() 