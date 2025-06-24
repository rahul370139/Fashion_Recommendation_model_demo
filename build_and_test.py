#!/usr/bin/env python3
"""
Script to build index and test search functionality
"""

import os
from mywardrobe import build_index, search

def main():
    # Paths
    image_dir = '/Users/rahul/Downloads/deepfashion1_data/images'
    mask_dir = '/Users/rahul/Downloads/deepfashion1_data/images'  # Using same dir for now
    emb_file = 'embeddings.npy'
    idx_file = 'index_paths.txt'
    
    print("🔍 Building index from fashion dataset...")
    print(f"📁 Image directory: {image_dir}")
    
    # Build index
    try:
        build_index(image_dir, mask_dir, emb_file, idx_file)
        print("✅ Index built successfully!")
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return
    
    # Test search with a sample image
    print("\n🔍 Testing search functionality...")
    
    # Find a sample image to test with
    sample_images = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            sample_images.append(fname)
            if len(sample_images) >= 3:  # Get first 3 images
                break
    
    if sample_images:
        test_image = os.path.join(image_dir, sample_images[0])
        print(f"📸 Testing with image: {sample_images[0]}")
        
        try:
            # Test search
            D, I, paths = search(test_image, "fashion", 3, emb_file, idx_file)
            print("✅ Search completed successfully!")
            
            # Show results
            print("\n📋 Search Results:")
            for i, (score, idx) in enumerate(zip(D, I)):
                print(f"{i+1}. {os.path.basename(paths[idx])} (score: {score:.4f})")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")
    else:
        print("❌ No images found in directory")

if __name__ == "__main__":
    main() 