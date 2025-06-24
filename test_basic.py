#!/usr/bin/env python3
"""
Basic test script to verify package structure
"""

def test_imports():
    """Test if we can import the package"""
    try:
        from mywardrobe import build_index, load_index, search, encode_query
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_utils():
    """Test utility functions"""
    try:
        from mywardrobe.utils import apply_mask, preprocess_image, cosine_sim
        print("✅ Utils imports successful")
        return True
    except Exception as e:
        print(f"❌ Utils import error: {e}")
        return False

def main():
    print("🧪 Testing basic functionality...")
    
    # Test imports
    if not test_imports():
        return
    
    # Test utils
    if not test_utils():
        return
    
    print("✅ Basic tests passed!")
    print("\n📝 Next steps:")
    print("1. The package structure is working")
    print("2. CLIP model loading needs to be debugged")
    print("3. Try running with a different Python environment or CLIP version")

if __name__ == "__main__":
    main() 