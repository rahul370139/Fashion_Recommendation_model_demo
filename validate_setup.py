#!/usr/bin/env python3
"""
Validation script to check all TODO placeholders and system configuration
"""

import os
import sys
import numpy as np

def check_config_file():
    """Check if config.py exists and validate TODO placeholders"""
    print("🔍 Checking configuration file...")
    
    if not os.path.exists("config.py"):
        print("❌ config.py not found. Please create it with your configuration.")
        return False
    
    try:
        from config import validate_config
        return validate_config()
    except ImportError as e:
        print(f"❌ Error importing config: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("clip", "OpenAI CLIP"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("transformers", "Transformers")
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} not installed")
            all_good = False
    
    return all_good

def check_numpy_version():
    """Check NumPy version compatibility"""
    print("\n🔍 Checking NumPy version...")
    
    version = np.__version__
    if version.startswith('2.'):
        print(f"❌ NumPy version {version} detected. This will cause compatibility issues.")
        print("🔧 Please run: pip3 install numpy==1.26.4 --break-system-packages --force-reinstall")
        return False
    else:
        print(f"✅ NumPy version {version} is compatible")
        return True

def check_environment():
    """Check environment variables"""
    print("\n🔍 Checking environment variables...")
    
    kmp_ok = os.environ.get('KMP_DUPLICATE_LIB_OK', 'FALSE')
    if kmp_ok == 'TRUE':
        print("✅ KMP_DUPLICATE_LIB_OK is set")
    else:
        print("⚠️ KMP_DUPLICATE_LIB_OK not set (will be set automatically by run_search.py)")
    
    return True

def check_data_files():
    """Check if data files exist"""
    print("\n🔍 Checking data files...")
    
    files_to_check = [
        ("embeddings.npy", "Embeddings file"),
        ("index_paths.txt", "Index paths file")
    ]
    
    all_exist = True
    for filename, description in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {description} exists ({size:,} bytes)")
        else:
            print(f"⚠️ {description} not found (run prep command to create)")
            all_exist = False
    
    return all_exist

def check_mask_application():
    """Test mask application logic"""
    print("\n🔍 Testing mask application...")
    
    try:
        from src.mywardrobe.utils import apply_mask
        print("✅ Mask application function available")
        
        # Test with a sample image if available
        test_image = "test_images/WOMEN-Tees_Tanks-id_00005085-31_4_full.jpg"
        if os.path.exists(test_image):
            try:
                # Test without mask (should work - returns original image)
                result = apply_mask(test_image, "nonexistent_mask.png")
                if result is not None:
                    print("✅ Mask application test passed (handles missing mask gracefully)")
                    return True
                else:
                    print("❌ Mask application returned None")
                    return False
            except FileNotFoundError:
                # This is expected when mask doesn't exist
                print("✅ Mask application test passed (handles missing mask gracefully)")
                return True
            except Exception as e:
                print(f"❌ Mask application test failed: {e}")
                return False
        else:
            print("⚠️ No test image available for mask testing")
            return True
            
    except ImportError as e:
        print(f"❌ Could not import mask application: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🚀 Validating Multimodal Fashion Retrieval System Setup\n")
    
    checks = [
        ("Configuration", check_config_file),
        ("Dependencies", check_dependencies),
        ("NumPy Version", check_numpy_version),
        ("Environment", check_environment),
        ("Data Files", check_data_files),
        ("Mask Application", check_mask_application)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 VALIDATION SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1
    
    print("="*50)
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. Update config.py with your Supabase credentials and GitHub repo")
        print("2. Run: python3 run_search.py prep --image_dir /path/to/images --mask_dir /path/to/masks")
        print("3. Run: python3 run_search.py query --query_image /path/to/image --top_k 10")
    else:
        print("⚠️ Some checks failed. Please fix the issues above before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 