#!/usr/bin/env python3
"""
Wrapper script to run the multimodal fashion retrieval system
with automatic environment setup and NumPy version checking.
"""

import os
import sys
import subprocess
import numpy as np

def check_numpy_version():
    """Check if NumPy version is compatible"""
    version = np.__version__
    if version.startswith('2.'):
        print(f"❌ NumPy version {version} detected. This will cause compatibility issues.")
        print("🔧 Please run: pip3 install numpy==1.26.4 --break-system-packages --force-reinstall")
        return False
    print(f"✅ NumPy version {version} is compatible")
    return True

def set_environment():
    """Set required environment variables"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("✅ Environment variables set")

def run_command(cmd):
    """Run a command with proper environment"""
    set_environment()
    
    if not check_numpy_version():
        return False
    
    try:
        # Add the src directory to Python path
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"src:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = 'src'
        
        result = subprocess.run(cmd, shell=True, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_search.py [prep|query|finetune] [options...]")
        print("\nExamples:")
        print("  python3 run_search.py prep --image_dir /path/to/images --mask_dir /path/to/masks")
        print("  python3 run_search.py query --query_image /path/to/image --top_k 10")
        print("  python3 run_search.py query --query_image /path/to/image --query_text 'casual summer top' --top_k 10")
        return
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == "prep":
        cmd = f"python3 main.py prep {' '.join(args)}"
    elif command == "query":
        cmd = f"python3 main.py query {' '.join(args)}"
    elif command == "finetune":
        cmd = f"python3 main.py finetune {' '.join(args)}"
    else:
        print(f"❌ Unknown command: {command}")
        return
    
    print(f"🚀 Running: {cmd}")
    success = run_command(cmd)
    
    if success:
        print("✅ Command completed successfully!")
    else:
        print("❌ Command failed!")

if __name__ == "__main__":
    main() 