"""
Environment-based configuration for Multimodal Fashion Retrieval System
Supports both environment variables and .env files
"""

import os
from pathlib import Path

# Try to load from .env file if it exists
env_file = Path(".env")
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded configuration from .env file")
    except ImportError:
        print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

# =============================================================================
# Configuration with environment variable fallbacks
# =============================================================================

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project-id.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")

# GitHub Configuration
GITHUB_REPO_SLUG = os.getenv("GITHUB_REPO_SLUG", "rahul370139/Fashion_Recommendation_model_demo")
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO_SLUG}"

# System Configuration
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
DEVICE = os.getenv("DEVICE", "auto")

# FAISS Configuration
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
FAISS_DIMENSION = int(os.getenv("FAISS_DIMENSION", "512"))

# Search Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.5"))

# File Paths
DEFAULT_EMBEDDINGS_FILE = os.getenv("DEFAULT_EMBEDDINGS_FILE", "embeddings.npy")
DEFAULT_PATHS_FILE = os.getenv("DEFAULT_PATHS_FILE", "index_paths.txt")

# Environment Variables
KMP_DUPLICATE_LIB_OK = os.getenv("KMP_DUPLICATE_LIB_OK", "TRUE")

def validate_config():
    """Validate that all required configuration is set"""
    issues = []
    
    if SUPABASE_URL == "https://your-project-id.supabase.co":
        issues.append("SUPABASE_URL not configured (set SUPABASE_URL env var or update config)")
    
    if SUPABASE_KEY == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...":
        issues.append("SUPABASE_KEY not configured (set SUPABASE_KEY env var or update config)")
    
    if GITHUB_REPO_SLUG == "rahul370139/Fashion_Recommendation_model_demo":
        issues.append("GITHUB_REPO_SLUG not configured (set GITHUB_REPO_SLUG env var or update config)")
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nTo fix:")
        print("1. Set environment variables:")
        print("   export SUPABASE_URL='your-actual-url'")
        print("   export SUPABASE_KEY='your-actual-key'")
        print("   export GITHUB_REPO_SLUG='rahul370139/Fashion_Recommendation_model_demo'")
        print("2. Or create a .env file with these variables")
        return False
    
    print("✅ Configuration validated successfully!")
    return True

if __name__ == "__main__":
    validate_config()