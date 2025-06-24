import sys
import os

# Add the parent directory to Python path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client
    
    # Validate configuration
    if SUPABASE_URL == "YOUR_SUPABASE_URL" or SUPABASE_KEY == "YOUR_SUPABASE_KEY":
        print("⚠️ Supabase credentials not configured in config.py")
        print("   Please update config.py with your Supabase URL and key")
        supabase = None
    else:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")

except ImportError as e:
    print(f"⚠️ Could not import Supabase configuration: {e}")
    print("   Please install supabase-py: pip install supabase-py")
    supabase = None
except Exception as e:
    print(f"⚠️ Error initializing Supabase client: {e}")
    supabase = None

def add_item(user_id, product_id):
    """Add an item to user's wardrobe"""
    if supabase is None:
        print("❌ Supabase not configured. Cannot add item.")
        return None
    
    try:
        return supabase.table("wardrobe").insert({
            "user_id": user_id, 
            "product_id": product_id
        }).execute()
    except Exception as e:
        print(f"❌ Error adding item to wardrobe: {e}")
        return None

def list_items(user_id):
    """List all items in user's wardrobe"""
    if supabase is None:
        print("❌ Supabase not configured. Cannot list items.")
        return None
    
    try:
        return supabase.table("wardrobe").select("*").eq("user_id", user_id).execute()
    except Exception as e:
        print(f"❌ Error listing wardrobe items: {e}")
        return None

def test_connection():
    """Test Supabase connection"""
    if supabase is None:
        return False
    
    try:
        # Try a simple query to test connection
        result = supabase.table("wardrobe").select("count", count="exact").limit(1).execute()
        print("✅ Supabase connection test successful")
        return True
    except Exception as e:
        print(f"❌ Supabase connection test failed: {e}")
        return False 