import sys
import os
import datetime
import json
from pathlib import Path
import threading
from typing import List, Dict, Optional

# Add the parent directory to Python path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import SUPABASE_URL, SUPABASE_KEY

# Try to import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✅ Supabase client available")
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase client not available, using in-memory database")

supabase: Optional['Client'] = None

def init_supabase():
    global supabase
    if not SUPABASE_AVAILABLE:
        return False
    if not SUPABASE_URL or SUPABASE_URL == "https://your-project-id.supabase.co":
        print("⚠️ SUPABASE_URL not configured")
        return False
    if not SUPABASE_KEY or SUPABASE_KEY == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...":
        print("⚠️ SUPABASE_KEY not configured")
        return False
    if supabase is None:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Supabase: {e}")
            return False
    return True

# In-memory database for development (fallback)
WARDROBE_DB = {}
WARDROBE_LOCK = threading.Lock()

# Optional: File-based persistence
WARDROBE_FILE = Path("data/wardrobe.json")

def _load_wardrobe():
    global WARDROBE_DB
    if WARDROBE_FILE.exists():
        try:
            with open(WARDROBE_FILE, 'r') as f:
                WARDROBE_DB = json.load(f)
            print("✅ Loaded wardrobe data from file")
        except Exception as e:
            print(f"⚠️ Could not load wardrobe data: {e}")
            WARDROBE_DB = {}

def _save_wardrobe():
    try:
        WARDROBE_FILE.parent.mkdir(exist_ok=True)
        with WARDROBE_LOCK:
            with open(WARDROBE_FILE, 'w') as f:
                json.dump(WARDROBE_DB, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Could not save wardrobe data: {e}")

# Load existing data on import
_load_wardrobe()

def add_item(user_id: str, product_path: str):
    """Add an item to user's wardrobe"""
    if init_supabase():
        try:
            item = {
                "user_id": user_id,
                "product_path": product_path,
                "added_at": datetime.datetime.utcnow().isoformat()
            }
            result = supabase.table("wardrobe").insert(item).execute()
            if result.data:
                print(f"✅ Added item to wardrobe for user {user_id} (Supabase)")
                return result.data[0]
            else:
                print("❌ Failed to add item to wardrobe (Supabase)")
                return None
        except Exception as e:
            print(f"❌ Error adding item to wardrobe (Supabase): {e}")
            # fallback to in-memory
    # Fallback to in-memory database
    try:
        with WARDROBE_LOCK:
            if user_id not in WARDROBE_DB:
                WARDROBE_DB[user_id] = []
            item = {
                "user_id": user_id,
                "product_path": product_path,
                "added_at": datetime.datetime.utcnow().isoformat()
            }
            WARDROBE_DB[user_id].append(item)
            _save_wardrobe()
        print(f"✅ Added item to wardrobe for user {user_id} (in-memory)")
        return item
    except Exception as e:
        print(f"❌ Error adding item to wardrobe (in-memory): {e}")
        return None

def list_items(user_id: str):
    """List all items in user's wardrobe"""
    if init_supabase():
        try:
            result = supabase.table("wardrobe").select("*").eq("user_id", user_id).execute()
            if result.data:
                return result.data
            else:
                return []
        except Exception as e:
            print(f"❌ Error listing wardrobe items (Supabase): {e}")
            # fallback to in-memory
    # Fallback to in-memory database
    try:
        return WARDROBE_DB.get(user_id, [])
    except Exception as e:
        print(f"❌ Error listing wardrobe items (in-memory): {e}")
        return []

def remove_item(user_id: str, product_path: str):
    """Remove an item from user's wardrobe"""
    if init_supabase():
        try:
            result = supabase.table("wardrobe").delete().eq("user_id", user_id).eq("product_path", product_path).execute()
            if result.data:
                print(f"✅ Removed item from wardrobe for user {user_id} (Supabase)")
                return True
            else:
                print("❌ Failed to remove item from wardrobe (Supabase)")
                return False
        except Exception as e:
            print(f"❌ Error removing item from wardrobe (Supabase): {e}")
            # fallback to in-memory
    # Fallback to in-memory database
    try:
        with WARDROBE_LOCK:
            if user_id in WARDROBE_DB:
                WARDROBE_DB[user_id] = [item for item in WARDROBE_DB[user_id] if item["product_path"] != product_path]
                _save_wardrobe()
        print(f"✅ Removed item from wardrobe for user {user_id} (in-memory)")
        return True
    except Exception as e:
        print(f"❌ Error removing item from wardrobe (in-memory): {e}")
        return False

def test_connection():
    """Test database connection"""
    if init_supabase():
        try:
            test_user = "test_user"
            test_path = "test_path"
            added_item = add_item(test_user, test_path)
            items = list_items(test_user)
            remove_item(test_user, test_path)
            if added_item and items:
                print("✅ Supabase connection test successful!")
                return True
            else:
                print("❌ Supabase connection test failed")
                return False
        except Exception as e:
            print(f"❌ Supabase connection test failed: {e}")
            # fallback to in-memory
    # Test in-memory database
    try:
        test_user = "test_user"
        test_path = "test_path"
        add_item(test_user, test_path)
        items = list_items(test_user)
        remove_item(test_user, test_path)
        print("✅ In-memory database connection test successful")
        return True
    except Exception as e:
        print(f"❌ Database connection test failed (in-memory): {e}")
        return False 