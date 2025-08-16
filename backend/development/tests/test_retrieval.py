import os
import io
import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Add backend-deploy to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend-deploy'))

# Create mock data immediately when module loads (for CI/CD)
def create_mock_data():
    """Create mock data for CI/CD environment"""
    # Create mock data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create a mock embeddings file if it doesn't exist
    if not os.path.exists("data/embeddings.npy"):
        # Create a small mock embeddings file
        mock_embeddings = np.random.rand(10, 512).astype(np.float32)
        np.save("data/embeddings.npy", mock_embeddings)
    
    # Create a mock paths file if it doesn't exist
    if not os.path.exists("data/paths.txt"):
        with open("data/paths.txt", "w") as f:
            for i in range(10):
                f.write(f"mock_image_{i}.jpg\n")

# Create mock data immediately
create_mock_data()

from api.app import app

client = TestClient(app)

# Provide a sample image path for testing
SAMPLE_IMAGE = "data/sample.jpg"

@pytest.mark.skipif(not os.path.exists(SAMPLE_IMAGE), reason="Sample image not found")
def test_search_endpoint():
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post(
            "/search",
            files={"file": ("sample.jpg", f, "image/jpeg")},
            data={"text": "green dress"}
        )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 12




def test_chat_endpoint():
    response = client.post(
        "/chat",
        data={"query": "What can I wear to brunch?"}
    )
    assert response.status_code == 200
    reply = response.json().get("reply", "")
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_wardrobe_add_and_get():
    # Add item
    response = client.post(
        "/wardrobe/add",
        data={"user_id": "u1", "product_path": "img.jpg"}
    )
    assert response.status_code == 200
    assert response.json().get("status") == "ok"

    # Get wardrobe
    response = client.get("/wardrobe/u1")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert any(item["product_path"] == "img.jpg" for item in items) 