import numpy as np
import tempfile
import shutil
from pathlib import Path
import pytest
from mywardrobe import build_index, load_index, search, encode_query

def fake_data(tmp):
    """Create fake test data - two 128-pix squares so tests run fast"""
    from PIL import Image, ImageDraw
    img_dir = tmp / "img"
    m_dir = tmp / "mask"
    img_dir.mkdir()
    m_dir.mkdir()
    
    for i, color in enumerate(("red", "blue", "green")):
        im = Image.new("RGB", (128, 128), color)
        im.save(img_dir / f"{i}.jpg")
        Image.new("L", (128, 128), 255).save(m_dir / f"{i}_segm.png")
    
    return img_dir, m_dir

def test_build_and_query(tmp_path):
    """Test building index and querying with single image"""
    img, mask = fake_data(tmp_path)
    emb, idx = tmp_path / "e.npy", tmp_path / "p.txt"
    
    # Build index
    build_index(img, mask, emb, idx)
    
    # Load index
    ix, paths = load_index(emb, idx)
    
    # Test single image query
    qvec = encode_query(str(img / "0.jpg"), "")
    D, I = ix.search(qvec.cpu().numpy(), 1)
    
    # Should find the red image (index 0)
    assert paths[I[0][0]].endswith("0.jpg")

def test_image_text_query(tmp_path):
    """Test querying with image + text"""
    img, mask = fake_data(tmp_path)
    emb, idx = tmp_path / "e.npy", tmp_path / "p.txt"
    
    # Build index
    build_index(img, mask, emb, idx)
    
    # Load index
    ix, paths = load_index(emb, idx)
    
    # Test image + text query
    qvec = encode_query(str(img / "1.jpg"), "blue clothing")
    D, I = ix.search(qvec.cpu().numpy(), 1)
    
    # Should find the blue image (index 1)
    assert paths[I[0][0]].endswith("1.jpg")

def test_search_function(tmp_path):
    """Test the high-level search function"""
    img, mask = fake_data(tmp_path)
    emb, idx = tmp_path / "e.npy", tmp_path / "p.txt"
    
    # Build index
    build_index(img, mask, emb, idx)
    
    # Test search function
    D, I, paths = search(str(img / "2.jpg"), "green outfit", 1, emb, idx)
    
    # Should find the green image (index 2)
    assert paths[I[0]].endswith("2.jpg")
    assert len(D) == 1
    assert len(I) == 1 