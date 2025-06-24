from fastapi import FastAPI, File, UploadFile, Form
from src.mywardrobe.retrieval_mock import load_index, encode_query_mock
import tempfile, shutil
import os

app = FastAPI()

# Check if mock index files exist, if not create them
if not (os.path.exists("embeddings_mock.npy") and os.path.exists("index_paths_mock.txt")):
    print("⚠️ Mock index files not found. Please run test_mock.py first to create them.")
    IX, PATHS = None, []
else:
    IX, PATHS = load_index("embeddings_mock.npy", "index_paths_mock.txt")

@app.get("/")
async def root():
    return {"message": "MyWardrobe API - Mock Version", "status": "running"}

@app.post("/search")
async def search_endpoint(file: UploadFile = File(...), text: str = Form("")):
    if IX is None:
        return {"error": "Index not loaded. Please run test_mock.py first."}
    
    # Save uploaded file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    
    # Use mock query encoding
    vec = encode_query_mock(tmp.name, text)
    
    # Search
    D, I = IX.search(vec.numpy(), 5)
    
    # Clean up temp file
    os.unlink(tmp.name)
    
    # Return results
    results = []
    for d, i in zip(D[0], I[0]):
        results.append({
            "path": PATHS[i], 
            "filename": os.path.basename(PATHS[i]),
            "score": float(d)
        })
    
    return results

@app.get("/health")
async def health_check():
    return {"status": "healthy", "index_loaded": IX is not None} 