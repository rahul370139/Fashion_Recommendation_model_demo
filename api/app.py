from fastapi import FastAPI, File, UploadFile, Form
from mywardrobe import load_index, encode_query
import tempfile, shutil

app = FastAPI()
IX, PATHS = load_index("e.npy", "p.txt")

@app.post("/search")
async def search_endpoint(file: UploadFile = File(...), text: str = Form("")):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    shutil.copyfileobj(file.file, tmp)
    vec = encode_query(tmp.name, text)
    D, I = IX.search(vec.numpy(), 5)
    return [{"path": PATHS[i], "score": float(d)} for d, i in zip(D[0], I[0])] 