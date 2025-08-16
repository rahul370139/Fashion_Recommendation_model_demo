from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, shutil
from src.mywardrobe.retrieval import load_index, encode_query
from src.mywardrobe.db import add_item, list_items, init_supabase
from api.chains import chat_with_stylist
import os

# Load index and paths as singletons
IX, PATHS = load_index("data/embeddings.npy", "data/paths.txt")

app = FastAPI(title="MyWardrobe API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- search --------------------------------------------------------------
@app.post("/search")
async def search(
    file: UploadFile = File(...),
    text: str = Form("")
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    shutil.copyfileobj(file.file, tmp)
    vec = encode_query(tmp.name, text).cpu().numpy()
    D, I = IX.search(vec, 12)
    os.unlink(tmp.name)
    return [
        {"path": PATHS[i], "score": float(d)}
        for d, i in zip(D[0], I[0])
    ]

# --- wardrobe CRUD -------------------------------------------------------
@app.post("/wardrobe/add")
async def add_to_wardrobe(user_id: str = Form(...), product_path: str = Form(...)):
    add_item(user_id, product_path)
    return {"status": "ok"}

@app.get("/wardrobe/{user_id}")
async def wardrobe(user_id: str):
    return list_items(user_id)

@app.post("/wardrobe/upload")
async def upload_wardrobe_image(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not init_supabase():
        raise HTTPException(500, "Supabase not available or not configured")

    from src.mywardrobe.db import supabase  # 🔄 use package path

    contents = await file.read()

    import uuid, os
    ext      = os.path.splitext(file.filename)[1]
    filename = f"{user_id}/{uuid.uuid4().hex}{ext}"
    bucket   = "wardrobe-images"

    # ---- upload --------------------------------------------------------
    try:
        res = supabase.storage.from_(bucket).upload(
            filename,
            contents,
            {"content-type": file.content_type or "image/jpeg"}
        )
    except Exception as e:
        raise HTTPException(500, f"Supabase upload failed: {e}")

    # If upload fails, res will be None or will raise above
    public_url = supabase.storage.from_(bucket).get_public_url(filename)
    item = add_item(user_id, public_url)
    return {"status": "ok", "url": public_url, "item": item}

# --- chat stylist (LangChain) -------------------------------------------
@app.post("/chat")
async def chat(query: str = Form(...)):
    answer = await chat_with_stylist(query)
    return {"reply": answer} 