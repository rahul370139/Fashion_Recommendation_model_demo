import streamlit as st
import requests
import os
import json
import tempfile

API = os.getenv("MW_API_ROOT", "http://localhost:8000")

st.set_page_config(page_title="MyWardrobe", page_icon="🛍️", layout="wide")
st.title("🛍️ MyWardrobe")

# Tabs for Search, Chat, Wardrobe
TABS = ["Search", "Chat", "Wardrobe"]
tab1, tab2, tab3 = st.tabs(TABS)

# --- SEARCH TAB ---
with tab1:
    st.header("Search for Fashion Inspiration")
    uploaded = st.file_uploader("Upload inspiration image", key="search_upload")
    query = st.text_input("Extra text (optional)", "", key="search_text")
    if st.button("Search", key="search_btn") and uploaded:
        with st.spinner("Searching..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                files = {"file": open(tmp.name, "rb")}
                data = {"text": query}
                try:
                    res = requests.post(f"{API}/search", files=files, data=data, timeout=60)
                    res.raise_for_status()
                    hits = res.json()
                    if not hits:
                        st.info("No results found.")
                    else:
                        cols = st.columns(3)
                        for idx, h in enumerate(hits[:9]):
                            with cols[idx % 3]:
                                st.image(h["path"], use_container_width=True)
                                st.caption(f"score {h['score']:.3f}")
                except Exception as e:
                    st.error(f"Search failed: {e}")

# --- CHAT TAB ---
with tab2:
    st.header("Chat with Your Fashion Stylist")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("Ask your stylist a question:", "", key="chat_input")
    if st.button("Send", key="chat_btn") and user_input:
        with st.spinner("Stylist is thinking..."):
            try:
                res = requests.post(f"{API}/chat", data={"query": user_input}, timeout=60)
                res.raise_for_status()
                reply = res.json().get("reply", "No reply.")
                st.session_state["chat_history"].append((user_input, reply))
            except Exception as e:
                st.error(f"Chat failed: {e}")
    # Display chat history
    for q, a in reversed(st.session_state["chat_history"]):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Stylist:** {a}")
        st.markdown("---")

# --- WARDROBE TAB ---
with tab3:
    st.header("Your Wardrobe")
    user_id = st.text_input("Enter your user ID:", "demo_user", key="wardrobe_user")

    st.subheader("Upload Clothing Image to Wardrobe")
    wardrobe_upload = st.file_uploader("Upload clothing image", key="wardrobe_upload")
    if st.button("Upload to Wardrobe", key="upload_wardrobe_btn") and wardrobe_upload and user_id:
        with st.spinner("Uploading image to wardrobe..."):
            try:
                # Use requests-toolbelt to send file as multipart/form-data
                import requests
                from requests_toolbelt.multipart.encoder import MultipartEncoder
                m = MultipartEncoder(fields={
                    "user_id": user_id,
                    "file": (wardrobe_upload.name, wardrobe_upload, wardrobe_upload.type)
                })
                res = requests.post(f"{API}/wardrobe/upload", data=m, headers={"Content-Type": m.content_type}, timeout=60)
                res.raise_for_status()
                st.success("Image uploaded to wardrobe!")
            except Exception as e:
                st.error(f"Failed to upload image: {e}")

    if st.button("View Wardrobe", key="view_wardrobe_btn") and user_id:
        with st.spinner("Loading wardrobe..."):
            try:
                res = requests.get(f"{API}/wardrobe/{user_id}", timeout=30)
                res.raise_for_status()
                items = res.json()
                if not items:
                    st.info("Your wardrobe is empty.")
                else:
                    st.subheader("Your Wardrobe Images")
                    cols = st.columns(3)
                    for idx, item in enumerate(items):
                        # If the product_path is a URL, show as image
                        if item["product_path"].startswith("http"):
                            with cols[idx % 3]:
                                st.image(item["product_path"], use_container_width=True)
                                st.caption(f"Added at: {item['added_at']}")
                        else:
                            st.write(f"Product: {item['product_path']}")
                            st.write(f"Added at: {item['added_at']}")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Failed to load wardrobe: {e}")

    st.subheader("Add Item to Wardrobe (by path)")
    add_path = st.text_input("Product path to add:", "", key="add_path")
    if st.button("Add to Wardrobe", key="add_wardrobe_btn") and add_path and user_id:
        with st.spinner("Adding item..."):
            try:
                res = requests.post(f"{API}/wardrobe/add", data={"user_id": user_id, "product_path": add_path}, timeout=30)
                res.raise_for_status()
                st.success("Item added to wardrobe!")
            except Exception as e:
                st.error(f"Failed to add item: {e}") 