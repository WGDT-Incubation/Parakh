import streamlit as st
from pdf2image import convert_from_path
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import imagehash
import faiss
import os
import pytesseract
from io import BytesIO

# --- Helper Functions ---

def pdf_to_images(pdf_bytes):
    pages = convert_from_path(pdf_bytes)
    return pages

def detect_and_crop_faces(pil_image):
    img = np.array(pil_image.convert('RGB'))[:, :, ::-1]
    detections = DeepFace.extract_faces(img_path=img, detector_backend='retinaface', enforce_detection=False)
    crops = []
    for d in detections:
        face_img = d['face']
        crops.append(face_img)
    return crops

def face_embedding(face_rgb):
    rep = DeepFace.represent(face_rgb, model_name='ArcFace', enforce_detection=False)
    if isinstance(rep, list):
        rep = rep[0]['embedding']
    return np.array(rep, dtype='float32')

def get_phash(face_rgb):
    pil = Image.fromarray(np.uint8(face_rgb))
    return imagehash.phash(pil)

def extract_name(pil_image):
    # Simple OCR-based name extraction (optional)
    text = pytesseract.image_to_string(pil_image)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[0]  # assume first line has name
    return "Unknown"

def process_uploaded_files(files):
    embeddings, meta = [], []
    for uploaded_file in files:
        filename = uploaded_file.name
        if filename.lower().endswith('.pdf'):
            # Convert PDF -> Images
            with BytesIO(uploaded_file.read()) as f:
                pages = convert_from_path(f)
            images = pages
        else:
            image = Image.open(uploaded_file)
            images = [image]

        for p_idx, pil_img in enumerate(images):
            faces = detect_and_crop_faces(pil_img)
            for f_idx, face_rgb in enumerate(faces):
                emb = face_embedding(face_rgb)
                ph = get_phash(face_rgb)
                name = extract_name(pil_img)
                embeddings.append(emb)
                meta.append({
                    "filename": filename,
                    "page": p_idx,
                    "face_idx": f_idx,
                    "phash": str(ph),
                    "name": name,
                    "face_img": Image.fromarray(np.uint8(face_rgb))
                })
    return np.vstack(embeddings), meta

def find_duplicates(emb_matrix, meta, threshold=0.45, top_k=5):
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_matrix)
    D, I = index.search(emb_matrix, top_k)
    results = []
    for i in range(len(meta)):
        for j in range(1, top_k):
            if I[i, j] == -1: continue
            dist = D[i, j]
            if dist < (threshold ** 2):
                if meta[i]["filename"] != meta[I[i, j]]["filename"]:
                    results.append((meta[i], meta[I[i, j]], dist))
    return results


# Add at top of file (after imports but before st.title)
st.cache_resource
def preload_models():
    DeepFace.build_model("ArcFace")
    DeepFace.build_model("RetinaFace")

preload_models()
st.success("Models preloaded successfully ✅")


# --- Streamlit UI ---

st.set_page_config(page_title="Duplicate Photo Detector", layout="wide")

st.title("🕵️‍♂️ Duplicate Passport Photo Detection App")
st.markdown("""
Upload multiple PDF or image files (forms, documents).  
The app will automatically detect passport-size photos and find duplicates across files, even if names differ.
""")

uploaded_files = st.file_uploader("Upload multiple files", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} files... please wait ⏳")

    with st.spinner("Detecting faces and generating embeddings..."):
        emb_matrix, meta = process_uploaded_files(uploaded_files)

    st.success("✅ Face extraction and embedding complete.")

    st.subheader("🔍 Finding duplicate photos...")
    duplicates = find_duplicates(emb_matrix, meta, threshold=0.45)

    if len(duplicates) == 0:
        st.info("No duplicate faces found.")
    else:
        st.success(f"Found {len(duplicates)} potential duplicate matches.")
        for idx, (m1, m2, dist) in enumerate(duplicates):
            col1, col2, col3 = st.columns([1,1,1.5])
            with col1:
                st.image(m1["face_img"], caption=f"{m1['filename']}\nName: {m1['name']}")
            with col2:
                st.image(m2["face_img"], caption=f"{m2['filename']}\nName: {m2['name']}")
            with col3:
                st.write(f"**Match Score:** {1 - dist:.3f}")
                if m1['name'] != m2['name']:
                    st.error("⚠️ Different Names")
                else:
                    st.success("Same Name (Possible Same Person)")

else:
    st.warning("Please upload PDF or image files to start analysis.")
