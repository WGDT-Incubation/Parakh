#!/usr/bin/env python3
"""
app_forgery_demo_v2.py

Streamlit UI for interactive document forgery detection (POC).
- Upload PDF or image
- See side-by-side original and detected result
- Adjust sensitivity with sliders
"""

import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import tempfile

# --------------------------
# Helper Functions
# --------------------------

from io import BytesIO

def extract_images_from_pdf(pdf_bytes):
    """Extract images from PDF and return list of PIL images."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                base_image = doc.extract_image(img[0])
                image_bytes = base_image["image"]
                # ✅ FIX: Use BytesIO to open the image correctly
                pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append(pil_img)
    return images


def detect_white_marker(cv_img, sat_thresh=30, val_thresh=200, min_area=150):
    """Detect low-saturation and high-brightness patches (white-marker or erasures)."""
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = cv2.bitwise_and(
        (s < sat_thresh).astype("uint8") * 255,
        (v > val_thresh).astype("uint8") * 255
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv_img.copy()
    suspicious = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            suspicious.append((x, y, w, h, area))
    return out, suspicious


# --------------------------
# Streamlit UI Layout
# --------------------------

st.set_page_config(page_title="Document Forgery Detection Demo", layout="wide")

st.title("🕵️ Document Forgery Detection — Interactive POC Demo")
st.markdown("""
This demo shows how AI can **detect manual or digital tampering** (like white-marker or erasure edits)  
in scanned or uploaded documents.

Upload a **PDF or image**, and adjust the detection sensitivity below.  
Red boxes mark **potentially tampered areas**.
""")

# Sidebar controls
st.sidebar.header("🎛️ Detection Sensitivity")
sat = st.sidebar.slider("Saturation Threshold (lower = more sensitive)", 10, 100, 30, 5)
val = st.sidebar.slider("Brightness Threshold (higher = stricter)", 150, 255, 200, 5)
min_area = st.sidebar.slider("Minimum Area (px²)", 50, 1000, 150, 50)

uploaded_file = st.file_uploader("📄 Upload a document (PDF or image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    file_type = uploaded_file.name.lower().split(".")[-1]

    # --- Handle PDF ---
    if file_type == "pdf":
        pdf_bytes = uploaded_file.read()
        images = extract_images_from_pdf(pdf_bytes)
        if not images:
            st.warning("No images found in this PDF.")
        else:
            st.success(f"Extracted {len(images)} image(s) from PDF.")

            for i, pil_img in enumerate(images):
                st.markdown(f"### 📄 Page {i+1}")
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                marked_img, suspicious = detect_white_marker(cv_img, sat_thresh=sat, val_thresh=val, min_area=min_area)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(pil_img, caption="Original", use_column_width=True)
                with col2:
                    st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB), caption=f"Detected (Regions: {len(suspicious)})", use_column_width=True)

                if len(suspicious) == 0:
                    st.success("✅ No suspicious regions found.")
                elif len(suspicious) < 10:
                    st.warning(f"⚠️ {len(suspicious)} possible small tampering(s) found.")
                else:
                    st.error(f"🚨 {len(suspicious)} suspicious areas detected — possible significant editing!")
                st.markdown("---")

    # --- Handle Image Upload ---
    else:
        pil_img = Image.open(uploaded_file).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        marked_img, suspicious = detect_white_marker(cv_img, sat_thresh=sat, val_thresh=val, min_area=min_area)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB),
                     caption=f"Detected (Regions: {len(suspicious)})",
                     use_column_width=True)

        if len(suspicious) == 0:
            st.success("✅ No suspicious regions detected.")
        elif len(suspicious) < 10:
            st.warning(f"⚠️ {len(suspicious)} possible white-marker edits detected.")
        else:
            st.error(f"🚨 High likelihood of tampering — {len(suspicious)} suspicious regions found!")
else:
    st.info("Upload a file from the left panel to begin the analysis.")
