import streamlit as st
import fitz
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import json
from io import BytesIO
import pandas as pd
import base64
import subprocess
import sys


import tempfile
import os
import faiss
from tqdm import tqdm
from insightface.app import FaceAnalysis
from pdf2image import convert_from_path
from pathlib import Path




# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="PARAKH - AI Document Analytics",
    layout="wide",
    page_icon="📄",
)

# ----------------------------
# CUSTOM CSS - Enhanced UI 
# ----------------------------
st.markdown("""
    <style>
        /* --- Base Layout --- */
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #eef2f7 0%, #ffffff 100%);
            overflow-x: hidden;
        }
        .main {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }

        /* --- Header Logos --- */
        .header {
            text-align: center;
            margin-top: 5px;
            margin-bottom: 10px;
        }
        .header img {
            display: block;
            margin: 0 auto;
        }
        .cag-logo {
            max-width: 150px;
            margin-bottom: 5px;
            animation: fadeIn 1s ease-in-out;
        }
        .parakh-logo {
            max-width: 360px;
            animation: fadeIn 1.5s ease-in-out;
        }
        .tagline {
            font-size: 20px;
            color: #0c2340;
            font-weight: 600;
            text-align: center;
            margin-top: 5px;
            margin-bottom: 20px;
            letter-spacing: 0.5px;
        }

        /* --- Intro Card --- */
        .intro-card {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            padding: 25px 40px;
            width: 70%;
            margin: 0 auto;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .intro-card:hover {
            transform: scale(1.01);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }
        .intro-card h3 {
            color: #0c2340;
            margin-bottom: 10px;
        }
        .intro-card ul {
            padding-left: 25px;
            line-height: 1.7;
            color: #333;
        }

        /* --- Uploader Section --- */
        .upload-section {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            padding: 20px 35px;
            width: 70%;
            margin: 20px auto;
            text-align: center;
        }

        /* --- Footer --- */
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9fafc;
            text-align: center;
            font-size: 14px;
            color: #444;
            padding: 8px 0;
            border-top: 1px solid #ddd;
        }
                /* --- Streamlit Default Padding Override --- */
        .block-container {
            padding-top: 0rem !important;       /* Remove top gap */
            padding-bottom: 1rem !important;
            margin-top: 45px !important;       /* Pull slightly upward */
        }

        /* --- Header Margin Tweak --- */
        .header {
            margin-top: 0px !important;         /* Remove extra top margin */
            margin-bottom: 8px !important;
        }
        /* --- Animations --- */
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------------
# HEADER SECTION (LOGO + TITLE) - updated
# ----------------------------
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert local logos to base64
cag_logo_b64 = get_base64_image("image/caglogo.png")
parakh_logo_b64 = get_base64_image("image/parakh.png")

# --- HEADER HTML (Correctly Sized + Centered) ---
st.markdown(f"""
    <style>
        .header {{
            text-align: center;
            margin-top: 5px;
            margin-bottom: 15px;
        }}
        .cag-logo {{
            width: 100px;            /* 👈 fixed size */
            height: auto;
            margin-bottom: 8px;
            border-radius: 8px;
            animation: fadeIn 1s ease-in-out;
        }}
        .parakh-logo {{
            width: 360px;           /* 👈 proportional size */
            height: auto;
            animation: fadeIn 1.5s ease-in-out;
        }}
        .tagline {{
            font-size: 20px;
            color: #0c2340;
            font-weight: 600;
            text-align: center;
            margin-top: 6px;
            margin-bottom: 25px;
            letter-spacing: 0.6px;
        }}
        @keyframes fadeIn {{
            from {{opacity: 0; transform: translateY(-10px);}}
            to {{opacity: 1; transform: translateY(0);}}
        }}
    </style>

    <div class="header">
        <img src="data:image/png;base64,{cag_logo_b64}" class="cag-logo" alt="CAG Logo">
        <br>
        <img src="data:image/png;base64,{parakh_logo_b64}" class="parakh-logo" alt="Parakh Logo">
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# INTRODUCTORY / FEATURE SECTION (TEXT IN SAME ALIGNMENT AS IMAGE) - Updated
# ----------------------------
st.markdown("""
<style>
    .intro-card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        padding: 15px 25px;               /* ↓ Reduced padding */
        width: 65%;                       /* ↓ Slightly narrower */
        margin: 5px auto 10px auto;
        text-align: left;
        transition: box-shadow 0.2s ease;
    }
    .intro-card:hover {
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .intro-card h3 {
        color: #0c2340;
        font-size: 18px;                  /* ↓ Smaller heading */
        margin-bottom: 4px;
    }
    .intro-card p {
        font-size: 14px;                  /* ↓ Reduced body text */
        color: #333;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    .intro-card ul {
        font-size: 13px;                  /* ↓ Compact list */
        color: #222;
        line-height: 1.4;
        padding-left: 18px;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .intro-card li {
        margin-bottom: 3px;               /* ↓ Tight line spacing */
    }
</style>

<div class="intro-card">
    <h3>👋 Hello, Auditor!</h3>
    <p>
        Upload a document (<b>PDF, DOCX, or Image</b>) and I’ll analyze its:
    </p>
    <ul>
        <li><b>Structural Integrity</b> – layout consistency, missing pages, duplications</li>
        <li><b>Semantic Coherence</b> – logical flow, contradictory or incomplete content</li>
        <li><b>Authenticity Indicators</b> – repeated signatures, overwritten text, metadata mismatches</li>
    </ul>
</div>
""", unsafe_allow_html=True)



# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.title("📂 Configuration")
page = st.sidebar.radio(
    "Select a view",
    ["Forgery Detection", "Doc Authenticity", "Duplicate Photo", "Blur Detection", "Document Validation"]
)


st.sidebar.markdown("---")
st.sidebar.caption("AI Document Integrity Demo")

# ----------------------------
# COMMON HELPER FUNCTIONS
# ----------------------------
def run_validator(input_dir, validator_script="azure_doc_validator.py"):
    output_path = Path(input_dir) / "document_validation_results.json"
    cmd = [
        sys.executable,  # ensures Streamlit's Python env is used
        validator_script,
        "--input_dir", str(input_dir),
        "--output", str(output_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        st.text_area("📜 Validator Output Log", proc.stdout + "\n" + proc.stderr, height=200)
        if proc.returncode == 0 and output_path.exists():
            st.success(f"✅ Validation complete. Results saved to {output_path}")
            return output_path
        else:
            st.error("❌ Validator failed or no JSON generated.")
            return None
    except Exception as e:
        st.error(f"Error running validator: {e}")
        return None


def load_json_results(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not read JSON: {e}")
        return None


def render_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    imgs = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(BytesIO(pix.tobytes("png")))
        imgs.append(img)
    return imgs


def detect_white_marker(cv_img, sat_thresh=30, val_thresh=200, min_area=150):
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

def detect_blur(cv_img, threshold=100):
    """Detect if an image is blurred using Laplacian variance."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurred = variance < threshold
    return variance, is_blurred


def highlight_blur_regions(cv_img, window_size=15, threshold=50):
    """
    Highlight local blurred regions using sliding window variance.
    Returns image with red boxes drawn around blurred regions.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred_map = np.zeros_like(gray, dtype=np.float32)

    # compute variance in local windows
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            patch = gray[y:y + window_size, x:x + window_size]
            variance = cv2.Laplacian(patch, cv2.CV_64F).var()
            blurred_map[y:y + window_size, x:x + window_size] = variance

    out = cv_img.copy()
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            if blurred_map[y:y + window_size, x:x + window_size].mean() < threshold:
                cv2.rectangle(out, (x, y), (x + window_size, y + window_size), (0, 0, 255), 1)

    overall_score = np.mean(blurred_map)
    return out, overall_score


def extract_images_from_pdf(pdf_bytes):
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                base_image = doc.extract_image(img[0])
                image_bytes = base_image["image"]
                pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append(pil_img)
    return images
# ---------- Image Edit Utility Functions ----------

def convert_pdf_to_image(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp.flush()
        images = convert_from_path(tmp.name)
    return images[0]

def perform_ela(image, quality=90):
    """Perform Error Level Analysis"""
    image = image.convert("RGB")
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    ela_image = Image.open(temp_filename)
    diff = ImageChops.difference(image, ela_image)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 0
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)
    return ela_image

def generate_heatmap_overlay(original, ela_image, threshold=120, alpha=0.45):
    """Create heatmap overlay for edited regions"""
    ela_gray = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(ela_gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(original), 1 - alpha, heatmap, alpha, 0)
    suspicious_pixels = np.count_nonzero(mask > 50)
    return Image.fromarray(overlay), suspicious_pixels

def edge_noise_analysis(image):
    """Calculate edge and noise consistency"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def clone_patch_inconsistency(image):
    """Detect patch duplication (copy-paste regions)"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return len(keypoints)

def preprocess_image(uploaded_file):
    if uploaded_file.type == "application/pdf":
        image = convert_pdf_to_image(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert("RGB")
    return image


def style_status(val):
    if val == "PASS":
        return "✅ PASS"
    elif val == "FAIL":
        return "❌ FAIL"
    elif val == "REVIEW":
        return "⚠️ REVIEW"
    else:
        return val
    
    # --------------------------
# Initialize InsightFace (ArcFace + RetinaFace)
# --------------------------
@st.cache_resource
def load_face_model():
    app = FaceAnalysis(name='buffalo_l')  # includes RetinaFace + ArcFace
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

face_app = load_face_model()

# --------------------------
# Helper functions - Duplicate photos finding
# --------------------------
def convert_pdf_to_images(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        valid_pages = [p for p in pages if p.size[0] > 100 and p.size[1] > 100]
        return valid_pages
    except Exception as e:
        st.warning(f"PDF conversion failed for {pdf_path}: {e}")
        return []

def extract_faces_and_embeddings(img_pil, filename):
    img = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img)
    results = []
    for f in faces:
        emb = f.normed_embedding
        bbox = f.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        face_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        results.append({
            "file": os.path.basename(filename),
            "embedding": emb,
            "face_img": face_img
        })
    return results

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ----------------------------
# PAGE 1 - FORGERY DETECTION
# ----------------------------
if page == "Forgery Detection":
    
    #st.title("🕵️‍♂️ CAG PARAKH - Document Forgery Detection")
    #st.write("""
    #Upload a **PDF or image** to detect suspicious erasures or white-marker regions.
    #Red boxes highlight potential tampered zones.
    #""")

    sat = st.sidebar.slider("Saturation Threshold", 10, 100, 30, 5)
    val = st.sidebar.slider("Brightness Threshold", 150, 255, 200, 5)
    min_area = st.sidebar.slider("Minimum Area (px²)", 50, 1000, 150, 50)

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_type = uploaded_file.name.lower().split(".")[-1]

        if file_type == "pdf":
            pdf_bytes = uploaded_file.read()
            images = extract_images_from_pdf(pdf_bytes)
            if not images:
                st.warning("No images found in PDF.")
            else:
                st.success(f"Extracted {len(images)} page image(s).")
                for i, pil_img in enumerate(images):
                    st.subheader(f"Page {i+1}")
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    marked_img, suspicious = detect_white_marker(cv_img, sat, val, min_area)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(pil_img, caption="Original", use_container_width=True)
                    with col2:
                        st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB),
                                 caption=f"Detected (Regions: {len(suspicious)})",
                                 use_container_width=True)
                    if len(suspicious) == 0:
                        st.success("✅ Clean document")
                    elif len(suspicious) < 10:
                        st.warning(f"⚠️ {len(suspicious)} small suspicious areas detected.")
                    else:
                        st.error(f"🚨 {len(suspicious)} suspicious zones found — likely tampering.")
                    st.markdown("---")

        else:
            pil_img = Image.open(uploaded_file).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            marked_img, suspicious = detect_white_marker(cv_img, sat, val, min_area)
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Original", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB),
                         caption=f"Detected (Regions: {len(suspicious)})",
                         use_container_width=True)
            if len(suspicious) == 0:
                st.success("✅ No suspicious regions detected.")
            elif len(suspicious) < 10:
                st.warning(f"⚠️ {len(suspicious)} possible edits detected.")
            else:
                st.error(f"🚨 {len(suspicious)} suspicious zones found — possible tampering detected.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# PAGE 2 - DOCUMENT VALIDATION
# ----------------------------

elif page == "Document Validation":
    
    #st.title("🔎 CAG PARAKH - Document Validation")

    # ----------------------------
    # SESSION STATE SETUP
    # ----------------------------
    if "validation_data" not in st.session_state:
        st.session_state.validation_data = None
    if "validation_df" not in st.session_state:
        st.session_state.validation_df = None
    if "input_dir" not in st.session_state:
        st.session_state.input_dir = None

    # ----------------------------
    # INPUT FIELD + RUN BUTTON
    # ----------------------------
    input_dir = st.text_input("📂 Enter directory path containing PDFs:", value="input_docs")
    run_button = st.button("▶️ Run Validator")

    # ----------------------------
    # RUN VALIDATOR AND STORE RESULTS
    # ----------------------------
    if run_button:
        if not Path(input_dir).exists():
            st.error(f"Input directory not found: {input_dir}")
        else:
            with st.spinner("🟢 Running CAG Parakh validator..."):
                result_json = run_validator(input_dir)
            if result_json and Path(result_json).exists():
                data = load_json_results(result_json)
                if data:
                    results = data.get("results", [])
                    df = pd.DataFrame(results)
                    if not df.empty:
                        df["File"] = df["file"].apply(lambda x: Path(x).name)
                        df["Score"] = df["score"].apply(lambda x: f"{round(x)} %")
                        df["Status"] = df["status"].apply(style_status)
                        df.rename(columns={"detected_name": "Detected Name"}, inplace=True)
                        df = df[["File", "Detected Name", "Score", "Status"]]

                        # Save in session state
                        st.session_state.validation_data = data
                        st.session_state.validation_df = df
                        st.session_state.input_dir = input_dir

                        st.success("✅ Validation completed. You can now explore results below!")

    # ----------------------------
    # SHOW RESULTS IF ALREADY AVAILABLE
    # ----------------------------
    if st.session_state.validation_df is not None:
        df = st.session_state.validation_df
        input_dir = st.session_state.input_dir

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write(f"**Summary:** ✅ {(df['Status'].str.contains('PASS')).sum()} PASS | ❌ {(df['Status'].str.contains('FAIL')).sum()} FAIL (out of {len(df)})")
        st.markdown("---")

        # ----------------------------
        # DOCUMENT PREVIEW SECTION
        # ----------------------------
        st.subheader("📘 Document Preview")
        selected_file = st.selectbox("Select document to preview:", options=df["File"].tolist(), key="selected_preview_file")

        if selected_file:
            pdf_path = Path(input_dir) / selected_file
            if pdf_path.exists():
                pages = render_pdf_pages(pdf_path)
                for i, p in enumerate(pages):
                    st.image(p, caption=f"Page {i+1}", use_container_width=True)
            else:
                st.warning("⚠️ PDF not found in specified directory.")

    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Doc Authenticity":
    
    #st.title("CAG PARAKH - Find Document Authenticity")
    st.write("""
    Upload a PDF / Image to detect if document is edited using any tool.
    """)
    uploaded_file = st.file_uploader("📁 Upload Document Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file:
        image = preprocess_image(uploaded_file)
        st.image(image, caption="Uploaded Document", use_container_width=True)

        st.subheader("📸 Found Edited Section")
        ela_img = perform_ela(image, quality=90)
        st.image(ela_img, caption="ELA Visualization (brighter = possible edits)", use_container_width=True)
        ela_array = np.array(ela_img)
        ela_mean = np.mean(ela_array)
        st.write(f"ELA Mean Intensity: **{ela_mean:.2f}**")

        st.subheader("🧩 Edge / Noise Consistency")
        noise_score = edge_noise_analysis(image)
        st.write(f"Noise Variance: **{noise_score:.2f}**")

        st.subheader("🧬 Clone / Patch Consistency")
        keypoint_count = clone_patch_inconsistency(image)
        st.write(f"Keypoint Features Detected: **{keypoint_count}**")

        st.subheader("🔥 Heatmap")
        heatmap_img, suspicious_regions = generate_heatmap_overlay(image, ela_img, threshold=120, alpha=0.5)
        st.image(heatmap_img, caption="Suspicious Regions Highlighted", use_container_width=True)

        # ---------- Improved Confidence Scoring ----------
        st.subheader("⚖️ Step 5: Final Confidence & Verdict")

        # --- Enhanced ELA metric: focus on strong local brightness (top 5%) ---
        ela_gray = cv2.cvtColor(np.array(ela_img), cv2.COLOR_RGB2GRAY)
        ela_sorted = np.sort(ela_gray.flatten())

        #top5_mean = np.mean(ela_sorted[int(len(ela_sorted) * 0.95):])  # top 5% brightest pixels
        #localized_ela_score = min(100, top5_mean * 1.2)  # amplify localized difference
        top1_mean = np.mean(ela_sorted[int(len(ela_sorted)*0.99):])   # top 1 % pixels
        localized_ela_score = min(100, top1_mean * 2.5)               # give it more weight


        # --- Noise / Edge / Clone Factors ---
        noise_factor = 100 - min(100, (noise_score / 6))         # smoother = suspicious
        #region_factor = min(70, suspicious_regions / 120)         # more bright region = suspicious
        region_factor = min(100, suspicious_regions / 60)   # was /120

        clone_factor = 100 - min(100, (keypoint_count / 15))      # low keypoints = suspicious texture

        # --- Combine weights ---
        confidence_score = np.clip(
            (localized_ela_score * 0.45) +
            (noise_factor * 0.25) +
            (region_factor * 0.2) +
            (clone_factor * 0.1),
            0, 100
        )

        # --- Verdict Threshold ---
        #verdict = "FORGED / EDITED" if confidence_score >= 40 else "GENUINE / UNTAMPERED"
        verdict = "FORGED / EDITED" if confidence_score >= 45 else "GENUINE / UNTAMPERED"

        # --- Display Result ---
        st.progress(confidence_score / 100)
        if verdict == "FORGED / EDITED":
            st.error(f"🚨 Document appears **{verdict}**")
            st.caption("Detected high ELA intensity in localized region → likely digitally added or modified content.")
        else:
            st.success(f"✅ Document appears **{verdict}**")
            st.caption("No major localized editing or tampering detected.")


        st.download_button(
            "💾 Download Heatmap Image",
            data=heatmap_img.tobytes(),
            file_name="forgery_heatmap.png",
            mime="image/png"
        )
elif page == "Duplicate Photo": 
    
    #st.title("📸 CAG PARAKH - Duplicate Photo Finding")


    st.write("""
    Detect duplicate passport-size photos across uploaded PDFs or images. Relative Dir Paths -  moredocs/Duplicate/same photo-1 or same photo-2 or same photo-3 or same photo-4
    """)

    dir_path = st.text_input("📂 Enter relative directory path containing PDFs or image files:")

    if dir_path and os.path.isdir(dir_path):

        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]

        if len(files) == 0:
            st.warning("No PDF or image files found in this folder.")
        else:
            st.success(f"Found {len(files)} documents.")
            start_btn = st.button("🚀 Start Duplicate Photo Detection")

            if start_btn:
                st.info("Detecting ... please wait.")
                face_records = []
                for file_path in tqdm(files):
                    if file_path.lower().endswith('.pdf'):
                        pages = convert_pdf_to_images(file_path)
                    else:
                        pages = [Image.open(file_path)]

                    for p in pages:
                        results = extract_faces_and_embeddings(p, file_path)
                        face_records.extend(results)

                st.success(f"Extracted {len(face_records)} total faces.")

                # --------------------------
                # Compare embeddings for duplicates
                # --------------------------
                if len(face_records) > 1:
                    st.info("Comparing photos to find duplicates...")
                    embeddings = np.array([r["embedding"] for r in face_records]).astype('float32')
                    n = len(embeddings)
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings)

                    # Normalize vectors for cosine similarity
                    faiss.normalize_L2(embeddings)

                    D, I = index.search(embeddings, k=5)
                    threshold = 0.85  # Adjust for stricter or looser matching
                    pairs = []
                    seen = set()

                    for i in range(n):
                        for j in range(1, 5):
                            if I[i, j] < 0:
                                continue
                            sim = D[i, j]
                            if sim >= threshold and (i, I[i, j]) not in seen and (I[i, j], i) not in seen:
                                seen.add((i, I[i, j]))
                                pairs.append((i, I[i, j], sim))

                    if len(pairs) == 0:
                        st.info("✅ No duplicate photos found.")
                    else:
                        st.success(f"Found {len(pairs)} duplicate photo pairs:")
                        for idx, (i1, i2, sim) in enumerate(pairs, 1):
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.image(face_records[i1]["face_img"], caption=f"{face_records[i1]['file']}")
                            with col2:
                                st.image(face_records[i2]["face_img"], caption=f"{face_records[i2]['file']}")
                            with col3:
                                st.write(f"**Similarity:** {sim:.3f}")
                                if face_records[i1]['file'] != face_records[i2]['file']:
                                    st.error("⚠️ Same Photo found in Different Documents also under shared folder.")
                                else:
                                    st.success("Same document face repetition.")
                else:
                    st.warning("Not enough faces detected to compare.")
    else:
        st.warning("Please enter a valid folder path to start.")    
# ----------------------------
# PAGE 5 - BLUR DETECTION
# ----------------------------
elif page == "Blur Detection":
    
    #st.title("🔍 CAG PARAKH - Blur & Image Quality Detection")
    st.write("""
    Upload an **image** or a **PDF document** to analyze its clarity.
    This module detects if the content is too blurred or low-quality for auditing.
    """)

    blur_threshold = st.sidebar.slider("Blur Sensitivity Threshold", 30, 300, 100, 10)
    window_size = st.sidebar.slider("Blur Region Window (px)", 10, 50, 15, 5)

    uploaded_file = st.file_uploader("📁 Upload a document or image", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_type = uploaded_file.name.lower().split(".")[-1]

        if file_type == "pdf":
            pdf_bytes = uploaded_file.read()
            images = extract_images_from_pdf(pdf_bytes)
            if not images:
                st.warning("No images found in PDF.")
            else:
                st.success(f"Extracted {len(images)} page image(s).")
                for i, pil_img in enumerate(images):
                    st.subheader(f"📄 Page {i+1}")
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # Detect blurred regions
                    blur_marked, score = highlight_blur_regions(cv_img, window_size, blur_threshold)
                    variance, is_blurred = detect_blur(cv_img, blur_threshold)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(pil_img, caption=f"Original (Sharpness: {round(variance, 2)})", use_container_width=True)
                    with col2:
                        st.image(cv2.cvtColor(blur_marked, cv2.COLOR_BGR2RGB),
                                 caption="Blurred Regions Highlighted", use_container_width=True)

                    if is_blurred:
                        st.error(f"🚫 Page appears **blurred** (sharpness={round(variance,2)}) — Please upload a clearer version.")
                    else:
                        st.success(f"✅ Page is **clear** and suitable for analysis. Sharpness={round(variance,2)}")
                    st.markdown("---")

        else:
            pil_img = Image.open(uploaded_file).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Detect blurred regions
            blur_marked, score = highlight_blur_regions(cv_img, window_size, blur_threshold)
            variance, is_blurred = detect_blur(cv_img, blur_threshold)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption=f"Original (Sharpness: {round(variance,2)})", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(blur_marked, cv2.COLOR_BGR2RGB),
                         caption="Blurred Regions Highlighted", use_container_width=True)

            if is_blurred:
                st.error(f"🚫 This image appears **blurred or low quality**. Sharpness={round(variance,2)} — Please re-upload a clearer version.")
            else:
                st.success(f"✅ Image is **clear** and readable. Sharpness={round(variance,2)}")

    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# FOOTER
# ----------------------------
st.markdown(
    "<div style='text-align: center;'>Powered by Wadhwani Foundation and Office of Principal Accountant General (Audit-I)</div>",
    unsafe_allow_html=True
)
