import streamlit as st
import fitz
import cv2
import numpy as np
from PIL import Image
import json
from io import BytesIO
import pandas as pd
import base64
import subprocess
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
# CUSTOM CSS
# ----------------------------
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9fafc;
        }
        .navbar {
            text-align: right;
            padding: 10px 60px 0px 0px;
            font-size: 16px;
        }
        .navbar a {
            text-decoration: none;
            color: #0c2340;
            margin-left: 20px;
            font-weight: 500;
        }
        .navbar a:hover {
            color: #4b61d1;
        }
        .login-btn {
            background-color: #0c2340;
            color: white !important;
            padding: 6px 16px;
            border-radius: 8px;
        }
        .upload-card {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        footer {
            text-align: center;
            font-size: 14px;
            color: #555;
            margin-top: 80px;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# NAVBAR
# ----------------------------
st.markdown("""
<div class="navbar">
    <a href="#">HOME</a>
    <a href="#">CONTACT / SUPPORT</a>
    <a href="#" class="login-btn">SECURE LOGIN</a>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER SECTION (LOGO + TITLE)
# ----------------------------
 
col1, col2, col3,col4,col5 = st.columns([1,1,2,1,1])
with col3:
    st.image("image/caglogo.png", width=100)


col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("image/parakh.png", width=400)



# ----------------------------
# INTRODUCTORY / FEATURE SECTION (TEXT IN SAME ALIGNMENT AS IMAGE)
# ----------------------------
st.markdown("""
<div style="
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    padding: 25px 40px;
    margin-top: 10px;
    margin-bottom: 25px;
    width: 75%;
    margin-left: auto;
    margin-right: auto;
    text-align: left;
    ">
    <h3 style="color:#0c2340;">👋 Hello, Auditor!</h3>
    <p style="font-size:16px; color:#333;">
    Upload a document (<b>PDF, DOCX, or Image</b>) and I’ll analyze its:
    </p>
            <p>
            <ul style="font-size:16px; color:#222; line-height:1.7;">
        <li><b>Structural Integrity</b> – layout consistency, missing pages, duplications</li>
        <li><b>Semantic Coherence</b> – logical flow, contradictory content, incomplete sections</li>
        <li><b>Authenticity Indicators</b> – repetitive signatures, overwritten text, metadata inconsistencies</li>
    </ul>
            </p>
    
</div>
""", unsafe_allow_html=True)


# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.title("📂 Configuration")
page = st.sidebar.radio(
    "Select a view",
    ["Forgery Detection", "Document Validation", "Doc Authenticity", "Duplicate Photo Finding","Blur Detection"]
)


st.sidebar.markdown("---")
st.sidebar.caption("AI Document Integrity Demo")

# ----------------------------
# COMMON HELPER FUNCTIONS
# ----------------------------
def run_validator(input_dir, validator_script="azure_doc_validator.py"):
    output_path = Path(input_dir) / "document_validation_results.json"
    cmd = [
        "python",
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


def style_status(val):
    if val == "PASS":
        return "✅ PASS"
    elif val == "FAIL":
        return "❌ FAIL"
    elif val == "REVIEW":
        return "⚠️ REVIEW"
    else:
        return val


# ----------------------------
# PAGE 1 - FORGERY DETECTION
# ----------------------------
if page == "Forgery Detection":
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
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
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.title("🔎 CAG PARAKH - Document Validation")
    st.write("""
    Upload a folder containing PDFs and click **Run Validator** to check:
    - Structural Integrity (layout, pages, duplications)
    - Semantic Coherence (logical flow, contradictions)
    - Authenticity Indicators (overwritten text, metadata issues)
    """)

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
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.title("CAG PARAKH - Find Document Authenticity")
    st.write("""
    Upload a folder containing PDFs and click **Run Analysis** to detect if document is not edited using any tool.
    """)
elif page == "Duplicate Photo Finding": 
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.title("📸 CAG PARAKH - Duplicate Photo Finding")
    st.write("""
    Upload a folder containing PDFs and click **Run Analysis** to detect if the same face photo is used
    across multiple documents.
    """)    
# ----------------------------
# PAGE 5 - BLUR DETECTION
# ----------------------------
elif page == "Blur Detection":
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.title("🔍 CAG PARAKH - Blur & Image Quality Detection")
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
st.markdown("""
<footer>
    <strong>Powered by Wadhwani Foundation and Office of Principal Accountant General (Audit-I)</strong>
</footer>
""", unsafe_allow_html=True)
