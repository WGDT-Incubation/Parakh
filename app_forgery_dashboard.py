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
import fitz

# ------------------------
# Executor
# ------------------------

def run_validator(input_dir, validator_script="azure_doc_validator.py"):
    """Run azure_doc_validator.py exactly as from terminal."""
    output_path = Path(input_dir) / "document_validation_results.json"
    cmd = [
        "python",
        validator_script,
        "--input_dir", str(input_dir),
        "--output", str(output_path)
    ]

    #st.write("🟢 Running:", " ".join(cmd))
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
    """Render all pages of PDF as PIL images."""
    doc = fitz.open(pdf_path)
    imgs = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(BytesIO(pix.tobytes("png")))
        imgs.append(img)
    return imgs


def style_status(val):
    if val == "PASS":
        return "✅ PASS"
    elif val == "FAIL":
        return "❌ FAIL"
    elif val == "REVIEW":
        return "⚠️ REVIEW"
    else:
        return val

def extract_images_from_pdf(pdf_bytes):
    """Extract images from PDF and return list of PIL images."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                base_image = doc.extract_image(img[0])
                image_bytes = base_image["image"]
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


def load_json_results(path):
    with open(path, "r") as f:
        return json.load(f)


def style_status(val):
    """Color and emoji formatting for PASS/FAIL."""
    if val == "PASS":
        return "✅ PASS"
    elif val == "FAIL":
        return "❌ FAIL"
    else:
        return val


# ----------------------------
#  UDemo
# ----------------------------

st.set_page_config(page_title="AI Document Intelligence Dashboard", layout="wide")

st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Select a view", ["Forgery Detection", "Document Validation"])

st.sidebar.markdown("---")
st.sidebar.caption("AI Document Integrity Demo")
logo_image_path = "image/caglogo.png"
st.sidebar.image(logo_image_path, use_container_width=True) 

# ----------------------------
# PAGE 1: CAG Parakh Forgery Detection
# ----------------------------
if page == "Forgery Detection":
    #logo_image_path = "image/caglogo.png"
    #st.sidebar.image(logo_image_path, use_container_width=True) 
    st.image = "image/caglogo.png"
    st.title("🕵️ CAG Parakh - Document Forgery Detection")
    st.write("""
    ⚠️ Upload a **PDF or image** to detect suspicious white-marker or digital erasure areas.  
    Red boxes highlight potential tampered zones.
    """)

    sat = st.sidebar.slider("Saturation Threshold", 10, 100, 30, 5)
    val = st.sidebar.slider("Brightness Threshold", 150, 255, 200, 5)
    min_area = st.sidebar.slider("Minimum Area (px²)", 50, 1000, 150, 50)

    uploaded_file = st.file_uploader("Upload a document (PDF or image)", type=["pdf", "jpg", "jpeg", "png"])

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

# ----------------------------
# PAGE 2: CAG Parakh Document Validation Results
# ----------------------------
elif page == "Document Validation":
    st.set_page_config(page_title="CAG Parakh Document Validator", layout="wide")
    st.title("🔎 CAG Parakh - Document Validator")

    st.write("""
    ⚠️ Upload the folder path containing PDFs and click **Run Validator**  
    """)

    # Initialize session state
    if "validation_done" not in st.session_state:
        st.session_state.validation_done = False
    if "result_json" not in st.session_state:
        st.session_state.result_json = None
    if "input_dir" not in st.session_state:
        st.session_state.input_dir = None
    if "data" not in st.session_state:
        st.session_state.data = None

    # --- Inputs (only directory now) ---
    input_dir = st.text_input("📂 Enter directory path containing PDFs:", value="input_docs")
    run_button = st.button("▶️ Run Validator")

    # --- Run validator only on button click ---
    if run_button:
        if not Path(input_dir).exists():
            st.error(f"Input directory not found: {input_dir}")
        else:
            with st.spinner("🟢 Running CAG Parakh validator, please wait ..."):
                result_json = run_validator(input_dir, validator_script="azure_doc_validator.py")
            if result_json and Path(result_json).exists():
                data = load_json_results(result_json)
                if data:
                    st.session_state.validation_done = True
                    st.session_state.result_json = result_json
                    st.session_state.input_dir = input_dir
                    st.session_state.data = data
            else:
                st.error("Validator failed or no JSON generated.")

    # --- Display results if available ---
    if st.session_state.validation_done and st.session_state.data:
        data = st.session_state.data
        input_dir = st.session_state.input_dir
        results = data.get("results", [])

        df = pd.DataFrame(results)
        if not df.empty:
            df["File"] = df["file"].apply(lambda x: Path(x).name)
            df["Score"] = df["score"].apply(lambda x: f"{round(x)} %")
            df["Status"] = df["status"].apply(style_status)
            df.rename(columns={"detected_name": "Detected Name"}, inplace=True)
            df = df[["File", "Detected Name", "Score", "Status"]]

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Simple summary (no big PASS/FAIL headline)
            pass_count = (df["Status"].str.contains("PASS")).sum()
            fail_count = (df["Status"].str.contains("FAIL")).sum()
            total = len(df)
            st.write(f"**Summary:** {pass_count} PASS  |  {fail_count} FAIL  (out of {total})")

            st.markdown("---")
            st.subheader("📘 Document Preview")

            # Dropdown for PDF preview
            selected_file = st.selectbox(
                "Select a document to preview:",
                options=df["File"].tolist(),
                key="pdf_selector"
            )

            if selected_file:
                pdf_path = Path(input_dir) / selected_file
                if pdf_path.exists():
                    pages = render_pdf_pages(pdf_path)
                    for i, p in enumerate(pages):
                        st.image(p, caption=f"Page {i+1}", use_container_width=True)
                else:
                    st.warning(f"{selected_file} not found in directory.")
    else:
        st.info("Enter your folder path and click 'Run Validator' to generate results.")

    
