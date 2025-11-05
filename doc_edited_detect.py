import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
from pdf2image import convert_from_path
import tempfile

st.set_page_config(page_title="AI Document Forgery Detection Suite", layout="wide")

st.title("🕵️‍♂️ AI Document Forgery Detection Suite")
st.caption("Detect edited or tampered documents using AI, visual forensics, and image comparison.")

tab1, tab2 = st.tabs(["📄 Single Document Analysis", "⚖️ Compare Two Documents"])

# ---------- Common Utilities ----------

def convert_pdf_to_image(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp.flush()
        images = convert_from_path(tmp.name)
    return images[0]

def perform_ela(image, quality=90):
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    ela_image = Image.open(temp_filename)
    diff = ImageChops.difference(image, ela_image)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return diff

def generate_heatmap_overlay(original, ela_image, threshold=150, alpha=0.45):
    ela_gray = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(ela_gray, threshold, 255, cv2.THRESH_BINARY)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(original), 1 - alpha, heatmap, alpha, 0)
    suspicious_regions = np.count_nonzero(mask > 0)
    return Image.fromarray(overlay), suspicious_regions

def edge_noise_analysis(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def preprocess_image(uploaded_file):
    if uploaded_file.type == "application/pdf":
        image = convert_pdf_to_image(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert("RGB")
    return image

def resize_to_match(img1, img2):
    return img2.resize(img1.size)

def compute_difference(original, suspect):
    img1 = np.array(original)
    img2 = np.array(suspect)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img2, 0.7, heatmap, 0.6, 0)
    similarity = 100 - (np.mean(diff) / 255 * 100)
    return Image.fromarray(overlay), diff, np.clip(similarity, 0, 100)

# ---------- TAB 1: Single Document ----------

with tab1:
    st.header("📄 Single Document Forgery Detection")
    st.write("Upload a single document (image or PDF) to analyze if it has been tampered or edited.")

    uploaded_file = st.file_uploader("Upload document", type=["jpg", "jpeg", "png", "pdf"], key="single_upload")

    if uploaded_file:
        # Load document
        image = preprocess_image(uploaded_file)
        st.image(image, caption="Uploaded Document", use_container_width=True)

        # Step 1: ELA
        st.subheader("📸 Step 1: Error Level Analysis (ELA)")
        ela_img = perform_ela(image)
        ela_array = np.array(ela_img)
        ela_mean = np.mean(ela_array)
        st.write(f"ELA Mean Intensity: **{ela_mean:.2f}**")

        # Step 2: Edge Noise Analysis
        st.subheader("🧩 Step 2: Edge & Noise Consistency")
        noise_score = edge_noise_analysis(image)
        st.write(f"Noise Variance: **{noise_score:.2f}**")

        # Step 3: Heatmap Overlay
        st.subheader("🔥 Step 3: Suspicious Region Heatmap")
        heatmap_img, suspicious_regions = generate_heatmap_overlay(image, ela_img, threshold=150, alpha=0.5)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Document", use_container_width=True)
        with col2:
            st.image(heatmap_img, caption="Heatmap Overlay (Suspicious Regions)", use_container_width=True)

        # Step 4: Confidence Score
        st.subheader("⚖️ Step 4: Confidence Score & Verdict")
        # More robust detection for text edits and low-noise images

        # Normalize metrics
        ela_score = min(100, (ela_mean / 1.5))  # increase ELA influence
        noise_factor = 100 - min(100, (noise_score / 6))  # lower noise = more suspicion
        region_factor = min(60, suspicious_regions / 200)  # higher sensitivity to regions

        # Combined confidence with weights
        confidence_score = np.clip((ela_score * 0.4) + (noise_factor * 0.3) + (region_factor * 0.3), 0, 100)

        # Verdict threshold slightly lowered
        verdict = "FORGED / EDITED" if confidence_score > 45 else "GENUINE / UNTAMPERED"


        st.progress(confidence_score / 100)
        if verdict == "FORGED / EDITED":
            st.error(f"🚨 Document appears **{verdict}** — Confidence: {confidence_score:.1f}%")
        else:
            st.success(f"✅ Document appears **{verdict}** — Confidence: {confidence_score:.1f}%")

        st.caption(f"{suspicious_regions} suspicious regions detected with inconsistent compression or noise patterns.")

# ---------- TAB 2: Compare Documents ----------

with tab2:
    st.header("⚖️ Compare Two Documents")
    st.write("Upload both the **Original** and the **Suspect** version for side-by-side comparison and heatmap difference detection.")

    colA, colB = st.columns(2)
    with colA:
        original_file = st.file_uploader("📘 Upload ORIGINAL Document", type=["jpg", "jpeg", "png", "pdf"], key="orig")
    with colB:
        suspect_file = st.file_uploader("📕 Upload SUSPECT Document", type=["jpg", "jpeg", "png", "pdf"], key="suspect")

    if original_file and suspect_file:
        original_img = preprocess_image(original_file)
        suspect_img = preprocess_image(suspect_file)
        suspect_resized = resize_to_match(original_img, suspect_img)

        st.subheader("🔥 Step 1: Visual Comparison")
        overlay_img, diff_map, similarity_score = compute_difference(original_img, suspect_resized)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.image(original_img, caption="Original Document", use_container_width=True)
        with col4:
            st.image(suspect_resized, caption="Suspect Document", use_container_width=True)
        with col5:
            st.image(overlay_img, caption="Detected Differences (Heatmap)", use_container_width=True)

        st.subheader("⚖️ Step 2: Similarity Score & Verdict")
        st.progress(similarity_score / 100)
        st.write(f"**Similarity Score:** {similarity_score:.2f}%")

        if similarity_score < 85:
            st.error("🚨 The documents appear **DIFFERENT / EDITED**.")
            st.caption("Significant visual differences detected.")
        else:
            st.success("✅ The documents appear **IDENTICAL / GENUINE**.")
            st.caption("No major differences found.")

        st.download_button(
            label="💾 Download Heatmap Image",
            data=overlay_img.tobytes(),
            file_name="document_difference_heatmap.png",
            mime="image/png"
        )
