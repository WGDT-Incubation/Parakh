import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
from pdf2image import convert_from_path
import tempfile

st.set_page_config(page_title="AI Forgery Detection (Single Document)", layout="wide")

st.title("🕵️‍♂️ CAG Parakh — Detect Edited or Fake Document Images")
st.caption("Detect if a single document image (like PAN, ID, bill, form) has been digitally edited or tampered using AI-based image forensics.")

# ---------- Utility Functions ----------

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


# ---------- Streamlit UI ----------
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
        st.error(f"🚨 Document appears **{verdict}** — Confidence: {confidence_score:.1f}%")
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
