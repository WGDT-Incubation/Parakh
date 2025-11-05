# ai_docint_forensics_app.py
import streamlit as st
import io, tempfile, os, math
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
from pdf2image import convert_from_path

# Azure SDK
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# ----------------- CONFIG -----------------

AZURE_FORM_RECOGNIZER_ENDPOINT = "https://<YOUR_RESOURCE_NAME>.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "<YOUR_KEY>"

client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_RECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_RECOGNIZER_KEY)
)

st.set_page_config(page_title="Azure DOCINT + Forensics: Forgery Detector", layout="wide")
st.title("Azure Document Intelligence + Forensics — Forgery Detection POC")
st.write("Upload a document (image or PDF). System uses Azure Document Intelligence (text/layout/styles) + ELA/image forensics to detect edits.")

# ---------- Utilities ----------
def convert_pdf_to_image(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp.flush()
        images = convert_from_path(tmp.name)
    return images[0]

def pil_to_bytes(img_pil, fmt="JPEG"):
    b = io.BytesIO()
    img_pil.save(b, format=fmt)
    b.seek(0)
    return b.getvalue()

def perform_ela(image: Image.Image, quality=90):
    image_rgb = image.convert("RGB")
    tempfile_name = "temp_ela.jpg"
    image_rgb.save(tempfile_name, 'JPEG', quality=quality)
    ela_img = Image.open(tempfile_name).convert("RGB")
    diff = ImageChops.difference(image_rgb, ela_img)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 0
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return diff

def generate_heatmap_overlay(original: Image.Image, ela_image: Image.Image, threshold=120, alpha=0.45):
    ela_gray = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(ela_gray, threshold, 255, cv2.THRESH_BINARY)
    # blur mask lightly to make heatmap nicer
    mask_blur = cv2.GaussianBlur(mask, (7,7), 0)
    heatmap = cv2.applyColorMap(mask_blur, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(original), 1 - alpha, heatmap, alpha, 0)
    suspicious_pixels = int(np.count_nonzero(mask_blur))
    return Image.fromarray(overlay), suspicious_pixels, mask_blur

def edge_noise_score(image: Image.Image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# ---------- Azure Document Intelligence extraction ----------
def analyze_with_azure(image: Image.Image):
    """
    Sends image bytes to Azure Document Analysis (prebuilt-document)
    Returns the raw analyze result and summary metrics we use for heuristics.
    """
    img_bytes = pil_to_bytes(image, fmt="JPEG")
    poller = client.begin_analyze_document("prebuilt-document", img_bytes)
    result = poller.result()
    # Extract style/font info & bounding boxes & confidences
    styles = []
    words = []
    paragraphs = []
    for page in result.pages:
        # page.spans etc - depending on SDK result structure
        pass

    # The SDK returns .styles and .paragraphs sometimes; try reading available fields
    try:
        styles = result.styles  # list of DocumentStyle
    except Exception:
        styles = []

    try:
        paragraphs = result.paragraphs  # list of DocumentParagraph
    except Exception:
        paragraphs = []

    # Collect words + bounding boxes + confidence if available
    entities = []
    # result contains "content" with spans & pages; as fallback, iterate through result.documents/ fields
    try:
        for page in result.pages:
            if page.words:
                for w in page.words:
                    entities.append({
                        "content": getattr(w, "content", ""),
                        "confidence": getattr(w, "confidence", None),
                        "polygon": getattr(w, "polygon", None),
                        "bounding_box": getattr(w, "bounding_box", None),
                    })
    except Exception:
        pass

    # Build style metrics
    distinct_styles = 0
    low_confidence_words = 0
    total_words = 0
    font_sizes = []
    # If styles present, capture counts
    try:
        for s in styles:
            if getattr(s, "is_handwritten", False):
                distinct_styles += 1
            else:
                distinct_styles += 1
    except Exception:
        distinct_styles = len(styles)

    for e in entities:
        total_words += 1
        conf = e.get("confidence")
        if conf is not None and conf < 0.65:
            low_confidence_words += 1

    azure_metrics = {
        "distinct_styles": distinct_styles,
        "low_confidence_words": low_confidence_words,
        "total_words": total_words,
        "entities_sample": entities[:10]
    }
    return result, azure_metrics

# ---------- Heuristic fusion ----------
def compute_confidence(ela_mean, noise_score, suspicious_pixels, azure_metrics):
    """
    Combine ELA, noise, mask pixels, and Azure signals into a confidence % (0-100)
    """
    # ELA influence (normalized)
    ela_score = min(100, ela_mean / 1.5)
    # noise: low noise often indicates synthetic edits on otherwise noisy camera images
    noise_factor = max(0, (300 - noise_score) / 3)  # higher when noise_score low
    # pixel region factor: suspicious area scaled
    region_factor = min(50, suspicious_pixels / 1000)

    # Azure signals: if many low-confidence words, or multiple styles, increase suspicion
    azure_style_factor = 0
    if azure_metrics:
        lc = azure_metrics.get("low_confidence_words", 0)
        tot = max(1, azure_metrics.get("total_words", 1))
        azure_style_factor = min(30, (lc / tot) * 100)  # % of low confidence words
        # distinct styles may indicate patched-in text
        azure_style_factor += min(20, azure_metrics.get("distinct_styles", 0) * 5)

    # Weighted sum
    confidence = (ela_score * 0.45) + (noise_factor * 0.2) + (region_factor * 0.15) + (azure_style_factor * 0.2)
    confidence = max(0, min(100, confidence))
    return confidence

# ---------- UI ----------
tab1, tab2 = st.tabs(["Single Document (Azure + Forensics)", "Compare Two Documents"])

with tab1:
    uploaded = st.file_uploader("Upload single document (image/pdf)", type=["jpg","jpeg","png","pdf"], key="single")
    if uploaded:
        if uploaded.type == "application/pdf":
            try:
                image = convert_pdf_to_image(uploaded)
            except Exception as e:
                st.error("PDF conversion failed — ensure poppler installed (mac: brew install poppler). Error: "+str(e))
                st.stop()
        else:
            image = Image.open(uploaded).convert("RGB")

        st.image(image, caption="Uploaded Document", use_container_width=True)

        with st.spinner("Running Azure Document Intelligence and image-forensics..."):
            # Azure analyze
            try:
                azure_result, azure_metrics = analyze_with_azure(image)
            except Exception as e:
                st.error("Azure analysis failed. Check endpoint/key. Error: " + str(e))
                azure_result, azure_metrics = None, None

            # Forensics
            ela_img = perform_ela(image, quality=90)
            ela_mean = float(np.mean(np.array(ela_img)))
            heatmap_img, suspicious_pixels, mask = generate_heatmap_overlay(image, ela_img, threshold=120, alpha=0.5)
            noise = edge_noise_score(image)

            # Confidence combining Azure + Forensics
            confidence = compute_confidence(ela_mean, noise, suspicious_pixels, azure_metrics)
            verdict = "FORGED / EDITED" if confidence > 45 else "GENUINE / UNTAMPERED"

        # Show results
        c1, c2 = st.columns([1,1])
        with c1:
            st.subheader("Forensics Visuals")
            st.image(ela_img, caption="ELA Map (brighter=possible edits)", use_container_width=True)
            st.image(heatmap_img, caption="Heatmap Overlay (suspicious areas)", use_container_width=True)
        with c2:
            st.subheader("Metrics & Azure Signals")
            st.write(f"ELA mean intensity: **{ela_mean:.2f}**")
            st.write(f"Edge/Noise variance: **{noise:.2f}**")
            st.write(f"Suspicious pixels (mask): **{suspicious_pixels}**")
            if azure_metrics:
                st.write("Azure document metrics:")
                st.write(f"- distinct style count (approx): {azure_metrics.get('distinct_styles')}")
                st.write(f"- low-confidence words: {azure_metrics.get('low_confidence_words')} / {azure_metrics.get('total_words')}")
                st.write("Sample words (up to 10):")
                st.json(azure_metrics.get("entities_sample", []))
            st.subheader("Confidence & Verdict")
            st.progress(int(confidence)/100.0)
            if verdict.startswith("FORGED"):
                st.error(f"🚨 Verdict: {verdict} — Confidence: {confidence:.1f}%")
            else:
                st.success(f"✅ Verdict: {verdict} — Confidence: {confidence:.1f}%")
            st.caption("Note: this is probabilistic. Use as first-level triage for audits. For legal-proof, combine with signature/source verification.")
        # allow download of heatmap for audit report
        buf = io.BytesIO()
        heatmap_img.save(buf, format="PNG")
        st.download_button("Download Suspicion Heatmap (PNG)", buf.getvalue(), file_name="heatmap.png", mime="image/png")

with tab2:
    st.write("Compare Original vs Suspect — highlight differences (pixel-diff heatmap + similarity score).")
    col1, col2 = st.columns(2)
    with col1:
        orig = st.file_uploader("Upload ORIGINAL document (image/pdf)", type=["jpg","jpeg","png","pdf"], key="orig")
    with col2:
        suspect = st.file_uploader("Upload SUSPECT document (image/pdf)", type=["jpg","jpeg","png","pdf"], key="suspect")

    if orig and suspect:
        # preprocess both
        if orig.type == "application/pdf":
            try:
                img_orig = convert_pdf_to_image(orig)
            except Exception as e:
                st.error("PDF conversion failed for Original. Install poppler. Error: "+str(e)); st.stop()
        else:
            img_orig = Image.open(orig).convert("RGB")
        if suspect.type == "application/pdf":
            try:
                img_sus = convert_pdf_to_image(suspect)
            except Exception as e:
                st.error("PDF conversion failed for Suspect. Install poppler. Error: "+str(e)); st.stop()
        else:
            img_sus = Image.open(suspect).convert("RGB")

        # resize suspect to original size for fair pixel diff
        img_sus = img_sus.resize(img_orig.size)

        # compute absolute diff
        a = np.array(img_orig.convert("L"))
        b = np.array(img_sus.convert("L"))
        diff = cv2.absdiff(a, b)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(img_sus), 0.7, heatmap, 0.6, 0)
        overlay_img = Image.fromarray(overlay)
        similarity = 100 - (np.mean(diff)/255.0*100.0)
        similarity = max(0, min(100, similarity))

        c3, c4, c5 = st.columns(3)
        with c3:
            st.image(img_orig, caption="Original", use_container_width=True)
        with c4:
            st.image(img_sus, caption="Suspect (aligned)", use_container_width=True)
        with c5:
            st.image(overlay_img, caption="Difference Heatmap", use_container_width=True)

        st.subheader("Similarity & Verdict")
        st.progress(int(similarity)/100.0)
        st.write(f"Similarity Score: **{similarity:.2f}%**")
        if similarity < 85:
            st.error("🚨 Documents differ significantly — suspect edited/forged.")
        else:
            st.success("✅ Documents similar / likely genuine.")

        # optional download diff
        buf2 = io.BytesIO()
        overlay_img.save(buf2, format="PNG")
        st.download_button("Download Compare Heatmap", buf2.getvalue(), file_name="compare_heatmap.png", mime="image/png")
