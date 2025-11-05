
"""
document_forgery_pipeline.py

Usage:
    python document_forgery_pipeline.py --pdf_dir ./pdfs --out_dir ./forensic_outputs

Outputs:
    - extracted_images/        (images extracted from PDFs)
    - forensic_outputs/        (ELA images, marker-detection annotated images, variance maps, copy-move visualizations)
    - report.json              (summary)
"""

import argparse
from pathlib import Path
import json
import fitz  # PyMuPDF
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import os
import math
import pytesseract  # optional; comment out if not installed
from skimage.feature import match_descriptors, ORB
from skimage.color import rgb2gray
from skimage import img_as_ubyte

# ----------------------------
# CONFIG - tune thresholds here
# ----------------------------
CONFIG = {
    "jpeg_recompress_quality": 90,
    "ela_scale": 25,
    "hsv_sat_thresh": 30,      # saturation threshold for white/erase detection
    "hsv_val_thresh": 200,     # value (brightness) threshold for white/erase detection
    "white_min_area": 150,     # minimum contour area to consider (in pixels)
    "variance_kernel": 25,     # kernel for local variance map
    "orb_n_keypoints": 2000,   # ORB keypoints for copy-move
    "orb_min_match_dist": 20,  # minimum distance to consider match non-trivial (px)
}

# ----------------------------
# Utility functions
# ----------------------------
def extract_images_from_pdfs(pdf_paths, out_dir: Path):
    """
    Extract images embedded in each PDF and save them as files.
    Returns dict: {pdf_filename: [image_filenames...], ...}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for pdf_path in pdf_paths:
        p = Path(pdf_path)
        if not p.exists():
            continue
        doc = fitz.open(str(p))
        images = []
        for page_index, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                img_name = f"{p.stem}_page{page_index+1}_img{img_index+1}.{ext}"
                img_path = out_dir / img_name
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                images.append(str(img_path))
        report[p.name] = images
    return report

def ela_image(pil_img: Image.Image, out_path: Path, scale=25, quality=90):
    """
    Perform Error Level Analysis:
    - Recompress image at given JPEG quality, compute difference, amplify and save.
    """
    temp = out_path.parent / (out_path.stem + "_tmp_recomp.jpg")
    pil_img.save(temp, "JPEG", quality=quality)
    recompressed = Image.open(temp).convert("RGB")
    original = pil_img.convert("RGB")
    diff = ImageChops.difference(original, recompressed)
    # amplify
    enhancer = ImageEnhance.Brightness(diff)
    ela = enhancer.enhance(scale)
    ela.save(out_path)
    try:
        temp.unlink()
    except Exception:
        pass
    return str(out_path)

def detect_white_marker_bboxes(cv_img: np.ndarray, sat_thresh=30, val_thresh=200, min_area=150):
    """
    Detect low-saturation & high-brightness patches (likely white-marker / correction fluid)
    Returns: annotated image (BGR), list of bboxes [(x,y,w,h,area),...], mask (uint8)
    """
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    sat_mask = (s < sat_thresh).astype('uint8') * 255
    val_mask = (v > val_thresh).astype('uint8') * 255
    mask = cv2.bitwise_and(sat_mask, val_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv_img.copy()
    bboxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x,y,w,h = cv2.boundingRect(c)
            bboxes.append((x,y,w,h,area))
            cv2.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 2)
    return out, bboxes, mask

def local_variance_map(gray: np.ndarray, k=25):
    """
    Compute local variance map using moving window.
    Returns normalized uint8 variance image for visualization.
    """
    kernel = (k, k)
    mean = cv2.boxFilter(gray.astype('float32'), -1, kernel, normalize=True)
    mean_sq = cv2.boxFilter((gray.astype('float32')**2), -1, kernel, normalize=True)
    var = mean_sq - (mean**2)
    # clip negative small numerical errors
    var[var < 0] = 0
    vis = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return vis

def orb_copy_move_detection(gray: np.ndarray, n_keypoints=2000, min_match_dist=20):
    """
    Use ORB descriptors to find potential duplicated patches (naive copy-move).
    Returns list of matches where points are far enough (x1,y1,x2,y2)
    and a visualization image (RGB) with lines drawn.
    """
    orb = cv2.ORB_create(n_keypoints)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 2:
        return [], None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des, des, k=2)
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        # ratio test
        if m.distance < 0.75 * n.distance:
            p1 = kp[m.queryIdx].pt
            p2 = kp[m.trainIdx].pt
            dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            # ignore trivial self-matches or very nearby matches
            if dist > min_match_dist:
                good.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    # build visualization
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x1,y1,x2,y2) in good:
        cv2.line(vis, (x1,y1), (x2,y2), (255,0,0), 1)
        cv2.circle(vis, (x1,y1), 2, (0,255,0), -1)
        cv2.circle(vis, (x2,y2), 2, (0,255,255), -1)
    return good, vis

def ocr_image_and_pdf_text_compare(img_path: Path, pdf_path: Path = None):
    """
    Run pytesseract OCR on image and compare with PDF text (if pdf_path given).
    Returns dict with ocr_text and pdf_text and a simple similarity metric.
    """
    ocr_text = pytesseract.image_to_string(str(img_path))
    pdf_text = ""
    similarity = None
    if pdf_path and Path(pdf_path).exists():
        try:
            doc = fitz.open(str(pdf_path))
            txts = []
            for page in doc:
                txts.append(page.get_text("text"))
            pdf_text = "\n".join(txts)
            # simple comparison: percent of OCR words present in PDF text
            ocr_words = [w.strip().lower() for w in ocr_text.split()]
            if len(ocr_words) > 0:
                present = sum(1 for w in ocr_words if w in pdf_text.lower())
                similarity = present / len(ocr_words)
        except Exception:
            pass
    return {"ocr_text": ocr_text, "pdf_text": pdf_text, "ocr_pdf_word_overlap_ratio": similarity}

# ----------------------------
# Main pipeline
# ----------------------------
def analyze_pdfs(pdf_dir: Path, out_dir: Path):
    pdf_list = sorted([str(p) for p in Path(pdf_dir).glob("*.pdf")])
    extracted_dir = out_dir / "extracted_images"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    forensic_dir = out_dir / "forensic_outputs"
    forensic_dir.mkdir(parents=True, exist_ok=True)

    extraction_report = extract_images_from_pdfs(pdf_list, extracted_dir)
    summary = {}

    for pdf_name, images in extraction_report.items():
        for img_path in images:
            img_p = Path(img_path)
            base = img_p.stem
            # read with PIL and cv2
            pil = Image.open(img_p).convert("RGB")
            cv_img = cv2.imread(str(img_p))
            if cv_img is None:
                continue
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 1) ELA
            ela_out = forensic_dir / f"{base}_ELA.png"
            ela_image(pil, ela_out, scale=CONFIG["ela_scale"], quality=CONFIG["jpeg_recompress_quality"])

            # 2) white marker detection
            marker_vis, bboxes, mask = detect_white_marker_bboxes(cv_img,
                                                                  sat_thresh=CONFIG["hsv_sat_thresh"],
                                                                  val_thresh=CONFIG["hsv_val_thresh"],
                                                                  min_area=CONFIG["white_min_area"])
            marker_out = forensic_dir / f"{base}_MARKER.png"
            cv2.imwrite(str(marker_out), marker_vis)

            # 3) variance map
            varmap = local_variance_map(gray, k=CONFIG["variance_kernel"])
            varmap_out = forensic_dir / f"{base}_VAR.png"
            cv2.imwrite(str(varmap_out), varmap)

            # 4) ORB copy-move
            matches, cm_vis = orb_copy_move_detection(gray, n_keypoints=CONFIG["orb_n_keypoints"],
                                                      min_match_dist=CONFIG["orb_min_match_dist"])
            cm_out = forensic_dir / f"{base}_COPYMOVE.png"
            if cm_vis is not None:
                cv2.imwrite(str(cm_out), cm_vis)

            # 5) OCR compare (optional)
            ocr_res = None
            try:
                ocr_res = ocr_image_and_pdf_text_compare(img_p, None)
            except Exception:
                ocr_res = {"ocr_text": None, "pdf_text": None, "ocr_pdf_word_overlap_ratio": None}

            summary[str(img_p)] = {
                "pdf_source": pdf_name,
                "ela_image": str(ela_out),
                "marker_detection_image": str(marker_out),
                "marker_bboxes_count": len(bboxes),
                "marker_bboxes": bboxes,
                "variance_map_image": str(varmap_out),
                "copy_move_image": str(cm_out) if cm_vis is not None else None,
                "copy_move_matches": len(matches),
                "ocr_result": ocr_res,
            }

    # Save full summary
    summary_path = out_dir / "report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Analysis complete.")
    print("Extracted images folder:", extracted_dir)
    print("Forensic outputs folder:", forensic_dir)
    print("Summary report:", summary_path)
    return summary_path

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document forgery pipeline for PDFs")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDFs")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for forensic results")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_pdfs(pdf_dir, out_dir)
