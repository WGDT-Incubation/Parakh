#!/usr/bin/env python3
"""
forgery_pipeline_with_cnn.py

Usage:
  # Analyze PDFs, extract images, run forensic checks, extract suspicious patches and score them using CNN model
  python forgery_pipeline_with_cnn.py --pdf_dir ./pdfs --out_dir ./forensic_results --mode analyze

  # Train CNN on labeled patches in a folder structure:
  # dataset/
  #   train/
  #     original/
  #     edited/
  #   val/
  #     original/
  #     edited/
  python forgery_pipeline_with_cnn.py --mode train --data_dir ./dataset --model_out model.h5

Notes:
- Dependencies:
    pip install pymupdf pillow opencv-python numpy tensorflow scikit-learn imageio tqdm
- Optional: pytesseract for OCR if you enable OCR.
"""

import argparse
from pathlib import Path
import json, os, math
import fitz
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------
# CONFIG (tune as needed)
# -----------------------
CONFIG = {
    "jpeg_recompress_quality": 90,
    "ela_scale": 25,
    "hsv_sat_thresh": 30,
    "hsv_val_thresh": 200,
    "white_min_area": 150,
    "variance_kernel": 25,
    "patch_size": (64, 64),     # patch size fed to CNN
    "cnn_epochs": 10,
    "cnn_batch_size": 32,
    "cnn_learning_rate": 1e-3,
    "min_patch_pixels": 50,     # ignore tiny patches
}

# -----------------------
# Utility & forensic funcs
# -----------------------
def extract_images_from_pdfs(pdf_paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for pdf_path in pdf_paths:
        p = Path(pdf_path)
        if not p.exists(): continue
        doc = fitz.open(str(p))
        imgs = []
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
                imgs.append(str(img_path))
        report[p.name] = imgs
    return report

def ela_image(pil_img: Image.Image, scale=25, quality=90):
    temp = pil_img.copy()
    tmp_path = "/tmp/_ela_tmp.jpg"
    pil_img.save(tmp_path, "JPEG", quality=quality)
    recompressed = Image.open(tmp_path).convert("RGB")
    diff = ImageChops.difference(pil_img.convert("RGB"), recompressed)
    ela = ImageEnhance.Brightness(diff).enhance(scale)
    return ela

def detect_white_marker_bboxes(cv_img, sat_thresh=30, val_thresh=200, min_area=150):
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_mask = (s < sat_thresh).astype('uint8') * 255
    val_mask = (v > val_thresh).astype('uint8') * 255
    mask = cv2.bitwise_and(sat_mask, val_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x,y,w,h = cv2.boundingRect(c)
            bboxes.append((x,y,w,h,area))
    return bboxes, mask

def local_variance_map(gray, k=25):
    mean = cv2.boxFilter(gray.astype('float32'), -1, (k,k))
    mean_sq = cv2.boxFilter((gray.astype('float32')**2), -1, (k,k))
    var = mean_sq - (mean**2)
    var[var < 0] = 0
    vis = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return vis

def orb_copy_move_detection(gray, n_keypoints=2000, min_match_dist=20):
    orb = cv2.ORB_create(n_keypoints)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des, des, k=2)
    good = []
    for m_n in matches:
        if len(m_n) < 2: continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            p1 = kp[m.queryIdx].pt
            p2 = kp[m.trainIdx].pt
            dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist > min_match_dist:
                good.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    return good

# -----------------------
# CNN model
# -----------------------
def build_small_cnn(input_shape=(64,64,3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG["cnn_learning_rate"]),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------
# Patch utilities
# -----------------------
def extract_patch(img_cv, bbox, pad=4, size=(64,64)):
    x,y,w,h,_ = bbox if len(bbox)==5 else (*bbox,0)
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(img_cv.shape[1], x+w+pad); y2 = min(img_cv.shape[0], y+h+pad)
    patch = img_cv[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, size)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return patch

# -----------------------
# Training and inference
# -----------------------
def load_dataset_from_dir(data_dir, img_size=(64,64)):
    """
    Expects:
      data_dir/train/original, data_dir/train/edited, same for val/
    Returns numpy arrays
    """
    X, y = [], []
    data_dir = Path(data_dir)
    for cls_name, label in [("original", 0), ("edited", 1)]:
        d = data_dir / "train" / cls_name
        if not d.exists(): continue
        for p in d.glob("*"):
            try:
                im = Image.open(p).convert("RGB").resize(img_size)
                X.append(np.array(im)/255.0)
                y.append(label)
            except Exception:
                continue
    X = np.array(X); y = np.array(y)
    return X, y

def train_cnn(data_dir, model_out, input_shape=(64,64,3)):
    X, y = load_dataset_from_dir(data_dir, img_size=input_shape[:2])
    if len(X)==0:
        raise RuntimeError("No training images found. Prepare dataset/train/{original,edited}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    model = build_small_cnn(input_shape=input_shape)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=CONFIG["cnn_epochs"], batch_size=CONFIG["cnn_batch_size"])
    model.save(model_out)
    print("Model saved to", model_out)
    return model

def score_patches_with_model(model, patches):
    """
    patches: list of numpy RGB arrays scaled 0..1
    returns probabilities list
    """
    if len(patches)==0:
        return []
    X = np.array(patches)
    preds = model.predict(X, verbose=0).flatten()
    return preds.tolist()

# -----------------------
# Main analysis flow
# -----------------------
def analyze_pdfs_and_score(pdf_dir, out_dir, model_path=None):
    pdf_list = sorted([str(p) for p in Path(pdf_dir).glob("*.pdf")])
    extracted_dir = Path(out_dir)/"extracted_images"; extracted_dir.mkdir(parents=True, exist_ok=True)
    forensic_dir = Path(out_dir)/"forensic_outputs"; forensic_dir.mkdir(parents=True, exist_ok=True)

    extraction_report = extract_images_from_pdfs(pdf_list, extracted_dir)
    summary = {}

    # load model if present
    model = None
    if model_path:
        model = tf.keras.models.load_model(model_path)

    for pdf_name, images in extraction_report.items():
        for img_path in images:
            img_p = Path(img_path)
            base = img_p.stem
            pil = Image.open(img_p).convert("RGB")
            cv_img = cv2.imread(img_p)
            if cv_img is None:
                continue
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 1 ELA
            ela = ela_image(pil, scale=CONFIG["ela_scale"], quality=CONFIG["jpeg_recompress_quality"])
            ela_out = forensic_dir / f"{base}_ELA.png"; ela.save(ela_out)

            # 2 white-marker candidate bboxes
            bboxes, mask = detect_white_marker_bboxes(cv_img,
                                                     sat_thresh=CONFIG["hsv_sat_thresh"],
                                                     val_thresh=CONFIG["hsv_val_thresh"],
                                                     min_area=CONFIG["white_min_area"])
            # annotate and save marker image
            vis = cv_img.copy()
            for (x,y,w,h,area) in bboxes:
                cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 2)
            marker_out = forensic_dir / f"{base}_MARKER.png"; cv2.imwrite(str(marker_out), vis)

            # 3 variance map
            varmap = local_variance_map(gray, k=CONFIG["variance_kernel"])
            var_out = forensic_dir / f"{base}_VAR.png"; cv2.imwrite(str(var_out), varmap)

            # 4 copy-move
            cm_matches = orb_copy_move_detection(gray)
            cm_out = forensic_dir / f"{base}_COPYMOVE.png"
            if cm_matches:
                cm_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for (x1,y1,x2,y2) in cm_matches:
                    cv2.line(cm_vis, (x1,y1),(x2,y2),(255,0,0),1)
                cv2.imwrite(str(cm_out), cm_vis)
            else:
                cm_out = None

            # 5 extract patches and score via CNN model if available
            patches = []
            patches_meta = []
            for bbox in bboxes:
                if bbox[4] < CONFIG["min_patch_pixels"]:
                    continue
                patch = extract_patch(cv_img, bbox, pad=4, size=CONFIG["patch_size"])
                if patch is None: continue
                patch = patch.astype('float32')/255.0
                patches.append(patch)
                patches_meta.append(bbox)

            probs = []
            if model is not None and len(patches)>0:
                probs = score_patches_with_model(model, patches)
            else:
                probs = [None]*len(patches)

            # record
            summary[str(img_p)] = {
                "pdf_source": pdf_name,
                "ela_image": str(ela_out),
                "marker_image": str(marker_out),
                "variance_image": str(var_out),
                "copy_move_image": str(cm_out) if cm_matches else None,
                "num_marker_bboxes": len(bboxes),
                "patches": [{"bbox": patches_meta[i], "cnn_prob_edited": probs[i]} for i in range(len(patches_meta))]
            }

    # save summary
    summary_path = Path(out_dir)/"report_with_cnn.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Done. Summary saved to", summary_path)
    return summary_path

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analyze","train"], required=True)
    parser.add_argument("--pdf_dir", type=str, default="./pdfs")
    parser.add_argument("--out_dir", type=str, default="./forensic_results")
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--model_out", type=str, default="forgery_cnn.h5")
    parser.add_argument("--model_in", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train_cnn(args.data_dir, args.model_out, input_shape=(CONFIG["patch_size"][0], CONFIG["patch_size"][1], 3))
    else:
        analyze_pdfs_and_score(args.pdf_dir, args.out_dir, model_path=args.model_in)
