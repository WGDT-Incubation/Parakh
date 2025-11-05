#!/usr/bin/env python3
"""
generate_synthetic_forgery_dataset.py

Generate synthetic 'edited' (forged) and 'original' patches
from a few sample document images to train a forgery detection CNN.

Usage:
  python generate_synthetic_forgery_dataset.py --src_dir ./sample_docs --out_dir ./dataset --num_patches 200

Dependencies:
  pip install opencv-python pillow numpy tqdm
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random, os

def random_patch(img, size=(64,64)):
    h, w, _ = img.shape
    if h < size[1] or w < size[0]:
        return cv2.resize(img, size)
    x = np.random.randint(0, w - size[0])
    y = np.random.randint(0, h - size[1])
    patch = img[y:y+size[1], x:x+size[0]]
    return patch

def apply_white_marker_effect(patch):
    patch = patch.copy()
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    # simulate 1–3 random white brush strokes
    for _ in range(np.random.randint(1,4)):
        x1, y1 = np.random.randint(0, patch.shape[1]), np.random.randint(0, patch.shape[0])
        x2, y2 = np.random.randint(0, patch.shape[1]), np.random.randint(0, patch.shape[0])
        thickness = np.random.randint(10,25)
        cv2.line(mask, (x1,y1), (x2,y2), 255, thickness)
    patch[mask > 0] = [255,255,255]
    # optional blur to look more real
    if random.random() > 0.5:
        patch = cv2.GaussianBlur(patch, (5,5), 1)
    return patch

def apply_digital_inpaint(patch):
    patch = patch.copy()
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    for _ in range(np.random.randint(1,3)):
        x1,y1 = np.random.randint(0, patch.shape[1]), np.random.randint(0, patch.shape[0])
        x2,y2 = np.random.randint(0, patch.shape[1]), np.random.randint(0, patch.shape[0])
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    patch = cv2.inpaint(patch, mask, 3, cv2.INPAINT_TELEA)
    return patch

def apply_contrast_distortion(patch):
    alpha = random.uniform(1.2, 1.5)  # contrast
    beta = random.randint(10, 40)     # brightness
    adjusted = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
    return adjusted

def generate_dataset(src_dir, out_dir, num_patches=200, size=(64,64)):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    for split in ["train","val"]:
        for cls in ["original","edited"]:
            (out_dir/split/cls).mkdir(parents=True, exist_ok=True)

    img_files = list(src_dir.glob("*"))
    if len(img_files) == 0:
        raise RuntimeError("No images found in source directory")

    train_split = int(num_patches * 0.8)

    for i in tqdm(range(num_patches), desc="Generating patches"):
        img = cv2.imread(str(random.choice(img_files)))
        if img is None: continue
        patch = random_patch(img, size=size)
        # save original
        split = "train" if i < train_split else "val"
        out_orig = out_dir / split / "original" / f"orig_{i}.png"
        cv2.imwrite(str(out_orig), patch)
        # create synthetic edit
        edited_patch = random.choice([
            apply_white_marker_effect,
            apply_digital_inpaint,
            apply_contrast_distortion
        ])(patch)
        out_edit = out_dir / split / "edited" / f"edit_{i}.png"
        cv2.imwrite(str(out_edit), edited_patch)
    print(f"Dataset created in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Folder with sample images (like receipts)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dataset folder")
    parser.add_argument("--num_patches", type=int, default=200)
    args = parser.parse_args()

    generate_dataset(args.src_dir, args.out_dir, args.num_patches)
