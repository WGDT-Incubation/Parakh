#!/usr/bin/env python3
"""
batch_duplicate_detector.py

Compute perceptual hashes (pHash) for images in a folder and group near-duplicates.
Saves CSV with image path, hash, and cluster id.

Usage:
  python batch_duplicate_detector.py --img_dir ./all_photos --out_csv duplicates.csv --threshold 8

Dependencies:
  pip install pillow imagehash pandas tqdm
"""

import argparse
from pathlib import Path
import imagehash
from PIL import Image
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import networkx as nx

def compute_phash(image_path):
    try:
        im = Image.open(image_path).convert('RGB')
        h = imagehash.phash(im)
        return str(h)
    except Exception:
        return None

def hamming_distance_hash(h1, h2):
    # imagehash hex string -> binary distance by using ImageHash library functions
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)

def cluster_hashes(paths_hashes, threshold=8):
    # Build graph with edges where Hamming distance <= threshold, then connected components
    G = nx.Graph()
    for p,h in paths_hashes:
        G.add_node(p, hash=h)
    # pairwise compare (ok for moderate size). For huge datasets use LSH or blocking.
    for (p1,h1),(p2,h2) in tqdm(list(combinations(paths_hashes,2)), desc="Comparing pairs"):
        if h1 is None or h2 is None: continue
        d = hamming_distance_hash(h1,h2)
        if d <= threshold:
            G.add_edge(p1,p2, weight=d)
    clusters = list(nx.connected_components(G))
    mapping = {}
    for i,comp in enumerate(clusters):
        for p in comp:
            mapping[p] = i
    # nodes with no edges will be their own component as well
    for p,h in paths_hashes:
        if p not in mapping:
            mapping[p] = max(mapping.values())+1 if mapping else 0
    return mapping

def main(img_dir, out_csv, threshold=8):
    p = Path(img_dir)
    imgs = sorted([str(x) for x in p.glob("*") if x.is_file()])
    rows = []
    phashes = []
    for img in tqdm(imgs, desc="Hashing images"):
        h = compute_phash(img)
        rows.append({"path": img, "phash": h})
        phashes.append((img,h))
    # cluster
    mapping = cluster_hashes(phashes, threshold=threshold)
    # build df
    for r in rows:
        r["cluster_id"] = mapping.get(r["path"], -1)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="duplicates.csv")
    parser.add_argument("--threshold", type=int, default=8)
    args = parser.parse_args()
    main(args.img_dir, args.out_csv, args.threshold)
