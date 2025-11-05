import os, re, json, csv, numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import pdfplumber, fitz, pytesseract, spacy, face_recognition
from PIL import Image
from rapidfuzz import fuzz
from io import BytesIO
import pandas as pd
from fpdf import FPDF  # uses fpdf2 (UTF-8 safe)

# === CONFIG ===
INPUT_DIR = "input_docs"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD NER MODELS ===
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except:
    nlp = spacy.load("en_core_web_sm")

# === UTILS ===
def extract_text(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        pass
    if not text:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text("text")
    if len(text.strip()) < 50:
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr = pytesseract.image_to_string(img, lang="eng+hin")
            text += ocr
    return text.strip()

def extract_images(path):
    imgs = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            try:
                im = Image.open(BytesIO(image_data)).convert("RGB")
                imgs.append(im)
            except Exception as e:
                print(f"[WARN] Failed to extract image from page {i+1}: {e}")
    return imgs

def detect_type(text):
    low = text.lower()
    return {
        "form": any(k in low for k in ["mukhyamantri","application id","beneficiary","scheme"]),
        "marksheet": any(k in low for k in ["marksheet","result","subject","percentage"]),
        "receipt": any(k in low for k in ["receipt","paid","fee","payment","challan"])
    }

def extract_name(text):
    doc = nlp(text)
    names = [ent.text.strip() for ent in doc.ents if ent.label_ in ("PERSON","PER")]
    if names:
        return names[0]
    m = re.search(r"Name of Applicant[:\-]?\s*([A-Z][A-Za-z .']{2,50})", text, re.I)
    return m.group(1).strip() if m else None

def fuzzy_match(a,b):
    if not a or not b: 
        return 0
    return max(fuzz.partial_ratio(a,b), fuzz.token_set_ratio(a,b))

def classify(score):
    if score >= 80: return "PASS"
    elif score >= 50: return "REVIEW"
    else: return "FAIL"

# === PROCESS ===
records = []
for pdf in Path(INPUT_DIR).glob("*.pdf"):
    text = extract_text(str(pdf))
    types = detect_type(text)
    name = extract_name(text)
    images = extract_images(str(pdf))
    rec = {
        "file": str(pdf),
        "types": types,
        "detected_name": name,
        "text_len": len(text),
        "images": images
    }
    records.append(rec)

# Find form and applicant
form_rec = max(records, key=lambda r: (r["types"]["form"], r["text_len"]))
applicant_name = form_rec["detected_name"]

# Match each doc
results = []
for r in records:
    score = fuzzy_match(applicant_name or "", r["detected_name"] or "")
    r["score"] = round(score,2)
    r["status"] = classify(score)
    results.append(r)

# === PHOTO FACE MATCH ===
faces = {}
for r in records:
    encs = []
    for im in r["images"]:
        arr = np.array(im)
        enc = face_recognition.face_encodings(arr)
        if enc: 
            encs.extend(enc)
    faces[r["file"]] = encs

face_results = []
for i, r1 in enumerate(records):
    for j, r2 in enumerate(records):
        if j <= i: 
            continue
        sims = []
        for e1 in faces[r1["file"]]:
            for e2 in faces[r2["file"]]:
                sim = np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2))
                sims.append(sim)
        if sims:
            avg = np.mean(sims)
            face_results.append({
                "pair": (Path(r1["file"]).name, Path(r2["file"]).name),
                "similarity": round(avg,3),
                "status": "PASS" if avg >= 0.65 else "FAIL"
            })

# === SAVE OUTPUTS ===
json_out = Path(OUTPUT_DIR, "document_validation_results.json")
csv_out = Path(OUTPUT_DIR, "document_validation_results.csv")

pd.DataFrame([
    {
        "file": Path(r["file"]).name,
        "form": r["types"]["form"],
        "marksheet": r["types"]["marksheet"],
        "receipt": r["types"]["receipt"],
        "detected_name": r["detected_name"],
        "score": r["score"],
        "status": r["status"]
    }
    for r in results
]).to_csv(csv_out, index=False)

# ---- SAFE JSON SAVE ----
serializable_results = []
for r in results:
    r_copy = {k: v for k, v in r.items() if k != "images"}
    serializable_results.append(r_copy)

with open(json_out, "w", encoding="utf-8") as f:
    json.dump({
        "applicant_name": applicant_name,
        "results": serializable_results,
        "face_results": face_results
    }, f, indent=2, ensure_ascii=False)

# === GENERATE REPORT PDF (Unicode-safe with TrueType font) ===
from fpdf import FPDF

font_path = Path("fonts/DejaVuSans.ttf")
if not font_path.exists():
    raise FileNotFoundError(
        f"Unicode font not found at {font_path}. "
        "Download DejaVuSans.ttf from https://dejavu-fonts.github.io/."
    )

report = FPDF()
report.add_page()
report.add_font("DejaVu", "", str(font_path))  # no 'uni' param needed in fpdf2 ≥2.5
report.set_font("DejaVu", size=14)
report.multi_cell(190, 10, "📄 Document Validation Report")

report.set_font("DejaVu", size=12)
app_text = f"Applicant: {applicant_name or 'Unknown'}"
report.multi_cell(190, 10, app_text)
report.ln(5)

for r in results:
    line = f"{Path(r['file']).name} — {r['status']} ({round(r['score'],2)}%)"
    report.multi_cell(190, 8, line)
report.ln(5)

report.set_font("DejaVu", size=12, style="B")
report.multi_cell(190, 10, "Face Match Summary:")
report.set_font("DejaVu", size=11)
if face_results:
    for fr in face_results:
        line = (
            f"{fr['pair'][0]} ↔ {fr['pair'][1]} : "
            f"{fr['status']} (similarity={fr['similarity']})"
        )
        report.multi_cell(190, 8, line)
else:
    report.multi_cell(190, 8, "No faces detected or comparison skipped.")

out_pdf = Path(OUTPUT_DIR) / "validation_summary.pdf"
report.output(str(out_pdf))

print(f"\n✅ Processing complete.")
print(f"Applicant detected: {applicant_name}")
print(f"Results saved in: {OUTPUT_DIR}")
print(f"PDF Report: {out_pdf}")
