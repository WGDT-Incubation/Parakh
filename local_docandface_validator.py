import os, re, json, io
from pathlib import Path
from dotenv import load_dotenv
import fitz
import numpy as np
from PIL import Image
import pdfplumber
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import face_recognition
from rapidfuzz import fuzz

# === Load environment ===
load_dotenv()
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")

if not AZURE_FORM_ENDPOINT or not AZURE_FORM_KEY:
    raise EnvironmentError("❌ Missing Azure Form Recognizer credentials")

# === Azure client ===
client = DocumentAnalysisClient(endpoint=AZURE_FORM_ENDPOINT,
                                credential=AzureKeyCredential(AZURE_FORM_KEY))

# === Util: extract text ===
def analyze_doc(path):
    with open(path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-document", document=f)
    result = poller.result()

    text_parts = []
    for page in result.pages:
        for line in page.lines:
            text_parts.append(line.content)
    text = " ".join(text_parts).strip()
    print(f"[INFO] Extracted {len(text.split())} words from {path}")
    return text

# === Util: extract images from PDF ===
def extract_images(pdf_path):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(doc):
            for img in page.get_images(full=True):
                base_img = doc.extract_image(img[0])
                img_bytes = base_img["image"]
                try:
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(im)
                except Exception as e:
                    print(f"[WARN] Image decode failed on {pdf_path} p{page_idx+1}: {e}")
    except Exception as e:
        print(f"[WARN] Could not read {pdf_path}: {e}")
    return images

# === Util: detect doc type ===
def detect_type(text):
    low = text.lower()
    return {
        "form": any(k in low for k in ["mukhyamantri", "application id", "beneficiary", "scheme"]),
        "marksheet": any(k in low for k in ["marksheet", "result", "subject", "percentage"]),
        "receipt": any(k in low for k in ["receipt", "paid", "fee", "payment", "challan"])
    }

# === Util: extract applicant name ===
def extract_name(text):
    clean_text = re.sub(r'[\n\r]+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    pattern = re.search(
        r"Name\s*of\s*Girl\s*in\s*English[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        clean_text, re.I)
    if pattern:
        return pattern.group(1).strip().title().rstrip('.')
    return None

# === Util: fuzzy name match ===
def fuzzy_match(a, b):
    return fuzz.token_set_ratio(a or "", b or "")

def classify(score):
    return "PASS" if score >= 85 else ("REVIEW" if score >= 60 else "FAIL")

# === Util: local face comparison ===
def compare_faces(img_list):
    """Compare first face found (form) vs others, return average similarity."""
    encodings = []
    for im in img_list:
        arr = np.array(im)
        enc = face_recognition.face_encodings(arr)
        if enc:
            encodings.append(enc[0])

    if len(encodings) < 2:
        return {"faces_found": len(encodings), "avg_similarity": None, "status": "NO_FACE"}

    base = encodings[0]
    sims = []
    for other in encodings[1:]:
        sim = np.dot(base, other) / (np.linalg.norm(base) * np.linalg.norm(other))
        sims.append(sim)

    avg = float(np.mean(sims))
    status = "PASS" if avg >= 0.65 else "FAIL"
    return {"faces_found": len(encodings), "avg_similarity": round(avg, 3), "status": status}

# === Main ===
def process_folder(input_dir="input_docs"):
    Path("results").mkdir(exist_ok=True)
    records = []

    for pdf in Path(input_dir).glob("*.pdf"):
        text = analyze_doc(str(pdf))
        types = detect_type(text)
        name = extract_name(text)
        imgs = extract_images(str(pdf))
        face_result = compare_faces(imgs)
        records.append({
            "file": str(pdf),
            "types": types,
            "detected_name": name,
            "face_result": face_result
        })

    # pick applicant
    form = max(records, key=lambda r: (r["types"]["form"], len(r["detected_name"] or "")))
    applicant = form["detected_name"]

    for r in records:
        r["name_score"] = fuzzy_match(applicant, r["detected_name"])
        r["name_status"] = classify(r["name_score"])

    output = {"applicant_name": applicant, "results": records}
    json_out = Path("results/document_validation_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Validation complete for {applicant}")
    print(f"📄 Results saved in: {json_out}")

if __name__ == "__main__":
    process_folder("input_docs")
