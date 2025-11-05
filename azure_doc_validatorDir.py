import os, re, json
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from rapidfuzz import fuzz

# === Load environment variables from .env ===
load_dotenv()

AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")

if not AZURE_FORM_ENDPOINT or not AZURE_FORM_KEY:
    raise EnvironmentError("❌ Azure credentials missing! Please set AZURE_FORM_ENDPOINT and AZURE_FORM_KEY in your .env file.")

# === Initialize Azure Client ===
client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_KEY)
)

# === Utility functions ===
def analyze_doc(path):
    with open(path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-document", document=f)
    result = poller.result()

    # Combine text from all lines across all pages
    text_parts = []
    for page in result.pages:
        for line in page.lines:
            text_parts.append(line.content)
    text = " ".join(text_parts).strip()

    # 🟢 Debug line — print how many words were extracted from this file
    print(f"[INFO] Extracted {len(text.split())} words from {path}")

    return text



def detect_type(text):
    low = text.lower()
    return {
        "form": any(k in low for k in ["mukhyamantri", "application id", "beneficiary", "scheme"]),
        "marksheet": any(k in low for k in ["marksheet", "result", "subject", "percentage"]),
        "receipt": any(k in low for k in ["receipt", "paid", "fee", "payment", "challan"])
    }

def extract_name(text):
    # Normalize spacing and punctuation
    clean_text = re.sub(r'[\n\r]+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.replace("...", ".").replace("…", ".")

    # 🟢 Case 1: "Name of Girl in English .... Ms. PRIYANSHI DEVI"
    pattern1 = re.search(
        r"Name\s*of\s*Girl\s*in\s*English[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        clean_text, re.I
    )
    if pattern1:
        name = pattern1.group(1).strip().title()
        return name.rstrip('.')

    # 🟢 Case 2: "Name of Applicant .... Ms. PRIYANSHI DEVI"
    pattern2 = re.search(
        r"Name\s*of\s*Applicant[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        clean_text, re.I
    )
    if pattern2:
        name = pattern2.group(1).strip().title()
        return name.rstrip('.')

    # 🟢 Case 3: generic fallback "Name:" or "Name -"
    pattern3 = re.search(
        r"\bName[:\-\s]+(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        clean_text, re.I
    )
    if pattern3:
        name = pattern3.group(1).strip().title()
        return name.rstrip('.')

    return None



def fuzzy_match(a, b):
    return fuzz.token_set_ratio(a, b) if a and b else 0

def classify(score):
    return "PASS" if score >= 85 else ("REVIEW" if score >= 60 else "FAIL")

# === Main Processing ===
def process_folder(input_dir="input_docs"):
    Path("results").mkdir(exist_ok=True)
    records = []

    for pdf in Path(input_dir).glob("*.pdf"):
        text = analyze_doc(str(pdf))
        types = detect_type(text)
        name = extract_name(text)
        records.append({"file": str(pdf), "types": types, "detected_name": name})

    form = max(records, key=lambda r: (r["types"]["form"], len(r["detected_name"] or "")))
    applicant = form["detected_name"]

    for r in records:
        r["score"] = fuzzy_match(applicant, r["detected_name"])
        r["status"] = classify(r["score"])

    output = {"applicant_name": applicant, "results": records}
    json_out = Path("results/document_validation_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Azure analysis complete for {applicant}")
    print(f"📄 Results saved in: {json_out}")

# === Run ===
if __name__ == "__main__":
    process_folder("input_docs")
