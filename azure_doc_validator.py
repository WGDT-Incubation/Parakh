import os, re, json, argparse
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from rapidfuzz import fuzz

# === Load environment variables ===
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

# -----------------------------
# Utility functions
# -----------------------------
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


def detect_type(text):
    low = text.lower()
    return {
        "form": any(k in low for k in ["mukhyamantri", "application id", "beneficiary", "scheme"]),
        "marksheet": any(k in low for k in ["marksheet", "result", "subject", "percentage"]),
        "receipt": any(k in low for k in ["receipt", "paid", "fee", "payment", "challan"])
    }


def extract_name(text):
    clean_text = re.sub(r'[\n\r]+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.replace("...", ".").replace("…", ".")

    patterns = [
        r"Name\s*of\s*Girl\s*in\s*English[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        r"Name\s*of\s*Applicant[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        r"\bName[:\-\s]+(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})"
    ]

    for p in patterns:
        m = re.search(p, clean_text, re.I)
        if m:
            name = m.group(1).strip().title().rstrip('.')
            return name
    return None


def fuzzy_match(a, b):
    return fuzz.token_set_ratio(a, b) if a and b else 0


def classify(score):
    return "PASS" if score >= 85 else ("REVIEW" if score >= 60 else "FAIL")


# -----------------------------
# Core logic
# -----------------------------
def process_folder(input_dir="input_docs", output_path="results/document_validation_results.json"):
    input_dir = Path(input_dir)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pdfs = list(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️ No PDF files found in {input_dir.resolve()}")
        return None

    records = []
    for pdf in pdfs:
        text = analyze_doc(str(pdf))
        types = detect_type(text)
        name = extract_name(text)
        records.append({"file": str(pdf), "types": types, "detected_name": name})

    # Determine form document (the one likely containing applicant name)
    form = max(records, key=lambda r: (r["types"]["form"], len(r["detected_name"] or "")))
    applicant = form["detected_name"] or "Unknown Applicant"

    for r in records:
        r["score"] = fuzzy_match(applicant, r["detected_name"])
        r["status"] = classify(r["score"])

    output = {"applicant_name": applicant, "results": records}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Azure analysis complete for {applicant}")
    print(f"📄 Results saved in: {output_path}")
    return output_path


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Azure Document Validator")
    parser.add_argument("--input_dir", type=str, default="input_docs", help="Folder containing PDF files")
    parser.add_argument("--output", type=str, default="results/document_validation_results.json", help="Output JSON path")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output)
