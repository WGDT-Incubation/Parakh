import os, re, json, io
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from rapidfuzz import fuzz
import fitz, requests
from PIL import Image
import numpy as np

# === Load environment variables ===
load_dotenv()
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_FACE_ENDPOINT = os.getenv("AZURE_FACE_ENDPOINT")
AZURE_FACE_KEY = os.getenv("AZURE_FACE_KEY")

if not AZURE_FORM_ENDPOINT or not AZURE_FORM_KEY:
    raise EnvironmentError("❌ Missing Form Recognizer credentials")
if not AZURE_FACE_ENDPOINT or not AZURE_FACE_KEY:
    raise EnvironmentError("❌ Missing Face API credentials")

# === Initialize Azure clients ===
form_client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_KEY)
)
face_client = FaceClient(AZURE_FACE_ENDPOINT, CognitiveServicesCredentials(AZURE_FACE_KEY))

# === --- TEXT EXTRACTION --- ===
def analyze_doc(path):
    with open(path, "rb") as f:
        poller = form_client.begin_analyze_document("prebuilt-document", document=f)
    result = poller.result()
    text_parts = [line.content for page in result.pages for line in page.lines]
    text = " ".join(text_parts).strip()
    print(f"[INFO] Extracted {len(text.split())} words from {path}")
    return text

# === --- IMAGE EXTRACTION --- ===
def extract_images(path):
    doc = fitz.open(path)
    imgs = []
    for i, page in enumerate(doc):
        for img in page.get_images(full=True):
            base_image = doc.extract_image(img[0])
            image_bytes = base_image["image"]
            imgs.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return imgs

# === --- NAME EXTRACTION --- ===
def extract_name(text):
    clean = re.sub(r'[\n\r]+', ' ', text)
    clean = re.sub(r'\s+', ' ', clean)
    clean = clean.replace("...", ".").replace("…", ".")
    patterns = [
        r"Name\s*of\s*Girl\s*in\s*English[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        r"Name\s*of\s*Applicant[^A-Za-z0-9]{0,10}(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})",
        r"\bName[:\-\s]+(?:Ms\.?|Mrs\.?|Miss\.?|Mr\.?)?\s*([A-Z][A-Z .']{2,50})"
    ]
    for pat in patterns:
        m = re.search(pat, clean, re.I)
        if m:
            return m.group(1).strip().rstrip('.').title()
    return None

# === --- CLASSIFIERS --- ===
def detect_type(text):
    low = text.lower()
    return {
        "form": any(k in low for k in ["mukhyamantri","application id","beneficiary","scheme"]),
        "marksheet": any(k in low for k in ["marksheet","result","subject","percentage"]),
        "receipt": any(k in low for k in ["receipt","paid","fee","payment","challan"])
    }

def fuzzy_match(a,b): return fuzz.token_set_ratio(a,b) if a and b else 0
def classify(score):  return "PASS" if score>=85 else ("REVIEW" if score>=60 else "FAIL")

# === --- FACE ANALYSIS HELPERS --- ===
def get_face_features(image):
    """Upload in-memory image to Face API and return detected faceId + quality metrics."""
    try:
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        if buf.getbuffer().nbytes > 6 * 1024 * 1024:
            image = image.resize((image.width // 2, image.height // 2))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        detected = face_client.face.detect_with_stream(
            image=buf,
            detection_model='detection_03',
            recognition_model='recognition_04',
            return_face_id=True,
            return_face_attributes=['blur', 'occlusion']
        )
        print(f"[FACE] {len(detected)} face(s) detected from image ({round(buf.getbuffer().nbytes/1024,1)} KB)")
        return detected
    except Exception as e:
        print(f"[ERROR] Face API call failed: {e}")
        return []


def compare_faces(face_ids):
    """Compare first face (form) to all others; return similarity scores."""
    if len(face_ids) < 2:
        return []
    base = face_ids[0]
    results = []
    for fid in face_ids[1:]:
        verify = face_client.face.verify_face_to_face(base, fid)
        results.append(verify.confidence)
    return results

# === --- MAIN PROCESS --- ===
def process_folder(input_dir="input_docs"):
    Path("results").mkdir(exist_ok=True)
    records=[]
    for pdf in Path(input_dir).glob("*.pdf"):
        text = analyze_doc(str(pdf))
        types = detect_type(text)
        name = extract_name(text)
        images = extract_images(str(pdf))

        # --- Face detection & quality ---
        faces = []
        for img in images[:3]:  # limit to first 3 images per doc
            try:
                detections = get_face_features(img)
                for f in detections:
                    faces.append({
                        "faceId": f.face_id,
                        "blur": f.face_attributes.blur.blur_level.value if f.face_attributes.blur else "Unknown",
                        "occluded": any([
                            f.face_attributes.occlusion.forehead_occluded if f.face_attributes.occlusion else False,
                            f.face_attributes.occlusion.eye_occluded if f.face_attributes.occlusion else False,
                            f.face_attributes.occlusion.mouth_occluded if f.face_attributes.occlusion else False
                        ])
                    })
            except Exception as e:
                print(f"[WARN] Face API failed on {pdf.name}: {e}")

        records.append({
            "file": str(pdf),
            "types": types,
            "detected_name": name,
            "faces": faces
        })

    # --- Establish applicant baseline ---
    form = max(records, key=lambda r: (r["types"]["form"], len(r["detected_name"] or "")))
    applicant = form["detected_name"]

    # --- Name-based matching ---
    for r in records:
        r["score"] = fuzzy_match(applicant, r["detected_name"])
        r["status"] = classify(r["score"])

    # --- Face comparison (form vs others) ---
    base_faces = [f["faceId"] for f in form["faces"] if f.get("faceId")]
    for r in records:
        if r == form or not base_faces:
            continue
        other_faces = [f["faceId"] for f in r["faces"] if f.get("faceId")]
        if not other_faces:
            r["photo_verification"] = {"status":"FAIL","reason":"No face detected"}
            continue
        try:
            sims = compare_faces(base_faces + other_faces)
            avg_conf = np.mean(sims) if sims else 0
            r["photo_verification"] = {
                "avg_confidence": round(float(avg_conf),3),
                "status": "PASS" if avg_conf>=0.65 else ("REVIEW" if avg_conf>=0.5 else "FAIL")
            }
        except Exception as e:
            r["photo_verification"]={"status":"ERROR","reason":str(e)}

        # --- Blur / Occlusion summary ---
        if any(f.get("blur")=="High" or f.get("occluded") for f in r["faces"]):
            r["photo_verification"]["blur_or_occlusion"]="True"

    output={"applicant_name":applicant,"results":records}
    json_out=Path("results/document_validation_results.json")
    with open(json_out,"w",encoding="utf-8") as f:
        json.dump(output,f,indent=2,ensure_ascii=False)

    print(f"\n✅ Azure analysis complete for {applicant}")
    print(f"📄 Results saved in: {json_out}")

# === --- RUN --- ===
if __name__=="__main__":
    process_folder("input_docs")
