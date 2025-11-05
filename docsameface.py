import streamlit as st
import os
from pdf2image import convert_from_path
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
from io import BytesIO
import tempfile
import uuid

# -------------------------
# CONFIGURATION
# -------------------------
AZURE_FACE_ENDPOINT = "https://pm-docface-cagup.cognitiveservices.azure.com/" #PM15 
AZURE_FACE_KEY = "" #PM15 


face_client = FaceClient(AZURE_FACE_ENDPOINT, CognitiveServicesCredentials(AZURE_FACE_KEY))

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to list of PIL Images"""
    return convert_from_path(pdf_path, dpi=200)

def detect_faces_azure(image_bytes):
    """Detect faces safely with validation"""
    try:
        if not image_bytes or len(image_bytes) < 5000:
            raise ValueError("Image too small or empty")

        image_stream = BytesIO(image_bytes)
        detected_faces = face_client.face.detect_with_stream(
            image=image_stream,
            detection_model="detection_03",
            recognition_model="recognition_04",
            return_face_id=True,
            return_face_rectangle=True
        )
        return detected_faces

    except Exception as e:
        st.warning(f"Error detecting face: {e}")
        return []


def find_similar_faces(face_id, face_id_list):
    """Find similar faces from a list of faceIds"""
    try:
        similar_faces = face_client.face.find_similar(face_id=face_id, face_ids=face_id_list)
        return similar_faces
    except Exception as e:
        st.warning(f"Error in find_similar: {e}")
        return []

def image_bytes_from_pil(img):
    """Convert PIL image to clean RGB JPEG bytes for Azure Face API"""
    if img.mode != "RGB":
        img = img.convert("RGB")  # ✅ ensure RGB
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG", quality=90)
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


def crop_face(pil_image, rect):
    """Crop face using rectangle coords returned by Azure"""
    left = rect.left
    top = rect.top
    width = rect.width
    height = rect.height
    return pil_image.crop((left, top, left + width, top + height))

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Azure Face Duplicate Photo Detector", layout="wide")
st.title("Parakh – Duplicate Passport Photo Detector")

st.markdown("""
This app scans all PDFs and image files in a chosen directory,
detects faces using **Azure Face API**, and finds duplicate passport-size photos
across different documents.
""")

# Directory input
dir_path = st.text_input("📂 Enter directory path containing PDF/Image files:")

if dir_path and os.path.isdir(dir_path):

    st.info(f"Reading files from: `{dir_path}`")

    # Collect all PDF and image files
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]

    if len(files) == 0:
        st.warning("No PDF or image files found in the given directory.")
    else:
        st.write(f"Found {len(files)} documents.")

        start_btn = st.button("🚀 Start Analysis")

        if start_btn:
            with st.spinner("Processing documents with Azure Face API..."):

                face_records = []
                # 1️⃣ Process each file
                for file_path in files:
                    try:
                        if file_path.lower().endswith('.pdf'):
                            pages = convert_pdf_to_images(file_path)
                        else:
                            pages = [Image.open(file_path)]

                        for p_idx, img in enumerate(pages):
                            img_bytes = image_bytes_from_pil(img)
                            
                            detected = detect_faces_azure(img_bytes)

                            for f_idx, face in enumerate(detected):
                                face_crop = crop_face(img, face.face_rectangle)
                                face_id = face.face_id
                                temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
                                face_crop.save(temp_path)

                                face_records.append({
                                    "file": os.path.basename(file_path),
                                    "page": p_idx + 1,
                                    "face_id": face_id,
                                    "crop_path": temp_path
                                })
                    except Exception as e:
                        st.error(f"Error processing {file_path}: {e}")

                st.success(f"Detected {len(face_records)} total faces across all documents.")

                # 2️⃣ Compare all faces pairwise
                results = []
                all_face_ids = [r["face_id"] for r in face_records]

                for i, record in enumerate(face_records):
                    others = [fid for j, fid in enumerate(all_face_ids) if j != i]
                    matches = find_similar_faces(record["face_id"], others)
                    for m in matches:
                        if m.confidence >= 0.75:
                            match_record = next((r for r in face_records if r["face_id"] == m.face_id), None)
                            if match_record and record["file"] != match_record["file"]:
                                results.append({
                                    "File A": record["file"],
                                    "Page A": record["page"],
                                    "File B": match_record["file"],
                                    "Page B": match_record["page"],
                                    "Confidence": round(m.confidence, 3),
                                    "FaceA": record["crop_path"],
                                    "FaceB": match_record["crop_path"]
                                })

                # 3️⃣ Display results
                if len(results) == 0:
                    st.info("✅ No duplicate passport photos found across documents.")
                else:
                    st.success(f"Found {len(results)} potential duplicate photo pairs:")
                    for idx, res in enumerate(results, 1):
                        st.markdown(f"### Match #{idx}")
                        col1, col2, col3 = st.columns([1,1,1])
                        with col1:
                            st.image(res["FaceA"], caption=f"{res['File A']} (Page {res['Page A']})")
                        with col2:
                            st.image(res["FaceB"], caption=f"{res['File B']} (Page {res['Page B']})")
                        with col3:
                            st.write(f"**Confidence:** {res['Confidence']}")
                            st.write("Potential duplicate detected 🔁")
else:
    st.warning("Please enter a valid directory path to start.")
