
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests, io
from PIL import Image

ENDPOINT = "https://pm-docface-cagup.cognitiveservices.azure.com/"
KEY = "" #PM15

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Download a sample face image from the web
url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg"
img_bytes = requests.get(url).content

print("Testing Azure Face API...")
detected = face_client.face.detect_with_stream(
    image=io.BytesIO(img_bytes),
    detection_model='detection_03',
    recognition_model='recognition_04',
    return_face_id=True,
)
print(f"✅ {len(detected)} face(s) detected")
