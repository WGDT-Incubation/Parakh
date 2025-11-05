import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# === Azure Keys ===
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")

client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT ,
    credential=AzureKeyCredential("") #PM15
)
print("✅ Connected:", client is not None)
