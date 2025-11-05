import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# === Azure Keys ===
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")

client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT ,
    credential=AzureKeyCredential("109bc8d3-f621-46ae-b6d9-967d36fb59d8")
)
print("✅ Connected:", client is not None)
