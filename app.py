from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
from statistics import mean
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import pandas as pd

# -----------------------------
# Azure Document Intelligence Setup
# -----------------------------
endpoint = "https://arcolab-adi-ci.cognitiveservices.azure.com/"
key = "EXqAjEMTaGXe366bGxgVhhGFekFDuE9n8q0zFrsCOJBl4zk0tFtzJQQJ99BIACGhslBXJ3w3AAALACOG7tmo"
model_id = "arcolab-adi-model1-bmr"

# -----------------------------
# Functions from your existing code
# -----------------------------
def analyze_custom_model(file_path, page_range=None):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(model_id=model_id, body=f, pages=page_range)
    result = poller.result()
    result_dict = {"pages": []}
    for page in result.pages:
        page_dict = {"pageNumber": page.page_number,
                     "words": [{"content": w.content, "confidence": w.confidence} for w in page.words]}
        result_dict["pages"].append(page_dict)
    return result_dict

def json_to_csv(data, input_file):
    pages = data.get("pages", [])
    structured_pages = []
    for page in pages:
        words = page.get("words", [])
        if not words:
            continue
        page_text = " ".join([w.get("content", "") for w in words])
        confidences = [w.get("confidence", 0) for w in words if "confidence" in w]
        structured_pages.append({
            "PageNumber": page.get("pageNumber"),
            "Text": page_text,
            "WordCount": len(words),
            "AverageConfidence": round(mean(confidences) * 100, 2) if confidences else None,
            "MinConfidence": round(min(confidences) * 100, 2) if confidences else None,
            "MaxConfidence": round(max(confidences) * 100, 2) if confidences else None
        })
    if not structured_pages:
        return None
    df = pd.DataFrame(structured_pages)
    csv_output_path = os.path.splitext(input_file)[0] + ".csv"
    df.to_csv(csv_output_path, index=False, encoding="utf-8")
    return csv_output_path

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Azure Document Intelligence PDF to CSV")

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), page_range: str = Form(None)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    page_range_value = page_range.strip() if page_range else None
    result = analyze_custom_model(file_path, page_range_value)
    csv_file = json_to_csv(result, file_path)
    
    return FileResponse(csv_file, media_type='text/csv', filename=os.path.basename(csv_file))
