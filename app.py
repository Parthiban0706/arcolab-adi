import os
import json
import pandas as pd
from statistics import mean
from flask import Flask
import gradio as gr
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Azure Document Intelligence Setup
# -----------------------------
endpoint = os.environ.get("AZURE_DI_ENDPOINT")  # Set this in Azure App Settings
key = os.environ.get("AZURE_DI_KEY")           # Set this in Azure App Settings
model_id = os.environ.get("AZURE_DI_MODEL_ID") or "arcolab-adi-model1-bmr"

# -----------------------------
# Analyze PDF using custom model
# -----------------------------
def analyze_custom_model(file_path, page_range=None):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=f,
            pages=page_range
        )
    result = poller.result()

    result_dict = {"pages": []}
    for page in result.pages:
        page_dict = {
            "pageNumber": page.page_number,
            "words": [{"content": w.content, "confidence": w.confidence} for w in page.words]
        }
        result_dict["pages"].append(page_dict)

    return result_dict

# -----------------------------
# Convert JSON to CSV
# -----------------------------
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
    csv_output_path = os.path.join("/tmp", os.path.basename(os.path.splitext(input_file)[0] + ".csv"))
    df.to_csv(csv_output_path, index=False, encoding="utf-8")
    return csv_output_path

# -----------------------------
# Gradio function
# -----------------------------
def process_pdf(uploaded_file, page_range):
    file_path = uploaded_file.name
    page_range = page_range.strip() or None
    analysis_result = analyze_custom_model(file_path, page_range)
    csv_file = json_to_csv(analysis_result, file_path)
    return csv_file

# -----------------------------
# Mount Gradio on Flask
# -----------------------------
@app.route("/")
def gradio_app():
    interface = gr.Interface(
        fn=process_pdf,
        inputs=[
            gr.File(label="Upload PDF", file_types=['.pdf']),
            gr.Textbox(label="Page Range (e.g., 1-3 or 2,4)", placeholder="Leave empty for all pages")
        ],
        outputs=gr.File(label="Download CSV"),
        title="Azure Document Intelligence PDF to CSV",
        description="Upload a PDF, optionally provide page range, and download extracted data as CSV"
    )
    return interface.launch(share=False, inline=True, prevent_thread_lock=True)

# -----------------------------
# Run Flask for local testing
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
