from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from io import BytesIO

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()

    reader = PdfReader(BytesIO(content))

    raw_text = ""
    for page in reader.pages:
        raw_text += (page.extract_text() or "") + "\n"

    result = ingestion_graph.invoke({
        "raw_text": raw_text,
        "chunks": [],
        "embeddings_count": 0,
        "status": "uploaded"
    })

    return {
        "filename": file.filename,
        "status": result["status"],
        "chunks": len(result["chunks"]),
        "embeddings_count": result["embeddings_count"]
    }