import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.ocr_service import extract_text_from_file
from app.services.vector_store import store_text_chunks

upload_router = APIRouter()

def chunk_text(text: str, source: str, chunk_size: int = 50000):
    """
    Splits extracted text into chunks with metadata.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append({"content": chunk, "meta": {"source": source}})
    return chunks

@upload_router.post("/")
async def upload_document(file: UploadFile = File(...)):
    try:
        print(f"Received file--: {file.filename}, size: {file.size} bytes")
        # Save uploaded file locally (so OCR/text extractor can use it)
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename or "uploaded_file")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text using OCR service
        extracted = extract_text_from_file(file_path)  # <-- Pass path, not UploadFile

        if not extracted or len(extracted) == 0 or "failed" in extracted.lower():
            raise ValueError(f"Text extraction failed: {extracted}")

        # Split into chunks and embed
        if isinstance(extracted, str):
            text_chunks = chunk_text(extracted, source=file.filename or "unknown_filename")
        elif isinstance(extracted, list):  
            # Already chunked by OCR service
            text_chunks = extracted
        else:
            raise ValueError("Unsupported return type from extract_text_from_file")

        if not text_chunks:
            raise ValueError("No text/chunks could be extracted from the document.")
        result = store_text_chunks(text_chunks)

        return {
            "status": "uploaded",
            "chunks": len(text_chunks),
            "message": result
        }

    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the uploaded document: {str(ex)}"}
        )




# @upload_router.post("/")
# async def upload_document(file: UploadFile = File(...)):
#     try:
#         text_chunks = extract_text_from_file(file)
#         store_text_chunks(text_chunks)
#         return {
#             "status": "uploaded",
#             "chunks": len(text_chunks)
#             }
#     except Exception as ex:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"An error occurred while processing the uploaded document : {str(ex)}"}
#         )
