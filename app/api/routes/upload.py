from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.ocr_service import extract_text_from_file
from app.services.vector_store import store_text_chunks

upload_router = APIRouter()

@upload_router.post("/")
async def upload_document(file: UploadFile = File(...)):
    try:
        text_chunks = extract_text_from_file(file)
        store_text_chunks(text_chunks)
        return {
            "status": "uploaded",
            "chunks": len(text_chunks)
            }
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the uploaded document : {str(ex)}"}
        )
