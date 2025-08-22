import os
import hashlib
import logging
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ...services.ocr_service import extract_text_from_file
from ...services.vector_store import store_text_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

upload_router = APIRouter()

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx', '.jpg', '.png'}
CHUNK_SIZE = 50000

def is_valid_file(file: UploadFile) -> bool:
    """
    Validate file extension and size.
    """
    try:
        if not file.filename:
            return False
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating file {file.filename}: {str(e)}")
        return False

def get_file_hash(content: bytes) -> str:
    """
    Generate SHA-256 hash of file content to prevent duplicates.
    """
    return hashlib.sha256(content).hexdigest()

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE) -> List[Dict[str, Any]]:
    """
    Splits extracted text into chunks with metadata.
    """
    try:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append({
                "content": chunk,
                "meta": {
                    "source": source,
                    "chunk_index": i // chunk_size,
                    "total_chunks": (len(words) + chunk_size - 1) // chunk_size
                }
            })
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text from {source}: {str(e)}")
        raise ValueError(f"Failed to chunk text: {str(e)}")

async def process_single_file(file: UploadFile) -> Dict[str, Any]:
    """
    Process a single file and return its results.
    """
    try:
        if not file.filename:
            return {
                "filename": "unknown",
                "status": "failed",
                "error": "No filename provided"
            }

        if not is_valid_file(file):
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            }

        # Generate unique filename using hash
        file_hash = get_file_hash(content)
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{file_hash}{ext}"
        file_path = os.path.join("uploads", unique_filename)

        # Save file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(content)

        # Extract text
        logger.info(f"Extracting text from {file.filename}")
        extracted = extract_text_from_file(file_path)

        if not extracted or len(extracted) == 0 or "failed" in str(extracted).lower():
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"Text extraction failed: {extracted}"
            }

        # Process chunks
        text_chunks = []
        if isinstance(extracted, str):
            text_chunks = chunk_text(extracted, source=file.filename or "unknown_file")
        elif isinstance(extracted, list):
            text_chunks = extracted
        else:
            return {
                "filename": file.filename,
                "status": "failed",
                "error": "Unsupported return type from text extraction"
            }

        if not text_chunks:
            return {
                "filename": file.filename,
                "status": "failed",
                "error": "No text could be extracted from the document"
            }

        # Store chunks
        result = store_text_chunks(text_chunks)
        
        return {
            "filename": file.filename,
            "status": "success",
            "chunks": len(text_chunks),
            "message": result,
            "file_hash": file_hash
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return {
            "filename": file.filename,
            "status": "failed",
            "error": str(e)
        }

@upload_router.post("/")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple files simultaneously.
    Returns a summary of the processing results for each file.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Process files concurrently
        tasks = [process_single_file(file) for file in files]
        results = await asyncio.gather(*tasks)

        # Calculate totals
        total_chunks = sum(
            result["chunks"] 
            for result in results 
            if result["status"] == "success"
        )

        successful_files = sum(1 for result in results if result["status"] == "success")
        failed_files = sum(1 for result in results if result["status"] == "failed")

        return {
            "status": "completed",
            "total_files": len(files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "results": results
        }

    except Exception as ex:
        logger.error(f"Error in upload endpoint: {str(ex)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"An error occurred while processing the uploaded documents: {str(ex)}",
                "results": []
            }
        )
