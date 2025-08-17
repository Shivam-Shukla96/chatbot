import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from typing import List

# def extract_text_from_file(file) -> List[dict]:
#     """
#     Extracts and chunks text from an uploaded file (PDF, image, or text).

#     Supports the following:
#     - PDFs with selectable text
#     - PDFs requiring OCR (if no text found)
#     - PNG, JPG, JPEG images using Tesseract OCR
#     - Plain text files (UTF-8 encoded)

#     Args:
#         file (UploadFile): The uploaded file object from FastAPI.

#     Returns:
#         List[dict]: A list of text chunks with metadata. If an error occurs during
#                     extraction, a single chunk with the error message is returned.
#     """
#     try:
#         contents = file.file.read()
#         filename = file.filename.lower()

#         if filename.endswith(".pdf"):
#             try:
#                 file.file.seek(0)
#                 reader = PdfReader(file.file)
#                 text = ""
#                 for page in reader.pages:
#                     text += page.extract_text() or ""

#                 if not text.strip():  # fallback to OCR
#                     images = convert_from_bytes(contents)
#                     text = "\n".join([pytesseract.image_to_string(img) for img in images])
#             except Exception as pdf_error:
#                 return [{"content": f"PDF processing failed: {str(pdf_error)}", "meta": {"source": "upload"}}]

#         elif filename.endswith((".png", ".jpg", ".jpeg")):
#             try:
#                 text = pytesseract.image_to_string(file.file)
#             except Exception as img_error:
#                 return [{"content": f"Image OCR failed: {str(img_error)}", "meta": {"source": "upload"}}]

#         else:
#             try:
#                 text = contents.decode("utf-8")
#             except Exception as decode_error:
#                 return [{"content": f"Text decoding failed: {str(decode_error)}", "meta": {"source": "upload"}}]

#         return chunk_text(text)

#     except Exception as general_error:
#         return [{"content": f"File processing failed: {str(general_error)}", "meta": {"source": "upload"}}]

import os
from typing import Optional
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file (PDF, image, or plain text).

    Supports:
    - PDFs with selectable text
    - PDFs requiring OCR (if no text found)
    - PNG, JPG, JPEG images using Tesseract OCR
    - Plain text files (UTF-8 encoded)

    Args:
        file_path (str): The path to the file saved locally.

    Returns:
        str: Extracted text content as a string.
             If extraction fails, returns an error message string.
    """
    try:
        filename = file_path.lower()

        # ----------------------------
        # Case 1: PDF handling
        # ----------------------------
        if filename.endswith(".pdf"):
            try:
                text = ""
                reader = PdfReader(file_path)

                # Try extracting text normally
                for page in reader.pages:
                    text += page.extract_text() or ""

                # If no selectable text, fallback to OCR
                if not text.strip():
                    images = convert_from_path(file_path)
                    text = "\n".join([pytesseract.image_to_string(img) for img in images])

            except Exception as pdf_error:
                return f"PDF processing failed: {str(pdf_error)}"

        # ----------------------------
        # Case 2: Image handling
        # ----------------------------
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                text = pytesseract.image_to_string(file_path)
            except Exception as img_error:
                return f"Image OCR failed: {str(img_error)}"

        # ----------------------------
        # Case 3: Plain text handling
        # ----------------------------
        elif filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as txt_error:
                return f"Text decoding failed: {str(txt_error)}"

        # ----------------------------
        # Case 4: Unsupported file
        # ----------------------------
        else:
            return f"Unsupported file format for: {filename}"

        return text if text.strip() else "No text could be extracted from the file."

    except Exception as general_error:
        return f"File processing failed: {str(general_error)}"

def chunk_text(text: str, max_tokens: int = 300) -> List[dict]:
    """
    Splits a large block of text into smaller paragraph-based chunks.

    Args:
        text (str): The raw extracted text to be chunked.
        max_tokens (int): Placeholder for future token-based chunking. Currently unused.

    Returns:
        List[dict]: List of dictionaries containing individual paragraph chunks
                    with metadata.
    """
    paragraphs = text.split("\n\n")
    return [{"content": p.strip(), "meta": {"source": "upload"}} for p in paragraphs if p.strip()]
