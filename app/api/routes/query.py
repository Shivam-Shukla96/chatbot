from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.vector_store import search_similar
from app.services.synthesis import summarize_themes
from typing import Optional
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query_router = APIRouter()

@query_router.get('/')
def ask_question(q: str, source: Optional[str] = None):
    print(f"Received query: {q}, source: {source}")
    try:
        results = search_similar(q, source=source)
        # print(f"Search results: {results}")
        
        if isinstance(results, dict) and results.get("status") == "error":
            return JSONResponse(
                status_code=500,
                content={"error": results.get("message")}
            )

        if not isinstance(results, list):
            return JSONResponse(
                status_code=500,
                content={"error": "Invalid format for search results."}
            )
        print(f"Results---: {results}")

        summary = summarize_themes(results)
        print(f"Summary:>> {summary}")
        return {"question": q, "answer": summary}
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the query: {str(ex)}"}
        )
