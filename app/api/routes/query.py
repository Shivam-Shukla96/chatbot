from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Optional
import os
import logging
from dotenv import load_dotenv
from groq import Groq
from ...services.search import search_similar
from ...services.synthesis import summarize_themes

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize GROQ client with API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")
client = Groq(
    api_key=api_key,
    base_url="https://api.groq.com/v1"
)

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

        
        # After getting results
        context = "\n".join([r["content"] for r in results])
        summary = summarize_themes(results, query=q)

        # Pass both results and original query for context-aware summarization
        # summary = summarize_themes(results, query=q)
        print(f"Summary:>> {summary}")
        return {
            "question": q,
            "answer": summary,
            "sources": [{"source": r["meta"]["source"], "similarity": r["meta"].get("similarity", 0)} for r in results]
        }
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the query: {str(ex)}"}
        )
