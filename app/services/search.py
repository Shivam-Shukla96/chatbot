"""
search.py

This module handles semantic search functionality using the vector store.
"""

import logging
from typing import List, Dict, Any, Optional
from .vector_store import query_similar_chunks

logger = logging.getLogger(__name__)

def search_similar(query: str, source: Optional[str] = None, n_results: int = 5) -> List[Dict[str, Any]] | Dict[str, str]:
    """
    Search for similar content in the vector store.
    
    Args:
        query (str): The search query
        source (Optional[str]): Optional source file to filter results
        n_results (int): Number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of similar chunks with metadata and similarity scores
    """
    try:
        # Get similar chunks from vector store
        results = query_similar_chunks(query, n_results=n_results)
        
        if not results:
            logger.warning(f"No results found for query: {query}")
            return []
            
        # Filter by source if specified
        if source:
            results = [
                r for r in results 
                if r.get("metadata", {}).get("source", "").lower() == source.lower()
            ]
            
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "content": result["content"],
                "meta": {
                    "source": result["metadata"].get("source", "unknown"),
                    "similarity": result["similarity"],
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "total_chunks": result["metadata"].get("total_chunks", 1)
                }
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in search_similar: {str(e)}")
        return {"status": "error", "message": str(e)}
