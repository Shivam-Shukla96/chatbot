"""
vector_store.py

This module handles storage and retrieval of text embeddings using ChromaDB and OpenAI.
It supports storing document chunks along with metadata and querying the most semantically similar content.
"""

import os
import logging
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app.core.config import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Configure logging
logger = logging.getLogger(__name__)

def initialize_vector_store():
    """Initialize ChromaDB client with error handling"""
    try:
        # Ensure directory exists
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        collection = client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Successfully initialized ChromaDB at {settings.CHROMA_DB_PATH}")
        
        return client, collection
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

# Initialize components
client, collection = initialize_vector_store()

Embedding = List[float]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using OpenAI's API"""
    try:
        if not texts:
            return []
            
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            
        # Clean and validate inputs
        validated_texts = [str(text).strip() for text in texts]
        if not any(validated_texts):
            logger.warning("No valid text to embed after cleaning")
            return []
            
        # Create OpenAI client
        if not api_key:
            raise ValueError("OpenAI API key is not set")
        client = OpenAI(api_key=api_key)
            
        # Get embeddings from OpenAI
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=validated_texts
        )
        # Extract embeddings from response
        embeddings = [data.embedding for data in response.data]
        return embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings from OpenAI: {str(e)}")
        raise

def batch_encode(texts: List[str]) -> List[List[float]]:
    """
    Encode a batch of texts using OpenAI embeddings API
    
    Args:
        texts (List[str]): List of text strings to encode
        
    Returns:
        List[List[float]]: List of embeddings
    """
    try:
        return get_embeddings(texts)
    except Exception as e:
        logger.error(f"Error encoding batch: {str(e)}")
        raise

def store_text_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stores text chunks in ChromaDB along with their sentence embeddings.

    Args:
        chunks (List[Dict]): List of dictionaries containing:
            - 'content' (str): Text content of the chunk
            - 'meta' (Dict): Metadata dictionary with at least 'source' (filename/origin)

    Returns:
        Dict: Status of the storage operation
    """
    try:
        if not chunks:
            return {"status": "error", "message": "No chunks provided"}

        # Get current collection size for ID generation
        current_size = len(collection.get()['ids'])
        
        # Prepare batch data
        contents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['meta'] for chunk in chunks]
        ids = [f"chunk_{current_size + i}" for i in range(len(chunks))]

        # Process in batches
        total_chunks = len(chunks)
        processed = 0
        batch_size = settings.BATCH_SIZE

        while processed < total_chunks:
            batch_end = min(processed + batch_size, total_chunks)
            batch_slice = slice(processed, batch_end)
            
            batch_contents = contents[batch_slice]
            batch_metadatas = metadatas[batch_slice]
            batch_ids = ids[batch_slice]

            # Generate embeddings for the batch using OpenAI
            try:
                batch_embeddings = get_embeddings(batch_contents)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                return {"status": "error", "message": f"OpenAI embedding generation failed: {str(e)}"}

            # Add to ChromaDB
            try:
                # Convert embeddings to numpy array for ChromaDB
                embeddings_array = np.array([np.array(emb, dtype=np.float64) for emb in batch_embeddings])
                collection.add(
                    embeddings=embeddings_array.tolist(),  # ChromaDB expects List[List[float]]
                    documents=batch_contents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                processed += len(batch_contents)
                logger.info(f"Processed {processed}/{total_chunks} chunks")
            
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {str(e)}")
                return {"status": "error", "message": f"ChromaDB storage failed: {str(e)}"}

        return {
            "status": "success",
            "message": f"Successfully stored {total_chunks} chunks",
            "chunks_stored": total_chunks
        }

    except Exception as e:
        logger.error(f"Error in store_text_chunks: {str(e)}")
        return {"status": "error", "message": str(e)}

# Alias for backward compatibility
def search_similar(query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
    return query_similar_chunks(query_text, n_results)

def query_similar_chunks(query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector store for chunks similar to the input text.
    
    Args:
        query_text (str): The text to find similar chunks for
        n_results (int): Number of results to return
        
    Returns:
        List[Dict]: List of similar chunks with their metadata and similarity scores
    """
    try:
        # Generate query embedding using OpenAI
        query_embedding = get_embeddings([query_text])[0]
        
        # Query the collection
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            return []

        # Format results
        formatted_results = []
        try:
            if results and isinstance(results, dict):
                # Safely get results
                ids_list = results.get('ids', [])
                docs_list = results.get('documents', [])
                meta_list = results.get('metadatas', [])
                dist_list = results.get('distances', [])

                if ids_list and len(ids_list) > 0:
                    ids = ids_list[0]
                    documents = docs_list[0] if docs_list else []
                    metadatas = meta_list[0] if meta_list else []
                    distances = dist_list[0] if dist_list else []

                    for i in range(len(ids)):
                        formatted_results.append({
                            "content": documents[i] if i < len(documents) else "",
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "similarity": 1 - float(distances[i]) if i < len(distances) else 0.0
                        })
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return []
            
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        return []
