"""
vector_store.py

This module handles storage and retrieval of text embeddings using ChromaDB and SentenceTransformers.
It supports storing document chunks along with metadata and querying the most semantically similar content.

Dependencies:
- chromadb
- sentence-transformers

"""

import chromadb
# from chromadb.config import Settings  # Removed unused import
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB persistent client and sentence embedding model
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="docs")
model = SentenceTransformer("all-MiniLM-L6-v2")


def store_text_chunks(chunks):
    """
    Stores text chunks in ChromaDB along with their sentence embeddings.

    Each chunk is embedded using a pre-trained SentenceTransformer model and added to the ChromaDB collection
    with corresponding metadata and a unique ID.

    Args:
        chunks (list of dict): List of dictionaries containing:
            - 'content' (str): Text content of the chunk.
            - 'meta' (dict): Metadata dictionary containing at least the 'source' (filename or origin).

    Returns:
        dict: Dictionary containing status and number of chunks stored or an error message.
    """
    try:
        # Get current number of stored documents to generate unique IDs
        current_size = len(collection.get()['ids'])

        # Loop through each chunk to embed and store
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk['content']).tolist()

            # Add the chunk with its embedding, metadata, and ID
            collection.add(
                documents=[chunk['content']],
                embeddings=[embedding],
                metadatas=[chunk['meta']],
                ids=[f"chunk_{current_size + i}"]
            )
            

        return {"status": "success", "message": f"{len(chunks)} chunks stored."}

    except Exception as e:
        return {"status": "error", "message": f"Failed to store chunks: {str(e)}"}


# def search_similar(query, k=5, source=None):
#     """
#     Searches the ChromaDB collection for the top-k most semantically similar documents to a given query.

#     The function encodes the query using the same SentenceTransformer model used for storage, retrieves
#     the top-k most relevant documents, and returns both their content and metadata along with similarity scores.

#     Args:
#         query (str): The user's input query for semantic search.
#         k (int, optional): Number of top results to return. Defaults to 5.
#         source (str, optional): The source document to filter by. Defaults to None.

#     Returns:
#         list or dict: List of dictionaries with 'content', 'meta' (including similarity score), and
#                      source information for each matched document, or an error dictionary in case of failure.
#     """
#     try:
#         # Encode query to obtain its vector representation
#         q_emb = model.encode(query).tolist()

#         # Query the ChromaDB collection for top-k matches
#         query_params = {
#             "query_embeddings": [q_emb],
#             "n_results": k,
#             "include": ['documents', 'metadatas', 'distances']
#         }
#         if source:
#             query_params["where"] = {"source": source}
        
#         results = collection.query(**query_params)
#         print(f"Query: {query}")
        
#         # Safely extract results
#         if not results or not isinstance(results, dict):
#             return {"status": "error", "message": "Search failed: no results returned"}
            
#         if 'documents' not in results or not results['documents'] or not results['documents'][0]:
#             return {"status": "error", "message": "No matching documents found"}
            
#         documents = results['documents'][0]
#         distances = []
#         metadatas = []
        
#         if 'distances' in results and results['distances']:
#             distances = results['distances'][0]
#         if 'metadatas' in results and results['metadatas']:
#             metadatas = results['metadatas'][0]
            
#         print(f"Found {len(documents)} matches")
        
#         # Convert distances to similarity scores and prepare results
#         results_list = []
#         for i, doc in enumerate(documents):
#             # Calculate similarity score (0 to 1 scale)
#             # similarity = 0.5  # Default mid-range similarity
#             # if distances:
#             #     max_distance = max(distances)
#             #     if max_distance > 0:
#             #         similarity = 1.0 - (distances[i] / max_distance)

#             # NEW (use cosine similarity directly)
#          similarity = 1.0 - distances[i] if distances else 0.0

#             # Get metadata
#         meta = metadatas[i] if i < len(metadatas) else {}
#         if not isinstance(meta, dict):
#                 meta = {}
            
#             # Create result entry
#         result = {
#                 "content": doc,
#                 "meta": {
#                     "source": meta.get("source", "unknown"),
#                     "similarity": similarity
#                 }
#             }
#         results_list.append(result)
#         print(f"Document {i+1} similarity: {similarity}")
        
#         if not results_list:
#             return {"status": "error", "message": "No relevant content found for the query."}
            
#         # Sort by similarity
#         results_list.sort(key=lambda x: x["meta"]["similarity"], reverse=True)
#         return results_list

#     except Exception as e:
#         return {"status": "error", "message": f"Search failed: {str(e)}"}

# def search_similar(query, k=5, source=None, min_similarity: float = 0.3):
def search_similar(query, k=5, source=None, ):
    """
    Searches the ChromaDB collection for the top-k most semantically similar documents to a given query.

    Args:
        query (str): The user's input query.
        k (int, optional): Number of top results to return. Defaults to 5.
        source (str, optional): Filter by document source. Defaults to None.
        min_similarity (float, optional): Minimum similarity threshold (0–1). Defaults to 0.3.

    Returns:
        list or dict: List of dicts with 'content' and 'meta' (source + similarity).
    """
    try:
        # Encode query
        q_emb = model.encode(query).tolist()

        # Query Chroma
        query_params = {
            "query_embeddings": [q_emb],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if source:
            query_params["where"] = {"source": source}
        
        results = collection.query(**query_params)
        print(f"Query: {query}")
        
        if not results or not isinstance(results, dict):
            return {"status": "error", "message": "Search failed: no results returned"}
            
        if not results.get("documents") or results["documents"] is None or not results["documents"][0]:
            return {"status": "error", "message": "No matching documents found"}
            
        documents = results["documents"][0]
        distances = results.get("distances")
        if distances is not None and len(distances) > 0:
            distances = distances[0]
        else:
            distances = []
        metadatas = results.get("metadatas")
        if metadatas is not None and len(metadatas) > 0:
            metadatas = metadatas[0]
        else:
            metadatas = []        
        print(f"Found {len(documents)} matches")

        results_list = []
        for i, doc in enumerate(documents):
            similarity = 1.0 - distances[i] if distances else 0.0

            # 🔑 Apply minimum similarity filter
            # if similarity < min_similarity:
            #     print(f"Skipping document {i+1}: similarity {similarity:.3f} below threshold {min_similarity}")
            #     continue

            meta = metadatas[i] if i < len(metadatas) else {}
            if not isinstance(meta, dict):
                meta = {}

            result = {
                "content": doc,
                "meta": {
                    "source": meta.get("source", "unknown"),
                    "similarity": similarity
                }
            }
            results_list.append(result)
            print(f"Document {i+1} similarity: {similarity:.3f}")
        
        if not results_list:
            return {"status": "error", "message": "No relevant content found for the query."}
            
        # Sort by similarity (high → low)
        results_list.sort(key=lambda x: x["meta"]["similarity"], reverse=True)
        return results_list

    except Exception as e:
        return {"status": "error", "message": f"Search failed: {str(e)}"}
