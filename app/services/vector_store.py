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
client = chromadb.PersistentClient(path="./chroma_db")
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
                metadatas=[chunk['meta']],
                ids=[f"chunk_{current_size + i}"],
                embeddings=[embedding]
            )

        return {"status": "success", "message": f"{len(chunks)} chunks stored."}

    except Exception as e:
        return {"status": "error", "message": f"Failed to store chunks: {str(e)}"}


def search_similar(query, k=5, source=None):
    """
    Searches the ChromaDB collection for the top-k most semantically similar documents to a given query.

    The function encodes the query using the same SentenceTransformer model used for storage, retrieves
    the top-k most relevant documents, and returns both their content and metadata.

    Args:
        query (str): The user's input query for semantic search.
        k (int, optional): Number of top results to return. Defaults to 5.
        source (str, optional): The source document to filter by. Defaults to None.

    Returns:
        list or dict: List of dictionaries with 'content' and 'meta' keys for each matched document,
                      or an error dictionary in case of failure.
    """
    try:
        # Encode query to obtain its vector representation
        q_emb = model.encode(query).tolist()

        # Query the ChromaDB collection for top-k matches
        if source:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                where={"source": source}
            )
        else:
            results = collection.query(query_embeddings=[q_emb], n_results=k)

        documents = results["documents"][0]  # type: ignore # Extract matched texts
        metadatas = results["metadatas"][0]  # type: ignore # Extract corresponding metadata

        # Format and return the matched results
        return [
            {
                "content": doc,
                "meta": {
                    "source": meta.get("source", "unknown")  # Default to 'unknown' if source is missing
                }
            }
            for doc, meta in zip(documents, metadatas)
        ]

    except Exception as e:
        return {"status": "error", "message": f"Search failed: {str(e)}"}
