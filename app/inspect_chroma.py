from chromadb import PersistentClient

# Connect to your Chroma DB (same path as in config.py)
client = PersistentClient(path="./chroma_db")

# List collections
collections = client.list_collections()
print("Available Collections:", collections)

# If you know your collection name (often "documents" or similar)
for col in collections:
    collection = client.get_collection(col.name)
    print(f"\nCollection: {col.name}")
    print("Total docs:", collection.count())

    # Fetch a few documents
    docs = collection.get(include=["metadatas", "documents"], limit=5)
    if docs["documents"] is not None:
        for i, doc in enumerate(docs["documents"]):
            print(f"Doc {i+1}: {doc}")
            if docs["metadatas"] is not None:
                print("Metadata:", docs["metadatas"][i])
            else:
                print("Metadata: None")
    else:
        print("No documents found in this collection.")
