
from pathlib import Path

import chromadb


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Persistent ChromaDB storage
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Create or load RecallR collection
collection = client.get_or_create_collection(
    name="recallr_memories",
    metadata={"hnsw:space": "cosine"},
)


def add_memory_vector(
    memory_id: str,
    embedding: list[float],
    document: str,
    metadata: dict,
):
    """Store a memory's embedding in ChromaDB."""

    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata],
    )


def search_memory_vectors(
    query_embedding: list[float],
    top_k: int = 5,
):
    """Find memories with embeddings similar to the query."""

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )


def delete_memory_vector(memory_id: str):
    """Delete a memory's vector from ChromaDB."""

    collection.delete(ids=[memory_id])
