
from sqlalchemy.orm import Session

from app.models.database_models import Memory
from app.services.embedding_service import embedding_service
from app.db.vector_store import search_memory_vectors


class RetrievalService:

    def search_memories(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50.")

        # 1. Convert search query into embedding
        query_embedding = embedding_service.generate_embedding(
            query.strip()
        )

        # 2. Search similar vectors in ChromaDB
        results = search_memory_vectors(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        if not ids:
            return []

        # 3. Fetch corresponding memories from SQLite
        memories = (
            db.query(Memory)
            .filter(Memory.id.in_(ids))
            .all()
        )

        memory_map = {memory.id: memory for memory in memories}

        # 4. Preserve ChromaDB ranking
        search_results = []

        for memory_id, distance in zip(ids, distances):

            memory = memory_map.get(memory_id)

            if memory is None:
                continue

            search_results.append({
                "memory": memory,
                "similarity_score": round(1.0 - distance, 4),
            })

        return search_results


retrieval_service = RetrievalService()
