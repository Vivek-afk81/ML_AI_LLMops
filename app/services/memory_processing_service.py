
from sqlalchemy.orm import Session

from app.models.database_models import Memory

from app.services.content_extractor import fetch_url, extract_content
from app.services.text_processor import text_processor
from app.services.embedding_service import embedding_service
from app.services.classification_service import classification_service

from app.db.vector_store import (
    add_memory_vector,
    delete_memory_vector,
)


class MemoryProcessingService:

    def process_memory(self, db: Session, memory: Memory) -> Memory:
        """Process a memory and store its embedding and tags."""

        vector_added = False

        try:
            # 1. Mark memory as processing
            memory.processing_status = "processing"
            db.commit()

            # 2. Prepare content according to input type
            if memory.input_type == "text":

                content = text_processor.prepare_memory_content(
                    original_text=memory.original_text
                )

            elif memory.input_type in {"url", "hybrid"}:

                fetched = fetch_url(memory.source_url)

                extracted = extract_content(
                    fetched["html"],
                    url=fetched["url"],
                )

                memory.title = extracted["title"]
                memory.extracted_content = extracted["text"]

                content = text_processor.prepare_memory_content(
                    personal_note=memory.personal_note,
                    extracted_content=memory.extracted_content,
                )

            else:
                raise ValueError("Unsupported memory input type.")

            # 3. Generate embedding
            memory.embedding_status = "processing"
            db.commit()

            embedding = embedding_service.generate_embedding(content)

            # 4. Generate automatic tags
            memory.classification_status = "processing"
            db.commit()

            tags = classification_service.classify_text(content)

            # 5. Prepare ChromaDB metadata
            metadata = {
                "input_type": memory.input_type,
            }

            if memory.source_url:
                metadata["source_url"] = memory.source_url

            if memory.title:
                metadata["title"] = memory.title

            # 6. Store vector in ChromaDB
            add_memory_vector(
                memory_id=memory.id,
                embedding=embedding,
                document=content,
                metadata=metadata,
            )

            vector_added = True

            # 7. Update SQLite record
            memory.tags = tags
            memory.embedding_status = "completed"
            memory.classification_status = "completed"
            memory.processing_status = "completed"
            memory.error_message = None

            db.commit()
            db.refresh(memory)

            return memory

        except Exception as exc:

            db.rollback()

            # Remove vector if a later database operation fails
            if vector_added:
                try:
                    delete_memory_vector(memory.id)
                except Exception:
                    pass

            memory.processing_status = "failed"

            if memory.embedding_status == "processing":
                memory.embedding_status = "failed"

            if memory.classification_status == "processing":
                memory.classification_status = "failed"

            memory.error_message = str(exc)[:2000]

            db.add(memory)
            db.commit()
            db.refresh(memory)

            return memory


# Shared service instance
memory_processing_service = MemoryProcessingService()
