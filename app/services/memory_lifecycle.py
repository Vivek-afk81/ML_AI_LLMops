
from sqlalchemy.orm import Session

from app.models.database_models import Memory


def update_memory_status(
    db: Session,
    memory: Memory,
    processing_status: str | None = None,
    embedding_status: str | None = None,
    classification_status: str | None = None,
    error_message: str | None = None,
) -> Memory:
    """Update the processing state of a memory."""

    valid_statuses = {"pending", "processing", "completed", "failed"}

    updates = {
        "processing_status": processing_status,
        "embedding_status": embedding_status,
        "classification_status": classification_status,
    }

    for field, value in updates.items():
        if value is not None:
            if value not in valid_statuses:
                raise ValueError(f"Invalid {field}: {value}")

            setattr(memory, field, value)

    if error_message is not None:
        memory.error_message = error_message

    db.add(memory)
    db.commit()
    db.refresh(memory)

    return memory
