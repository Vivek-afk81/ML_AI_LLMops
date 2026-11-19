
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.sqlite import get_db, SessionLocal
from app.models.database_models import Memory
from app.models.schemas import MemoryCreate, MemoryResponse
from app.services.memory_processing_service import memory_processing_service


router = APIRouter(
    prefix="/memories",
    tags=["Memories"],
)


def process_memory_in_background(memory_id: str):
    """Process memory using an independent database session."""

    db = SessionLocal()

    try:
        memory = db.query(Memory).filter(
            Memory.id == memory_id
        ).first()

        if memory is None:
            return

        memory_processing_service.process_memory(db, memory)

    finally:
        db.close()


@router.post(
    "/",
    response_model=MemoryResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_memory(
    payload: MemoryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a memory and schedule background processing."""

    memory = Memory(
        input_type=payload.input_type,
        original_text=payload.original_text,
        personal_note=payload.personal_note,
        source_url=str(payload.source_url) if payload.source_url else None,
        processing_status="pending",
        embedding_status="pending",
        classification_status="pending",
    )

    try:
        db.add(memory)
        db.commit()
        db.refresh(memory)

        background_tasks.add_task(
            process_memory_in_background,
            memory.id,
        )

        return memory

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to create memory.",
        ) from exc


@router.get("/{memory_id}", response_model=MemoryResponse)
def get_memory(
    memory_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve a memory and its processing status."""

    memory = db.query(Memory).filter(
        Memory.id == memory_id
    ).first()

    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="Memory not found.",
        )

    return memory
