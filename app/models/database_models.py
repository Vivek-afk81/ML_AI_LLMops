
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    JSON,
    Index,
)

from app.db.sqlite import Base


class Memory(Base):
    """
    SQLAlchemy model representing a saved memory.
    """

    __tablename__ = "memories"

    # Unique identifier
    id = Column(String(36), primary_key=True)

    # Input information
    input_type = Column(String(20), nullable=False)
    original_text = Column(Text, nullable=True)
    personal_note = Column(Text, nullable=True)

    # URL information
    source_url = Column(Text, nullable=True)
    title = Column(String(500), nullable=True)
    extracted_content = Column(Text, nullable=True)

    # Automatically predicted categories
    tags = Column(JSON, nullable=True)

    # Processing information
    processing_status = Column(
        String(30),
        nullable=False,
        default="pending",
        index=True,
    )

    embedding_status = Column(
        String(30),
        nullable=False,
        default="pending",
    )

    classification_status = Column(
        String(30),
        nullable=False,
        default="pending",
    )

    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


    # Index for common status-based queries
    __table_args__ = (
        Index("idx_memory_status_created", "processing_status", "created_at"),
    )
