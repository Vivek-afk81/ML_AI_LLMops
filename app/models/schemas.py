
from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


# ==========================================
# Memory Creation Request
# ==========================================

class MemoryCreate(BaseModel):

    input_type: Literal["text", "url", "hybrid"]

    original_text: str | None = None
    personal_note: str | None = None
    source_url: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_input_type(self):

        if self.input_type == "text":
            if not self.original_text or not self.original_text.strip():
                raise ValueError(
                    "original_text is required for text memories"
                )

            if self.source_url is not None:
                raise ValueError(
                    "source_url is not allowed for text memories"
                )

        elif self.input_type == "url":
            if self.source_url is None:
                raise ValueError(
                    "source_url is required for URL memories"
                )

            if self.personal_note is not None:
                raise ValueError(
                    "Use input_type='hybrid' when providing a personal note"
                )

        elif self.input_type == "hybrid":
            if self.source_url is None:
                raise ValueError(
                    "source_url is required for hybrid memories"
                )

            if not self.personal_note or not self.personal_note.strip():
                raise ValueError(
                    "personal_note is required for hybrid memories"
                )

        return self


# ==========================================
# Memory Response
# ==========================================

class MemoryResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: str
    input_type: str

    original_text: str | None = None
    personal_note: str | None = None

    source_url: str | None = None
    title: str | None = None
    extracted_content: str | None = None

    tags: list[str] | None = None

    processing_status: str
    embedding_status: str
    classification_status: str

    error_message: str | None = None

    created_at: datetime
    updated_at: datetime


# ==========================================
# Semantic Search Request
# ==========================================

class SearchRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )



# Search Result


class SearchResult(BaseModel):

    memory: MemoryResponse
    similarity_score: float | None = None



# Processing Status Response


class JobStatusResponse(BaseModel):

    id: str
    processing_status: str

    embedding_status: str
    classification_status: str

    error_message: str | None = None
