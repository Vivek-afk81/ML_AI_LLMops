
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.sqlite import get_db
from app.models.schemas import SearchRequest, SearchResult
from app.services.retrieval_service import retrieval_service


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post("/", response_model=list[SearchResult])
def search_memories(
    payload: SearchRequest,
    db: Session = Depends(get_db),
):
    """Find relevant memories using semantic similarity."""

    try:
        return retrieval_service.search_memories(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Memory search failed.",
        ) from exc
