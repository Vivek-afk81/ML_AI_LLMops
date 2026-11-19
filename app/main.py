
from fastapi import FastAPI

from app.db.sqlite import Base, engine
from app.models import database_models
from app.api.routes.memories import router as memories_router
from app.api.routes.search import router as search_router

# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="RecallR API",
    description="API-first personal memory vault with semantic search.",
    version="1.0.0",
)


app.include_router(
    memories_router,
    prefix="/api/v1",
)

app.include_router(
    search_router,
    prefix="/api/v1",
)


@app.get("/")
def home():
    return {
        "application": "RecallR",
        "message": "Personal Memory Vault API is running.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "RecallR",
    }
