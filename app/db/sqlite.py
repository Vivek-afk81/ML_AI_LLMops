
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Database directory
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database path
DATABASE_URL = f"sqlite:///{DB_DIR / 'recallr.db'}"


# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


# Create database session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# Base class for SQLAlchemy models
Base = declarative_base()


def get_db():
    """
    Provide a database session to the caller.
    Close the session after use.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
