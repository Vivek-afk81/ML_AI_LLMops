import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.sqlite import Base
from app.models.database_models import Memory


@pytest.fixture
def db_session():

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestingSession = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    session = TestingSession()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_create_memory(db_session):

    memory = Memory(
        input_type="text",
        original_text="Learn Python",
    )

    db_session.add(memory)
    db_session.commit()
    db_session.refresh(memory)

    assert memory.id is not None
    assert memory.original_text == "Learn Python"


def test_memory_default_statuses(db_session):

    memory = Memory(
        input_type="text",
        original_text="Learn Docker",
    )

    db_session.add(memory)
    db_session.commit()
    db_session.refresh(memory)

    assert memory.processing_status == "pending"
    assert memory.embedding_status == "pending"
    assert memory.classification_status == "pending"


def test_retrieve_memory(db_session):

    memory = Memory(
        input_type="text",
        original_text="Learn SQLAlchemy",
    )

    db_session.add(memory)
    db_session.commit()

    saved_memory = db_session.query(Memory).filter_by(
        original_text="Learn SQLAlchemy"
    ).first()

    assert saved_memory is not None
    assert saved_memory.original_text == "Learn SQLAlchemy"
