import uuid

from app.db.vector_store import (
    collection,
    add_memory_vector,
    search_memory_vectors,
    delete_memory_vector,
)


def test_add_memory_vector():
    memory_id = str(uuid.uuid4())

    try:
        add_memory_vector(
            memory_id=memory_id,
            embedding=[0.1, 0.2, 0.3],
            document="Learn Python programming",
            metadata={"category": "Programming"},
        )

        result = collection.get(ids=[memory_id])

        assert memory_id in result["ids"]
        assert result["documents"][0] == "Learn Python programming"

    finally:
        delete_memory_vector(memory_id)


def test_search_memory_vectors():
    memory_id = str(uuid.uuid4())

    try:
        add_memory_vector(
            memory_id=memory_id,
            embedding=[0.1, 0.2, 0.3],
            document="Learn Python programming",
            metadata={"category": "Programming"},
        )

        result = search_memory_vectors(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=1,
        )

        assert memory_id in result["ids"][0]
        assert result["documents"][0][0] == "Learn Python programming"

    finally:
        delete_memory_vector(memory_id)


def test_delete_memory_vector():
    memory_id = str(uuid.uuid4())

    add_memory_vector(
        memory_id=memory_id,
        embedding=[0.1, 0.2, 0.3],
        document="Temporary memory",
        metadata={"category": "Testing"},
    )

    delete_memory_vector(memory_id)

    result = collection.get(ids=[memory_id])

    assert memory_id not in result["ids"]
