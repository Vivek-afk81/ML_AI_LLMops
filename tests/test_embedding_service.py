
import pytest

from app.services.embedding_service import embedding_service


def test_generate_embedding_returns_384_dimensions():
    result = embedding_service.generate_embedding(
        "I want to learn Machine Learning"
    )

    assert isinstance(result, list)
    assert len(result) == 384


def test_generate_embedding_returns_float_values():
    result = embedding_service.generate_embedding(
        "Learn Python programming"
    )

    assert all(isinstance(value, float) for value in result)


def test_generate_embedding_rejects_empty_text():
    with pytest.raises(ValueError, match="Text cannot be empty"):
        embedding_service.generate_embedding("")


def test_generate_embedding_rejects_whitespace():
    with pytest.raises(ValueError, match="Text cannot be empty"):
        embedding_service.generate_embedding("   ")


def test_generate_embeddings_returns_correct_count():
    texts = [
        "Learn Python",
        "Learn Docker",
        "Learn Machine Learning",
    ]

    results = embedding_service.generate_embeddings(texts)

    assert len(results) == 3
    assert all(len(vector) == 384 for vector in results)


def test_generate_embeddings_rejects_empty_list():
    with pytest.raises(ValueError, match="Text list cannot be empty"):
        embedding_service.generate_embeddings([])
