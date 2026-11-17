import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:

    def __init__(self):
        model_config = settings["embedding_model"]

        self.model_id = model_config["model_id"]
        self.device = settings["inference"]["device"]
        self.normalize = model_config["normalize_embeddings"]
        self.embedding_dimension = model_config["embedding_dimension"]

        # Model will load only when first needed
        self._model = None

    def _load_model(self):
        """Load the embedding model once."""

        if self._model is None:
            self._model = SentenceTransformer(
                self.model_id,
                device=self.device,
            )

        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """Convert one text into an embedding vector."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        model = self._load_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embedding.tolist()

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""

        if not texts:
            raise ValueError("Text list cannot be empty.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        model = self._load_model()

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embeddings.tolist()


# Shared service instance
embedding_service = EmbeddingService()
