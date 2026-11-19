
from transformers import pipeline

from app.core.config import settings


class ClassificationService:

    def __init__(self):
        config = settings["classification_model"]

        self.model_id = config["model_id"]
        self.candidate_labels = config["candidate_labels"]
        self.threshold = config["threshold"]
        self.multi_label = config["multi_label"]

        # Load model only when classification is first requested.
        self._classifier = None

    def _load_model(self):
        """Load zero-shot classification model once."""

        if self._classifier is None:
            self._classifier = pipeline(
                task="zero-shot-classification",
                model=self.model_id,
                device=-1,  # CPU inference
            )

        return self._classifier

    def classify_text(self, text: str) -> list[str]:
        """Generate multiple relevant tags for a memory."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        if not self.candidate_labels:
            raise ValueError("Candidate labels cannot be empty.")

        classifier = self._load_model()

        result = classifier(
            text.strip(),
            candidate_labels=self.candidate_labels,
            multi_label=self.multi_label,
        )

        tags = [
            label
            for label, score in zip(
                result["labels"],
                result["scores"],
            )
            if score >= self.threshold
        ]

        return tags


# Shared service instance
classification_service = ClassificationService()
