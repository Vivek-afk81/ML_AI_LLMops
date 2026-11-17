import re


class TextProcessor:

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize input text."""

        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        # Remove null characters
        text = text.replace("\x00", "")

        # Normalize multiple spaces and line breaks
        text = re.sub(r"\s+", " ", text)

        # Remove leading and trailing spaces
        text = text.strip()

        if not text:
            raise ValueError("Text cannot be empty after cleaning.")

        return text

    @classmethod
    def prepare_memory_content(
        cls,
        original_text: str | None = None,
        personal_note: str | None = None,
        extracted_content: str | None = None,
    ) -> str:
        """Combine available memory content into clean text."""

        parts = []

        for content in [
            original_text,
            personal_note,
            extracted_content,
        ]:
            if content and content.strip():
                parts.append(cls.clean_text(content))

        if not parts:
            raise ValueError("No content available to process.")

        return "\n".join(parts)


# Shared processor instance
text_processor = TextProcessor()
