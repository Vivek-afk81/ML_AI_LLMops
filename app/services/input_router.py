class InputRouter:

    @staticmethod
    def route_input(
        original_text: str | None = None,
        source_url: str | None = None,
        personal_note: str | None = None,
    ) -> dict:
        """Identify the memory input type and prepare its data."""

        has_text = bool(original_text and original_text.strip())
        has_url = bool(source_url and source_url.strip())
        has_note = bool(personal_note and personal_note.strip())

        # URL + personal note
        if has_url and has_note:
            return {
                "input_type": "hybrid",
                "original_text": None,
                "source_url": source_url.strip(),
                "personal_note": personal_note.strip(),
            }

        # URL only
        if has_url and not has_text and not has_note:
            return {
                "input_type": "url",
                "original_text": None,
                "source_url": source_url.strip(),
                "personal_note": None,
            }

        # Plain text only
        if has_text and not has_url and not has_note:
            return {
                "input_type": "text",
                "original_text": original_text.strip(),
                "source_url": None,
                "personal_note": None,
            }

        raise ValueError(
            "Provide either text, a URL, or a URL with a personal note."
        )


# Shared router instance
input_router = InputRouter()
