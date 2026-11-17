import pytest

from app.services.text_processor import text_processor


def test_clean_text_removes_extra_spaces():
    result = text_processor.clean_text(
        "  Learn    Python   "
    )

    assert result == "Learn Python"


def test_clean_text_normalizes_line_breaks():
    result = text_processor.clean_text(
        "Learn Python\n\nand\tMachine Learning"
    )

    assert result == "Learn Python and Machine Learning"


def test_clean_text_removes_null_characters():
    result = text_processor.clean_text(
        "Learn\x00Python"
    )

    assert result == "LearnPython"


def test_clean_text_rejects_empty_string():
    with pytest.raises(ValueError):
        text_processor.clean_text("   ")


def test_clean_text_rejects_non_string():
    with pytest.raises(ValueError):
        text_processor.clean_text(123)


def test_prepare_memory_content_combines_sources():
    result = text_processor.prepare_memory_content(
        original_text="Learn Python",
        personal_note="Important resource",
        extracted_content="Python documentation",
    )

    assert result == (
        "Learn Python\n"
        "Important resource\n"
        "Python documentation"
    )


def test_prepare_memory_content_rejects_empty_input():
    with pytest.raises(ValueError):
        text_processor.prepare_memory_content()
