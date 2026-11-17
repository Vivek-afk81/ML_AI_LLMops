import pytest

from app.services.input_router import input_router


def test_routes_url_only():
    assert input_router.route_input(source_url=" https://example.com ") == {
        "input_type": "url",
        "original_text": None,
        "source_url": "https://example.com",
        "personal_note": None,
    }


def test_routes_plain_text_only():
    assert input_router.route_input(original_text=" Remember this ") == {
        "input_type": "text",
        "original_text": "Remember this",
        "source_url": None,
        "personal_note": None,
    }


def test_routes_url_with_personal_note():
    assert input_router.route_input(
        source_url=" https://example.com ",
        personal_note=" Read later ",
    ) == {
        "input_type": "hybrid",
        "original_text": None,
        "source_url": "https://example.com",
        "personal_note": "Read later",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"original_text": "text", "source_url": "https://example.com"},
        {"personal_note": "note"},
    ],
)
def test_rejects_invalid_input_combinations(kwargs):
    with pytest.raises(
        ValueError,
        match="Provide either text, a URL, or a URL with a personal note.",
    ):
        input_router.route_input(**kwargs)
