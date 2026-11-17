import socket
import pytest

from app.services.content_extractor import validate_url
from app.services.content_extractor import extract_content

from email.message import Message
from urllib.parse import urlparse

import app.services.content_extractor as extractor

def mock_dns(monkeypatch, ip="8.8.8.8"):
    """Return a controlled DNS response."""

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
        ],
    )


def test_valid_https_url(monkeypatch):
    mock_dns(monkeypatch)

    result = validate_url("https://example.com")

    assert result == "https://example.com/"


def test_reject_localhost():
    with pytest.raises(ValueError, match="Localhost"):
        validate_url("http://localhost")


def test_reject_private_ip(monkeypatch):
    mock_dns(monkeypatch, "192.168.1.10")

    with pytest.raises(ValueError, match="non-public IP"):
        validate_url("https://example.com")


def test_reject_metadata_ip(monkeypatch):
    mock_dns(monkeypatch, "169.254.169.254")

    with pytest.raises(ValueError, match="non-public IP"):
        validate_url("https://example.com")


def test_reject_unsupported_scheme():
    with pytest.raises(ValueError, match="HTTP and HTTPS"):
        validate_url("ftp://example.com")


def test_reject_credentials():
    with pytest.raises(ValueError, match="credentials"):
        validate_url("https://user:password@example.com")


def test_reject_invalid_port():
    with pytest.raises(ValueError, match="ports 80 and 443"):
        validate_url("https://example.com:8080")


    def __init__(self, status=200, body=b"<html>Hello</html>", headers=None):
        self.status = status
        self.body = body
        self.headers = Message()

        self.headers["Content-Type"] = "text/html; charset=utf-8"

        if headers:
            for key, value in headers.items():
                self.headers[key] = value

        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size):
        return self.body[:size]

class FakeResponse:

    def __init__(
        self,
        status=200,
        body=b"<html>Hello</html>",
        headers=None
    ):
        self.status = status
        self.body = body
        self.headers = Message()

        response_headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(body)),
        }

        if headers:
            response_headers.update(headers)

        for key, value in response_headers.items():
            self.headers[key] = value

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size):
        return self.body[:size]

class FakeConnection:
    def close(self):
        pass


def mock_destination(monkeypatch):
    def fake_validate(url):
        parsed = urlparse(url)
        port = parsed.port or 443

        return url, parsed, "8.8.8.8", port

    monkeypatch.setattr(
        extractor,
        "_validate_and_resolve",
        fake_validate,
    )


def test_fetch_valid_html(monkeypatch):
    mock_destination(monkeypatch)

    monkeypatch.setattr(
        extractor,
        "_fetch_once",
        lambda *args: (FakeResponse(), FakeConnection()),
    )

    result = extractor.fetch_url("https://example.com")

    assert result["status_code"] == 200
    assert result["content_type"] == "text/html"
    assert "Hello" in result["html"]


def test_redirect_destination_is_revalidated(monkeypatch):
    mock_destination(monkeypatch)

    responses = [
        FakeResponse(
            status=302,
            headers={"Location": "https://example.org/article"},
        ),
        FakeResponse(),
    ]

    calls = []

    def fake_fetch(url, parsed, ip, port):
        calls.append(url)
        return responses.pop(0), FakeConnection()

    monkeypatch.setattr(extractor, "_fetch_once", fake_fetch)

    result = extractor.fetch_url("https://example.com")

    assert len(calls) == 2
    assert calls[1] == "https://example.org/article"
    assert result["status_code"] == 200


def test_reject_non_html_content(monkeypatch):
    mock_destination(monkeypatch)

    monkeypatch.setattr(
        extractor,
        "_fetch_once",
        lambda *args: (
            FakeResponse(
                headers={"Content-Type": "application/pdf"}
            ),
            FakeConnection(),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported webpage"):
        extractor.fetch_url("https://example.com")


def test_reject_oversized_content(monkeypatch):
    mock_destination(monkeypatch)
    monkeypatch.setattr(extractor, "MAX_CONTENT_SIZE", 5)

    monkeypatch.setattr(
        extractor,
        "_fetch_once",
        lambda *args: (
            FakeResponse(
                body=b"123456",
                headers={"Content-Length": "6"},
            ),
            FakeConnection(),
        ),
    )

    with pytest.raises(ValueError, match="2 MB limit"):
        extractor.fetch_url("https://example.com")

SAMPLE_HTML = """
<html>
<head>
    <title>Understanding Machine Learning</title>
</head>
<body>
    <nav>Home About Contact</nav>
    <main>
        <article>
            <h1>Understanding Machine Learning</h1>
            <p>Machine learning enables computers to learn from data.</p>
            <p>It is widely used in prediction and classification tasks.</p>
        </article>
    </main>
    <footer>Copyright 2026</footer>
</body>
</html>
"""


def test_extract_content():
    result = extract_content(SAMPLE_HTML)

    assert result["title"] == "Understanding Machine Learning"
    assert "Machine learning enables computers" in result["text"]
    assert "prediction and classification" in result["text"]


def test_extract_content_with_url():
    result = extract_content(
        SAMPLE_HTML,
        url="https://example.com/article",
    )

    assert result["title"] == "Understanding Machine Learning"
    assert result["text"]


def test_extract_empty_html():
    import pytest

    with pytest.raises(ValueError, match="HTML content cannot be empty"):
        extract_content("")


def test_extract_content_without_article():
    import pytest

    html = "<html><head><title>Empty</title></head><body></body></html>"

    with pytest.raises(ValueError, match="Unable to extract meaningful webpage content"):
        extract_content(html)