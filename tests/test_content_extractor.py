import socket
import pytest

from app.services.content_extractor import validate_url


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
