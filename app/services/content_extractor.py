import http.client
import ipaddress
import socket
import ssl
import certifi

from urllib.parse import urlparse, urlunparse, urljoin

import trafilatura

MAX_CONTENT_SIZE = 2 * 1024 * 1024  # 2 MB
REQUEST_TIMEOUT = 10
MAX_REDIRECTS = 5


def _validate_and_resolve(url: str):
    """Validate URL and return a verified public destination IP."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("URL must contain a valid hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL port.") from exc

    if port is not None and port not in {80, 443}:
        raise ValueError("Only ports 80 and 443 are allowed.")

    hostname = parsed.hostname.rstrip(".").lower()

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Localhost URLs are not allowed.")

    port = port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("Unable to resolve URL hostname.") from exc

    if not addresses:
        raise ValueError("Hostname did not resolve to an IP address.")

    public_ips = []

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%")[0])

        if not ip.is_global:
            raise ValueError(
                "URLs resolving to private or non-public IP addresses are not allowed."
            )

        public_ips.append(str(ip))

    normalized_url = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",
        )
    )

    return normalized_url, parsed, public_ips[0], port


def validate_url(url: str) -> str:
    """Validate a URL before fetching."""

    normalized_url, _, _, _ = _validate_and_resolve(url)
    return normalized_url


class PinnedHTTPConnection(http.client.HTTPConnection):
    """Connect directly to an already validated IP address."""

    def __init__(self, hostname, port, ip_address):
        super().__init__(hostname, port, timeout=REQUEST_TIMEOUT)
        self.ip_address = ip_address

    def connect(self):
        self.sock = socket.create_connection(
            (self.ip_address, self.port),
            timeout=self.timeout,
        )


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a validated IP while verifying the original hostname."""

    def __init__(self, hostname, port, ip_address):
        super().__init__(
            hostname,
            port,
            timeout=REQUEST_TIMEOUT,
            context=ssl.create_default_context(
                cafile=certifi.where()
            ),
        )
        self.ip_address = ip_address

    def connect(self):
        raw_socket = socket.create_connection(
            (self.ip_address, self.port),
            timeout=self.timeout,
        )

        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except Exception:
            raw_socket.close()
            raise


def _fetch_once(url: str, parsed, ip_address: str, port: int):
    """Fetch one URL without automatic redirects."""

    hostname = parsed.hostname

    if parsed.scheme.lower() == "https":
        connection = PinnedHTTPSConnection(
            hostname, port, ip_address
        )
    else:
        connection = PinnedHTTPConnection(
            hostname, port, ip_address
        )

    path = parsed.path or "/"

    if parsed.params:
        path += ";" + parsed.params

    if parsed.query:
        path += "?" + parsed.query

    host_header = hostname

    if parsed.port:
        host_header = f"{hostname}:{parsed.port}"

    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": host_header,
                "User-Agent": "RecallR/1.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
        )

        response = connection.getresponse()

        return response, connection

    except Exception:
        connection.close()
        raise


def fetch_url(url: str) -> dict:
    """Safely fetch a public webpage with redirect and size limits."""

    current_url = url

    for redirect_count in range(MAX_REDIRECTS + 1):

        # Revalidate and resolve every redirect destination.
        normalized_url, parsed, ip_address, port = (
            _validate_and_resolve(current_url)
        )

        try:
            response, connection = _fetch_once(
                normalized_url,
                parsed,
                ip_address,
                port,
            )
        except (OSError, TimeoutError, ssl.SSLError, http.client.HTTPException) as exc:
            raise RuntimeError(
                "Unable to fetch webpage due to a network error."
            ) from exc



        try:
            if response.status in {301, 302, 303, 307, 308}:

                if redirect_count >= MAX_REDIRECTS:
                    raise ValueError("Too many redirects.")

                location = response.getheader("Location")

                if not location:
                    raise ValueError("Redirect location is missing.")

                current_url = urljoin(normalized_url, location)
                continue

            if response.status < 200 or response.status >= 300:
                raise ValueError(
                    f"Website returned HTTP {response.status}."
                )

            content_type = (
                response.getheader("Content-Type") or ""
            ).lower().split(";")[0].strip()

            if content_type not in {
                "text/html",
                "application/xhtml+xml",
            }:
                raise ValueError("Unsupported webpage content type.")

            content_length = response.getheader("Content-Length")

            if content_length:
                try:
                    if int(content_length) > MAX_CONTENT_SIZE:
                        raise ValueError("Webpage exceeds the 2 MB limit.")
                except ValueError as exc:
                    if str(exc) == "Webpage exceeds the 2 MB limit.":
                        raise
                    raise ValueError("Invalid Content-Length header.") from exc

            body = response.read(MAX_CONTENT_SIZE + 1)

            if len(body) > MAX_CONTENT_SIZE:
                raise ValueError("Webpage exceeds the 2 MB limit.")

            charset = response.headers.get_content_charset() or "utf-8"

            try:
                content = body.decode(charset, errors="replace")
            except LookupError:
                content = body.decode("utf-8", errors="replace")

            return {
                "url": normalized_url,
                "status_code": response.status,
                "content_type": content_type,
                "html": content,
            }

        finally:
            connection.close()

    raise ValueError("Unable to fetch webpage.")


def extract_content(html: str, url: str | None = None) -> dict:
    """Extract meaningful text and title from webpage HTML."""

    if not isinstance(html, str) or not html.strip():
        raise ValueError("HTML content cannot be empty.")

    try:
        text = trafilatura.extract(
            html,
            url=url,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )

        if not text or not text.strip():
            raise ValueError(
                "Unable to extract meaningful webpage content."
            )

        metadata = trafilatura.extract_metadata(
            html,
            default_url=url,
        )

        title = metadata.title if metadata and metadata.title else None

        return {
            "title": title,
            "text": text.strip(),
        }

    except ValueError:
        raise

    except Exception as exc:
        raise RuntimeError(
            "Webpage content extraction failed."
        ) from exc
