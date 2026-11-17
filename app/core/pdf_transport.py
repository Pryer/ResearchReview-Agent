"""Public PDF transport: validated DNS answers are pinned through connection setup."""
import ipaddress
import socket
from urllib.parse import urlsplit, urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.util.connection import create_connection
from app.core.exceptions import PDFDownloadError
from app.core.retry_policy import check_retrieval_execution


def public_target(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError("unsupported scheme or credentials")
        host = parsed.hostname.encode("idna").decode("ascii")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = [entry[4][0] for entry in answers]
        if not addresses:
            raise ValueError("empty DNS result")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global or ip.is_multicast or ip.is_reserved or (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None):
                raise ValueError("non-public address")
        return parsed.scheme, host, port, addresses[0]
    except (ValueError, UnicodeError, OSError) as exc:
        raise PDFDownloadError("PDF network target rejected") from exc


class PinnedAdapter(HTTPAdapter):
    def __init__(self, target):
        super().__init__(max_retries=0)
        scheme, host, port, address = target
        base = HTTPSConnection if scheme == "https" else HTTPConnection
        class Connection(base):
            def _new_conn(self):
                # WHY: 连接只使用已校验的 IP；原主机仍用于 Host、TLS SNI 与证书校验。
                from urllib3.exceptions import NewConnectionError
                try:
                    return create_connection((address, self.port), self.timeout, self.source_address, self.socket_options)
                except OSError as exc:
                    raise NewConnectionError(self, "PDF connection failed") from exc
        pool_base = HTTPSConnectionPool if scheme == "https" else HTTPConnectionPool
        class Pool(pool_base):
            ConnectionCls = Connection
        self.pool = Pool(host, port, maxsize=1, block=True)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        return self.pool

    def close(self):
        self.pool.close()
        super().close()


def request_public_pdf(url, *, timeout, headers, **ignored):
    """Validate every redirect; never inherit environment proxies or netrc credentials."""
    for hop in range(6):
        check_retrieval_execution()
        adapter = PinnedAdapter(public_target(url))
        session = requests.Session()
        session.trust_env = False
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        try:
            response = session.get(url, timeout=timeout, headers=headers, stream=True, allow_redirects=False)
        except BaseException:
            session.close()
            raise
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            session.close()
            if not location or hop == 5:
                raise PDFDownloadError("PDF redirect limit or missing location")
            url = urljoin(url, location)
            continue
        original_close = response.close
        def close():
            try:
                original_close()
            finally:
                session.close()
        response.close = close
        return response
