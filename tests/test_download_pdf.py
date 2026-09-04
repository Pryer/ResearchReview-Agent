"""PDF 下载器的重试、真实性校验与批量隔离测试。"""

from __future__ import annotations

from pathlib import Path

import fitz
import requests

from app.tools import download_pdf


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status_code: int = 200,
        content_type: str = "application/pdf",
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"status {self.status_code}",
                response=self,
            )

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index:index + chunk_size]

    def close(self) -> None:
        self.closed = True


def _valid_pdf_bytes() -> bytes:
    document = fitz.open()
    for page_number in range(2):
        page = document.new_page()
        page.insert_text((72, 72), f"ResearchReview-Agent PDF test page {page_number + 1}")
    data = document.tobytes()
    document.close()
    # PDF 允许 %%EOF 后存在额外字节；补齐最小文件阈值。
    return data + b"\n%" + (b"x" * 2048)


def _configure_fast_test(monkeypatch) -> None:
    config = download_pdf.get_settings()
    monkeypatch.setattr(config, "pdf_download_retries", 2)
    monkeypatch.setattr(config, "pdf_download_backoff_seconds", 0.0)
    monkeypatch.setattr(config, "pdf_download_max_mb", 5)
    monkeypatch.setattr(config, "pdf_download_connect_timeout", 1)
    monkeypatch.setattr(config, "agent_request_timeout", 1)


def test_download_retries_retryable_http_status_and_saves_atomically(
    monkeypatch,
    tmp_path: Path,
):
    _configure_fast_test(monkeypatch)
    responses = [
        FakeResponse(b"busy", status_code=503, content_type="text/plain"),
        FakeResponse(_valid_pdf_bytes()),
    ]
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(download_pdf.requests, "get", fake_get)
    path = download_pdf.download_open_access_pdf(
        {
            "paper_id": "openalex:W1",
            "pdf_url": "https://example.org/paper.pdf",
            "is_open_access": True,
        },
        str(tmp_path),
    )

    assert len(calls) == 2
    assert path is not None
    assert download_pdf.validate_pdf_file(path) is True
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_html_disguised_as_pdf(monkeypatch, tmp_path: Path):
    _configure_fast_test(monkeypatch)
    html = b"<html><body>login required</body></html>" + (b"x" * 2048)
    monkeypatch.setattr(
        download_pdf.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(html, content_type="text/html"),
    )

    path = download_pdf.download_open_access_pdf(
        {
            "paper_id": "openalex:login",
            "pdf_url": "https://example.org/download",
            "is_open_access": True,
        },
        str(tmp_path),
    )

    assert path is None
    assert not list(tmp_path.iterdir())


def test_cnki_pdf_is_skipped_before_network_request(monkeypatch, tmp_path: Path):
    _configure_fast_test(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("CNKI PDF must not trigger an HTTP request")

    monkeypatch.setattr(download_pdf.requests, "get", fail_if_called)
    paper = {
        "paper_id": "cnki:paper-1",
        "source": "CNKI",
        "pdf_url": "https://kns.cnki.net/download/example.pdf",
        "is_open_access": True,
    }

    assert download_pdf.is_cnki_paper(paper) is True
    assert download_pdf.allows_pdf_download(paper) is False
    assert download_pdf.is_open_access(paper) is False
    assert download_pdf.download_open_access_pdf(paper, str(tmp_path)) is None
    assert not list(tmp_path.iterdir())


def test_batch_ignores_existing_cnki_pdf_but_keeps_other_sources(
    monkeypatch,
    tmp_path: Path,
):
    cached_cnki = tmp_path / "cnki.pdf"
    cached_cnki.write_bytes(_valid_pdf_bytes())
    cached_other = tmp_path / "other.pdf"
    cached_other.write_bytes(_valid_pdf_bytes())

    result = download_pdf.batch_download_pdfs(
        [
            {
                "paper_id": "cnki:paper-2",
                "source": "cnki",
                "pdf_url": "https://kns.cnki.net/example.pdf",
            },
            {
                "paper_id": "openalex:W2",
                "source": "openalex",
                "pdf_url": "https://example.org/open.pdf",
            },
        ],
        existing={
            "cnki:paper-2": str(cached_cnki),
            "openalex:W2": str(cached_other),
        },
    )

    assert result["cnki:paper-2"] is None
    assert result["openalex:W2"] == str(cached_other)
    assert cached_cnki.exists()  # 忽略但不删除用户已有文件


def test_invalid_cached_pdf_is_replaced(monkeypatch, tmp_path: Path):
    _configure_fast_test(monkeypatch)
    paper = {"paper_id": "arxiv:1234.5678", "pdf_url": "https://arxiv.org/pdf/1234.5678"}
    cached = download_pdf._pdf_path_for(paper, str(tmp_path))
    cached.write_bytes(b"not a pdf" + (b"x" * 2048))
    monkeypatch.setattr(
        download_pdf.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(_valid_pdf_bytes()),
    )

    path = download_pdf.download_open_access_pdf(paper, str(tmp_path))

    assert path == str(cached)
    assert download_pdf.validate_pdf_file(path) is True


def test_pdf_url_without_open_access_permission_is_rejected(monkeypatch, tmp_path: Path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("unverified PDF URL must not trigger a network request")

    monkeypatch.setattr(download_pdf.requests, "get", fail_if_called)
    paper = {
        "paper_id": "crossref:closed",
        "source": "crossref",
        "pdf_url": "https://publisher.example/article.pdf",
        "is_open_access": False,
    }

    assert download_pdf.is_open_access(paper) is False
    assert download_pdf.download_open_access_pdf(paper, str(tmp_path)) is None


def test_batch_download_isolates_single_paper_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(download_pdf.get_settings(), "pdf_download_max_workers", 2)
    good_path = tmp_path / "good.pdf"
    good_path.write_bytes(_valid_pdf_bytes())

    def fake_download(paper, save_dir=None):
        if paper["paper_id"] == "p1":
            raise RuntimeError("single paper failed")
        return str(good_path)

    monkeypatch.setattr(download_pdf, "download_open_access_pdf", fake_download)
    result = download_pdf.batch_download_pdfs([
        {"paper_id": "p1", "pdf_url": "https://example.org/1.pdf", "is_open_access": True},
        {"paper_id": "p2", "pdf_url": "https://example.org/2.pdf", "is_open_access": True},
        {"paper_id": "p3", "pdf_url": None},
    ])

    assert result["p1"] is None
    assert result["p2"] == str(good_path)
    assert result["p3"] is None
