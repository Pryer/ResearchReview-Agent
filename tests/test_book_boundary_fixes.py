"""下载、核验身份和报告出口的边界回归；网络全部使用替身。"""
import json
import re
import socket
import threading

import pytest

from app.core import pdf_transport
from app.core.exceptions import PDFDownloadError
from app.tools import download_pdf


def dns(address):
    return [(socket.AF_INET6 if ':' in address else socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))]


@pytest.mark.parametrize('address', ['127.0.0.1', '10.1.2.3', '169.254.169.254', '::1', '::ffff:127.0.0.1', 'fc00::1', '224.0.0.1', 'ff02::1'])
def test_private_dns_rejected(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: dns(address))
    with pytest.raises(PDFDownloadError):
        pdf_transport.public_target('https://papers.example/p.pdf')


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'ftp://example.org/a', 'https://user:password@example.org/a'])
def test_scheme_and_credentials_rejected(url):
    with pytest.raises(PDFDownloadError):
        pdf_transport.public_target(url)


def test_oa_host_is_exact():
    assert not download_pdf.is_open_access({'pdf_url': 'https://arxiv.org.evil.invalid/a'})
    assert not download_pdf.is_open_access({'pdf_url': 'https://evil.invalid/?arxiv.org'})
    assert download_pdf.is_open_access({'pdf_url': 'https://arxiv.org/pdf/a'})


def test_redirect_private_target_never_connected(monkeypatch):
    calls = []
    monkeypatch.setattr(socket, 'getaddrinfo', lambda host, *a, **k: dns('127.0.0.1' if host == 'private.example' else '93.184.216.34'))
    class Response:
        status_code = 302
        headers = {'Location': 'http://private.example/p.pdf'}
        def close(self): pass
    def get(self, url, **kwargs):
        assert kwargs['allow_redirects'] is False
        assert self.trust_env is False
        calls.append(url)
        return Response()
    monkeypatch.setattr(pdf_transport.requests.Session, 'get', get)
    with pytest.raises(PDFDownloadError):
        pdf_transport.request_public_pdf('https://public.example/a', timeout=(1, 1), headers={})
    assert calls == ['https://public.example/a']


def test_connection_pins_validated_ip_preserves_tls_host(monkeypatch):
    connected = []
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: dns('93.184.216.34'))
    adapter = pdf_transport.PinnedAdapter(pdf_transport.public_target('https://papers.example/a'))
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: pytest.fail('must not resolve again'))
    monkeypatch.setattr(pdf_transport, 'create_connection', lambda address, *a: connected.append(address) or object())
    connection = adapter.pool._new_conn()
    adapter.cert_verify(adapter.pool, "https://papers.example/a", True, None)
    assert adapter.pool.cert_reqs == "CERT_REQUIRED"
    assert connection.host == "papers.example"
    connection._new_conn()
    assert connected == [('93.184.216.34', 443)]
    assert connection.host == 'papers.example'
    adapter.close()


def test_cancel_mid_stream_cleans_partial_file(monkeypatch, tmp_path):
    from app.core.retry_policy import retrieval_cancellation
    stop = threading.Event()
    class Response:
        status_code = 200
        headers = {}
        closed = False
        def raise_for_status(self): pass
        def iter_content(self, **kwargs):
            yield b'%PDF-'
            stop.set()
            yield b'incomplete'
        def close(self): self.closed = True
    response = Response()
    monkeypatch.setattr(download_pdf, 'request_public_pdf', lambda *a, **k: response)
    with retrieval_cancellation(stop.is_set), pytest.raises(InterruptedError):
        download_pdf.download_open_access_pdf({'paper_id': 'p1', 'is_open_access': True, 'pdf_url': 'https://example.org/a'}, str(tmp_path))
    assert response.closed
    assert not list(tmp_path.iterdir())


def test_control_exception_from_download_is_not_isolated(monkeypatch):
    from app.agent.execution_budget import AgentExecutionStale
    error = AgentExecutionStale('lost owner')
    def fail(*a, **k): raise error
    monkeypatch.setattr(download_pdf, 'download_open_access_pdf', fail)
    with pytest.raises(AgentExecutionStale) as caught:
        download_pdf.batch_download_pdfs([{'paper_id': 'p1', 'is_open_access': True, 'pdf_url': 'https://example.org/a'}])
    assert caught.value is error


def test_report_redacts_strings_without_changing_research_text():
    from app.services.execution_report_collector import safe
    text = 'DOI 10.1234/abc 94.2% 2024-2025'
    value = {'api_key': 'secret', 'error': 'Authorization: Bearer synthetic-token',
             'nested': ['api_key=synthetic-secret', 'https://user:password@example.org/a?token=private'], 'paper': text, 'serialized': '"api_key": "synthetic-json-secret"'}
    result = safe(value)
    encoded = json.dumps(result)
    assert 'synthetic-json-secret' not in encoded
    assert 'synthetic-token' not in encoded and 'synthetic-secret' not in encoded
    assert 'password' not in encoded and 'private' not in encoded
    assert result['paper'] == text
    assert value['api_key'] == 'secret'


def test_check_identity_changes_only_with_verifier_inputs(monkeypatch):
    from app.agent.checking import check_input_fingerprint
    from app.core.config import get_settings
    import app.prompt.claim_verification as prompt
    state = {'review': 'a'}
    first = check_input_fingerprint(state)
    assert check_input_fingerprint({**state, 'steps': ['log']}) == first
    monkeypatch.setattr(get_settings(), 'llm_model', 'another-model')
    second = check_input_fingerprint(state)
    assert second != first
    monkeypatch.setattr(prompt, 'CLAIM_ENTAILMENT_PROMPT', prompt.CLAIM_ENTAILMENT_PROMPT + '\nnew rule')
    assert check_input_fingerprint(state) != second


def test_semantic_and_local_report_reuse_respects_model_identity():
    from app.tools.verify_claims import verify_review_claims
    from tests.test_verify_claims import _card
    from app.services.verifier_profile import set_verifier_receipt
    class LLM:
        model = 'model-a'
        calls = 0
        def complete(self, prompt, **kwargs):
            self.calls += 1
            set_verifier_receipt({'provider': 'backup', 'response_model': self.model})
            items = json.loads(re.search(r'待验证项目：(\[.*?\])\n', prompt, re.S).group(1))
            return json.dumps({'results': [{'claim_id': i['claim_id'], 'label': 'entailed', 'confidence': 0.95} for i in items]})
    llm = LLM()
    cache = {}
    text = '该方法在 FEVER 数据集上达到 80% accuracy [p1]。'
    first = verify_review_claims(text, [_card()], llm=llm, entailment_cache=cache)
    verify_review_claims(text, [_card()], llm=llm, entailment_cache=cache)
    assert llm.calls == 1
    assert first['verifier_executions'][0]['provider'] == 'backup'
    llm.model = 'model-b'
    second = verify_review_claims(text, [_card()], llm=llm, entailment_cache=cache,
                                  verification_scope={'mode': 'local', 'previous_report': first})
    assert llm.calls == 2
    assert first['verifier_profile'] != second['verifier_profile']


def test_raw_read1_allows_cancellation_before_next_block(monkeypatch):
    from app.core.retry_policy import retrieval_cancellation
    stopped = threading.Event()
    class Raw:
        calls = 0
        def read1(self, size, **kwargs):
            self.calls += 1
            assert size == 64 * 1024
            stopped.set()
            return b"tiny"
    class Response:
        raw = Raw()
    with retrieval_cancellation(stopped.is_set):
        chunks = download_pdf._response_chunks(Response())
        assert next(chunks) == b"tiny"
        with pytest.raises(InterruptedError):
            next(chunks)
    assert Response.raw.calls == 1


def test_real_llm_adapter_records_successful_provider_receipt(monkeypatch):
    from types import SimpleNamespace
    from app.services.llm_service import LLMService
    from app.services.verifier_profile import verifier_receipt, set_verifier_receipt
    service = LLMService()
    response = SimpleNamespace(usage=None, model="actual-backup", id="synthetic-response",
                               choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    def call(kwargs, response_format, client=None, record=None, **ignored):
        record(response)
        return response
    monkeypatch.setattr(service, "_call_with_possible_fallback", call)
    monkeypatch.setattr(service, "_record_usage", lambda *a, **k: None)
    set_verifier_receipt(None)
    assert service._attempt_once({"model": "requested-backup"}, "json", False, object(),
                                 operation="verify_claim_entailment", provider="backup") == "{}"
    assert verifier_receipt() == {"provider": "backup", "requested_model": "requested-backup",
                                  "response_model": "actual-backup", "response_id": "synthetic-response"}
