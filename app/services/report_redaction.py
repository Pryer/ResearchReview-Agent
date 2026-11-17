"""报告副本的凭证脱敏，避免改动证据和运行状态。"""
import re

_PATTERNS = [
    (re.compile(r"(?i)(\b(?:authorization|proxy-authorization)[\"']?\s*[:=]\s*)(?:bearer|basic)\s+[^\s,;\"'<>]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|client[_-]?secret|secret)[\"']?\s*[=:]\s*[\"']?)[^\s\"'&,;<>]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@"), r"\1[REDACTED]@"),
    (re.compile(r"(?i)([?&](?:token|access_token|refresh_token|api_key|key|signature)=)[^&#\s]+"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[REDACTED]"),
]


def redact_text(text):
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
