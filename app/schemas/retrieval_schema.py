"""混合检索的轻量、可序列化契约。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any


RETRIEVAL_VERSION = "hybrid.v1"


def fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrievalDocument:
    identity: str
    title: str
    abstract: str
    keywords: tuple[str, ...]
    language: str

    @property
    def text(self) -> str:
        # WHY: 模型输入是索引视图；原论文对象保持原文，供元数据与证据定位使用。
        return "\n".join(part for part in (
            self.title.strip(), self.abstract.strip(), " ".join(self.keywords).strip()
        ) if part)[:6000]

    @property
    def content_fingerprint(self) -> str:
        return fingerprint((RETRIEVAL_VERSION, self.identity, self.text))


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    role: str
    family: str
    source: str

    @property
    def query_fingerprint(self) -> str:
        return fingerprint((RETRIEVAL_VERSION, self.text, self.role, self.family))
