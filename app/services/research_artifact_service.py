"""研究资料与主 Agent 上下文快照的会话级存取服务。"""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any
from pydantic import BaseModel

from sqlalchemy.orm import Session

from app.agent.context_builder import build_context_snapshot
from app.database.repositories import ResearchArtifactRepository
from app.schemas.artifact_schema import ARTIFACT_TYPES, STATE_ARTIFACT_FIELDS


class ArtifactNotFoundError(LookupError):
    pass


class ArtifactVersionError(ValueError):
    pass


class ResearchArtifactService:
    def __init__(self, db: Session) -> None:
        self.repo = ResearchArtifactRepository(db)

    def persist_main_context(self, session_id: str, state: dict[str, Any]) -> str:
        snapshot = build_context_snapshot(state)
        version = max(1, int(state.get("context_snapshot_version") or 0) + 1)
        ref = self.persist_payload(
            session_id=session_id,
            artifact_type="main_agent_context",
            version=version,
            payload=snapshot.model_dump(mode="json"),
            provenance={
                "source_state_version": snapshot.source_state_version,
                "context_schema_version": snapshot.schema_version,
            },
        )
        state["context_snapshot_version"] = version
        state["main_context_artifact_ref"] = ref
        state["main_agent_context"] = snapshot.context.model_dump(mode="json")
        state["main_context_snapshot"] = snapshot.model_dump(mode="json")
        return state["main_context_artifact_ref"]

    def persist_payload(
        self,
        *,
        session_id: str,
        artifact_type: str,
        payload: dict[str, Any],
        provenance: dict[str, Any] | None = None,
        version: int = 1,
        identity: str = "",
    ) -> str:
        spec = ARTIFACT_TYPES.get(artifact_type)
        if spec is None:
            raise ValueError(f"unregistered artifact type: {artifact_type}")
        self.validate_payload(payload, max_bytes=spec.max_bytes)
        if version < 1:
            raise ArtifactVersionError("artifact version must be positive")
        provenance = {**(provenance or {}), "schema_version": spec.schema_version}
        self.validate_payload(provenance, max_bytes=spec.max_bytes)
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20]
        artifact_id = "art-" + hashlib.sha256(
            f"{artifact_type}|{session_id}|{version}|{identity}|{fingerprint}|{json.dumps(provenance, sort_keys=True)}".encode("utf-8")
        ).hexdigest()[:40]
        self.repo.save(
            artifact_id=artifact_id,
            session_id=session_id,
            artifact_type=artifact_type,
            version=version,
            fingerprint=fingerprint,
            payload=payload,
            provenance=provenance,
        )
        return f"artifact://{artifact_id}"

    def resolve(
        self,
        session_id: str,
        artifact_ref: str,
        *,
        state: dict[str, Any] | None = None,
        allowed_types: set[str] | None = None,
        role: str = "service",
        expected_version: int | None = None,
    ) -> dict | None:
        ref = str(artifact_ref)
        if ref.startswith("artifact://"):
            result = self.repo.get(session_id, ref.removeprefix("artifact://"))
            if result is None:
                from app.database.models import ResearchArtifactModel
                row = self.repo.db.get(ResearchArtifactModel, ref.removeprefix("artifact://"))
                if row is not None:
                    raise PermissionError("artifact belongs to another session")
                raise ArtifactNotFoundError("artifact does not exist")
            spec = ARTIFACT_TYPES.get(result["artifact_type"])
            if spec is None or role not in spec.roles:
                raise PermissionError("artifact role is not allowed")
            if allowed_types is not None and result["artifact_type"] not in allowed_types:
                raise PermissionError(
                    f"artifact type {result['artifact_type']} is not allowed for this task"
                )
            if expected_version is not None and result["version"] != expected_version:
                raise ArtifactVersionError("artifact version mismatch")
            schema_version = result["provenance"].get("schema_version", 1)
            legacy_versions = {"main_agent_context": {"main-context-v1", "main-context-v2"},
                               "conversation_summary": {"research-memory-v1"}}
            if schema_version not in {spec.schema_version} | legacy_versions.get(result["artifact_type"], set()):
                raise ArtifactVersionError("unsupported artifact schema version")
            self.validate_payload(result["payload"], max_bytes=spec.max_bytes)
            # WHY: 旧 context 的 fingerprint 是源状态摘要；新版及其他类型是载荷 hash。
            if result["artifact_type"] != "main_agent_context" or result["provenance"].get("context_schema_version"):
                digest = hashlib.sha256(json.dumps(result["payload"], ensure_ascii=False, sort_keys=True,
                                                   default=str).encode("utf-8")).hexdigest()[:20]
                if digest != result["fingerprint"]:
                    from app.database.repositories import ResearchArtifactCorruptionError
                    raise ResearchArtifactCorruptionError("artifact payload fingerprint mismatch")
            return copy.deepcopy(result)
        if ref.startswith("state://"):
            if (state or {}).get("session_id", session_id) != session_id:
                raise PermissionError("legacy state belongs to another session")
            result = self._resolve_legacy_state_ref(ref, state or {})
            if result is None:
                raise ArtifactNotFoundError("legacy artifact does not exist")
            spec = ARTIFACT_TYPES[result["artifact_type"]]
            if role not in spec.roles or (allowed_types is not None and result["artifact_type"] not in allowed_types):
                raise PermissionError("legacy artifact type or role is not allowed")
            if expected_version not in {None, 1}:
                raise ArtifactVersionError("legacy artifact has only version 1")
            return copy.deepcopy(result)
        raise ValueError("unsupported artifact reference")

    @staticmethod
    def validate_payload(payload, *, max_bytes: int = 262144):
        def walk(value):
            if isinstance(value, (bytes, bytearray)):
                raise ValueError("binary payload requires a controlled file reference")
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).lower() in {"pdf_base64", "file_base64", "binary_content"}:
                        raise ValueError("embedded files are not allowed")
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, str) and value.lstrip().startswith(("data:application/pdf", "JVBERi0")):
                raise ValueError("embedded PDF is not allowed")
        walk(payload)
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > max_bytes:
            raise ValueError("artifact payload exceeds configured size; split into fragments")

    @staticmethod
    def controlled_file_ref(path: str) -> dict:
        from app.core.config import get_settings
        settings = get_settings()
        target = Path(path).resolve()
        roots = [Path(settings.pdf_save_dir).resolve(), Path(settings.parsed_save_dir).resolve()]
        if not any(target.is_relative_to(root) for root in roots):
            raise PermissionError("file reference is outside research storage")
        if not target.is_file():
            raise ArtifactNotFoundError("referenced file does not exist")
        return {"path": str(target), "size_bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    def externalize_state(self, session_id: str, state: dict) -> dict:
        # WHY: 节点仍可携带 SourceDiagnostic 等模型；检查点必须保存结构，不能 default=str 丢字段。
        def structured(value):
            if isinstance(value, BaseModel):
                return structured(value.model_dump(mode="python"))
            if isinstance(value, dict):
                return {key: structured(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [structured(item) for item in value]
            return value
        stored = structured(copy.deepcopy(state))
        manifest = dict(stored.get("artifact_manifest") or {})
        for field, kind in STATE_ARTIFACT_FIELDS.items():
            if field not in stored:
                continue
            value = stored.pop(field)
            is_text = isinstance(value, str)
            shape = "text" if is_text else "list"
            if field == "agent_task_results":
                # WHY: 单个任务补丁可能覆盖数百篇论文，不能把完整补丁硬塞进 256 KiB 单条资料。
                self.validate_payload(value, max_bytes=8 * 1024 * 1024)
                value = json.dumps(value, ensure_ascii=False)
                shape = "json_list"
            values = [value[i:i + 16000] for i in range(0, len(value), 16000)] if shape != "list" else value
            if not isinstance(values, list):
                raise ValueError(f"invalid state artifact field: {field}")
            entries = []
            parents = {
                "paper_details": ("ranked_papers",), "paper_cards": ("paper_details",),
                "claim_plans": ("paper_cards",), "writing_plans": ("claim_plans",),
                "review": ("writing_plans", "claim_plans"),
            }.get(field, ())
            source_refs = [entry["ref"] for parent in parents for entry in manifest.get(parent, {}).get("items", [])]
            for index, item in enumerate(values):
                payload = {"value": item}
                ref = self.persist_payload(session_id=session_id, artifact_type=kind,
                    payload=payload, identity=f"{field}:{index}",
                    provenance={"field": field, "index": index, "source_refs": source_refs})
                entries.append({"ref": ref, "type": kind, "version": 1,
                                "paper_id": item.get("paper_id") if isinstance(item, dict) else None})
            manifest[field] = {"items": entries, "shape": shape}
        parsed = stored.pop("parsed_papers", None)
        if parsed is not None:
            entries = []
            for paper_id, document in parsed.items():
                # WHY: JSON 分片保留字符定位；页码仅在解析器提供时记录，绝不猜测。
                serialized = json.dumps(document, ensure_ascii=False)
                for offset in range(0, len(serialized), 16000):
                    ref = self.persist_payload(session_id=session_id, artifact_type="document_fragment",
                        payload={"value": serialized[offset:offset + 16000]}, identity=f"{paper_id}:{offset}",
                        provenance={"paper_id": paper_id, "char_start": offset,
                                    "char_end": min(offset + 16000, len(serialized)), "encoding": "json"})
                    entries.append({"ref": ref, "type": "document_fragment", "version": 1, "paper_id": paper_id})
            manifest["parsed_papers"] = {"items": entries, "shape": "documents"}
        if stored.get("pdf_paths"):
            stored["controlled_pdf_files"] = {pid: self.controlled_file_ref(path)
                                               for pid, path in stored.pop("pdf_paths").items() if path}
        stored["artifact_manifest"] = manifest
        return stored

    def hydrate_state(self, session_id: str, state: dict, *, role: str = "service") -> dict:
        hydrated = copy.deepcopy(state)
        for field, manifest in (state.get("artifact_manifest") or {}).items():
            expected_type = "document_fragment" if field == "parsed_papers" else STATE_ARTIFACT_FIELDS.get(field)
            expected_shape = "documents" if field == "parsed_papers" else "text" if field == "review" else "list"
            allowed_shapes = {expected_shape} | ({"json_list"} if field == "agent_task_results" else set())
            if expected_type is None or manifest.get("shape") not in allowed_shapes:
                raise ValueError("artifact manifest field or shape is invalid")
            entries = manifest["items"]
            if any(item["type"] != expected_type for item in entries):
                raise PermissionError("artifact manifest type does not match its field")
            values = [self.resolve(session_id, item["ref"], allowed_types={item["type"]},
                                   expected_version=item["version"], role=role)["payload"]["value"]
                      for item in entries]
            if manifest["shape"] == "documents":
                docs = {}
                for entry, text in zip(entries, values):
                    docs[entry["paper_id"]] = docs.get(entry["paper_id"], "") + text
                hydrated[field] = {key: json.loads(value) for key, value in docs.items()}
            elif manifest["shape"] == "json_list":
                hydrated[field] = json.loads("".join(values))
                if not isinstance(hydrated[field], list):
                    raise ValueError("task result fragments must restore a list")
            else:
                hydrated[field] = "".join(values) if manifest["shape"] == "text" else values
        if state.get("controlled_pdf_files"):
            hydrated["pdf_paths"] = {}
            for pid, expected in state["controlled_pdf_files"].items():
                actual = self.controlled_file_ref(expected["path"])
                if actual != expected:
                    raise ValueError("referenced file content changed")
                hydrated["pdf_paths"][pid] = expected["path"]
        return hydrated

    @staticmethod
    def _resolve_legacy_state_ref(ref: str, state: dict[str, Any]) -> dict | None:
        """只读兼容旧 state:// 引用；新任务应使用不可变 artifact:// 引用。"""
        if ref.startswith("state://paper-card/"):
            paper_id = ref.removeprefix("state://paper-card/")
            value = next(
                (item for item in state.get("paper_cards") or []
                 if str(item.get("paper_id") or "") == paper_id),
                None,
            )
            return {"artifact_type": "paper_card", "payload": value} if value else None
        if ref.startswith("state://paper/"):
            paper_id = ref.removeprefix("state://paper/")
            value = next((item for item in (state.get("paper_details") or []) + (state.get("ranked_papers") or [])
                          if str(item.get("paper_id") or "") == paper_id), None)
            return {"artifact_type": "paper_metadata", "payload": value} if value else None
        if ref.startswith("state://writing-plan/"):
            raw = ref.removeprefix("state://writing-plan/")
            if not raw.isdigit():
                raise ValueError("invalid writing plan reference")
            plans = state.get("writing_plans") or []
            index = int(raw)
            value = plans[index] if index < len(plans) else None
            return {"artifact_type": "writing_plan", "payload": value} if value else None
        raise ValueError("unsupported state artifact reference")
