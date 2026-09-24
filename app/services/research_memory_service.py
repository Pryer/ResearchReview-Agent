"""追加式历史归档及按游标增量构建的研究记忆。"""
import json
from uuid import uuid4
from sqlalchemy import select
from app.database.models import ResearchEventModel, ResearchMemoryModel
from app.services.research_artifact_service import ResearchArtifactService


class ResearchMemoryService:
    def __init__(self, db):
        self.db = db
        self.artifacts = ResearchArtifactService(db)

    def append_event(self, *, session_id, event_type, payload, source_refs=None, event_key=None):
        key = event_key or uuid4().hex
        existing = self.db.scalar(select(ResearchEventModel).where(
            ResearchEventModel.session_id == session_id, ResearchEventModel.event_key == key))
        if existing is not None:
            return str(existing.event_id)
        self.artifacts.validate_payload(payload, max_bytes=8 * 1024 * 1024)
        row = ResearchEventModel(session_id=session_id, event_key=key, event_type=event_type,
            payload_json=json.dumps({"payload": payload, "source_refs": source_refs or []}, ensure_ascii=False))
        self.db.add(row)
        self.db.flush()
        return str(row.event_id)

    def archive_history(self, session_id, history):
        # WHY: 在展示窗口截断前逐条落库；标记随会话保存，重复内容也保留独立发生次数。
        for item in history:
            if not isinstance(item, dict):
                continue
            key = item.setdefault("_archive_event_key", "history:" + uuid4().hex)
            self.append_event(session_id=session_id, event_type="history_entry",
                payload={k: v for k, v in item.items() if k != "_archive_event_key"}, event_key=key)
        return history[-50:]

    def legacy_imported(self, session_id):
        return self.db.scalar(select(ResearchEventModel.event_id).where(
            ResearchEventModel.session_id == session_id,
            ResearchEventModel.event_key == "legacy-import-complete")) is not None

    def import_legacy_history(self, session_id, state):
        # WHY: 旧版本的布尔标记不代表已迁入新事件表，数据库事件标记才是跨进程幂等依据。
        if self.legacy_imported(session_id):
            state["research_memory_migrated"] = True
            return []
        refs = []
        # 旧 artifact 事件只导入一次；保留原 payload 和来源，不能凭摘要恢复已丢失内容。
        for event in self.artifacts.repo.list(session_id, "conversation_event"):
            body = event.get("payload") or {}
            refs.append(self.append_event(session_id=session_id,
                event_type=body.get("event_type") or "legacy_event", payload=body.get("payload") or {},
                source_refs=(event.get("provenance") or {}).get("source_refs") or [],
                event_key="legacy:" + event["artifact_id"]))
        for index, item in enumerate(state.get("conversation_history") or []):
            if not isinstance(item, dict):
                continue
            key = item.setdefault("_archive_event_key", "legacy-history:" + str(index))
            refs.append(self.append_event(session_id=session_id,
                event_type=str(item.get("type") or "legacy_message"),
                payload={"role": item.get("role"), "content": item.get("content"), "legacy_import": True},
                event_key=key))
        state["research_memory_migrated"] = True
        state["research_memory_legacy_loss_unknown"] = True
        self.append_event(session_id=session_id, event_type="legacy_import_complete",
            payload={"earlier_history_unknown": True}, event_key="legacy-import-complete")
        return refs

    def rebuild_summary(self, session_id):
        row = self.db.get(ResearchMemoryModel, session_id)
        if row is None:
            row = ResearchMemoryModel(session_id=session_id, cursor=0, summary_json="{}")
            self.db.add(row)
            self.db.flush()
        state = json.loads(row.summary_json)
        events = self.db.scalars(select(ResearchEventModel).where(
            ResearchEventModel.session_id == session_id, ResearchEventModel.event_id > row.cursor
        ).order_by(ResearchEventModel.event_id)).all()
        if not events and row.summary_ref:
            return row.summary_ref
        decisions = {str(item.get("decision_id") or item.get("decision") or ""): item
                     for item in state.get("decisions") or []}
        questions = {str(item.get("question_id") or item.get("question") or ""): item
                     for item in state.get("open_questions") or []}
        findings = list(state.get("findings") or [])
        citations = list(state.get("citations") or [])
        for event in events:
            encoded = json.loads(event.payload_json)
            body = encoded.get("payload") or {}
            if event.event_type == "research_result":
                questions.clear()
                # 当前主控快照中的决定集合已处理用户更正，不能复活历史决定。
                decisions.clear()
                findings = []
            findings.extend(body.get("findings") or [])
            citations.extend(str(item) for item in body.get("citations") or [])
            for item in body.get("decisions") or []:
                decisions[str(item.get("decision_id") or item.get("decision") or "")] = item
            for key in body.get("invalidated_decision_ids") or []:
                decisions.pop(str(key), None)
            for item in body.get("open_questions") or []:
                questions[str(item.get("question_id") or item.get("question") or "")] = item
            for key in body.get("resolved_question_ids") or []:
                questions.pop(str(key), None)
        cursor = events[-1].event_id if events else row.cursor
        users = [v for v in decisions.values() if v.get("source") == "user"]
        others = [v for v in decisions.values() if v.get("source") != "user"][-24:]
        blocking = [q for q in questions.values() if q.get("blocking")]
        optional = [q for q in questions.values() if not q.get("blocking")][-24:]
        payload = {"cursor": cursor, "findings": findings[-40:], "decisions": users + others,
                   "open_questions": blocking + optional, "citations": list(dict.fromkeys(citations))[-80:]}
        ref = self.artifacts.persist_payload(session_id=session_id, artifact_type="conversation_summary",
            payload=payload, version=max(1, cursor), identity="summary-" + str(cursor),
            provenance={"after_event_id": row.cursor, "through_event_id": cursor,
                        "previous_summary": row.summary_ref})
        row.cursor, row.summary_ref = cursor, ref
        row.summary_json = json.dumps(payload, ensure_ascii=False)
        self.db.flush()
        return ref
