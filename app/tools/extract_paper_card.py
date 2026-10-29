"""论文卡片抽取工具。

从论文全文或摘要中抽取结构化的 PaperCard。
优先基于全文抽取，无全文时基于摘要抽取。
"""

from __future__ import annotations

import json
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional

from app.core.exceptions import PaperCardExtractionError
from app.schemas.vector_schema import VectorIntegrityError, VectorStoreUnavailable
from app.services.evidence_support import authorized_field_claim, literal_support
from app.core.logger import get_logger
from app.schemas.paper_schema import (
    AccessLevel,
    EvidenceClaim,
    EvidenceSpan,
    PaperCard,
    PaperEvidenceState,
    PublicationStatus,
    VerificationStatus,
)

logger = get_logger(__name__)


def _coerce_publication_status(value: Any) -> PublicationStatus:
    """把检索层出版状态归一为枚举；无法确认时保持 unknown。"""
    text = str(getattr(value, "value", value) or "").strip().lower()
    try:
        return PublicationStatus(text)
    except ValueError:
        return PublicationStatus.UNKNOWN


def extract_paper_card(
    paper: Dict[str, Any],
    parsed_text: Optional[Dict[str, Any]] = None,
    llm=None,
    topic: str = "",
    *, should_cancel: Optional[Callable[[], bool]] = None,
) -> PaperCard:
    """抽取单篇论文的结构化卡片。

    优先基于全文抽取；无全文时基于摘要抽取。

    Args:
        paper: 论文元数据字典。
        parsed_text: PDF 解析后的结构化文本。
        llm: LLM 客户端。
        topic: 研究主题（用于相关性说明）。

    Returns:
        PaperCard 实例。
    """
    # CNKI 默认只使用详情页提供的摘要级证据。即使调用方传入历史缓存的
    # PDF 解析结果，也不得把它升级为全文证据。
    from app.tools.download_pdf import is_cnki_paper

    if should_cancel and should_cancel():
        raise InterruptedError("卡片抽取已取消")
    if is_cnki_paper(paper):
        if parsed_text and parsed_text.get("full_text"):
            logger.info(
                "Ignoring cached full text for abstract-only source: paper_id=%s source=cnki",
                paper.get("paper_id") or "unknown",
            )
        parsed_text = None

    has_full_text = bool(parsed_text and parsed_text.get("full_text"))
    has_abstract = bool((paper.get("abstract") or "").strip())
    if not has_full_text and not has_abstract:
        return _unavailable_card(paper, "no_accessible_content", topic)
    if llm is None:
        raise PaperCardExtractionError("PaperCard 抽取需要 LLM；规则抽卡路径已移除")
    try:
        card = (extract_from_full_text(paper, parsed_text, llm, topic, should_cancel=should_cancel)
                if has_full_text else extract_from_abstract(paper, llm, topic))
        card = _quality_gate(_attach_evidence(_deduplicate_card_fields(card), paper, parsed_text))
        if not validate_paper_card(card):
            return _unavailable_card(paper, "invalid_card_structure", topic)
        return card
    except Exception as exc:
        from app.agent.execution_budget import is_control_exception
        if isinstance(exc, (InterruptedError, VectorIntegrityError, VectorStoreUnavailable)) or is_control_exception(exc):
            raise
        # WHY: 模型失败不能被规则抽卡包装成成功；只返回不可用于写作的身份记录。
        logger.warning("PaperCard extraction failed: paper=%s error=%s", paper.get("paper_id"), type(exc).__name__)
        return _unavailable_card(paper, f"model_failure:{type(exc).__name__}", topic)


def _unavailable_card(paper: Dict[str, Any], reason: str, topic: str = "") -> PaperCard:
    metadata = {**paper, "paper_id": str(paper.get("paper_id") or ""), "title": str(paper.get("title") or "")}
    card = _dict_to_card({}, metadata, "metadata", topic)
    return card.model_copy(update={
        "quality_status": "invalid", "quality_issues": [f"extraction:{reason}"],
        "unsupported_fields": ["research_problem", "method", "results", "contributions", "limitations"],
    })


def extract_from_full_text(
    paper: Dict[str, Any],
    parsed_text: Dict[str, Any],
    llm,
    topic: str = "",
    *, should_cancel: Optional[Callable[[], bool]] = None,
) -> PaperCard:
    """逐块摘录后确定性聚合；保留原文与处理覆盖，不追加聚合模型调用。"""
    from app.core.config import get_settings
    from app.agent.execution_budget import check_execution, is_control_exception
    from app.prompt.paper_card import CHUNK_FIELD_EXTRACTION_PROMPT
    from app.schemas.paper_schema import ExtractionCoverage, ExtractionChunkRecord
    from app.tools.chunk_pdf import chunk_pdf
    from app.tools.chunk_retriever import retrieve_chunks

    settings = get_settings()
    chunks = chunk_pdf(parsed_text, max_chars=settings.card_extraction_chunk_chars,
                       paper_id=str(paper.get("paper_id") or ""))
    retrieval = retrieve_chunks(str(paper.get("paper_id") or ""), chunks,
                                settings=settings, topic=topic, should_cancel=should_cancel, parsed_text=parsed_text)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    selected = set(retrieval.selected_chunk_ids)
    # WHY: 先按检索/章节计划执行，诊断仍覆盖全部块；Top-K 命中本身不构成证据。
    chunks = [by_id[i] for i in retrieval.selected_chunk_ids] + [
        chunk for chunk in chunks if chunk["chunk_id"] not in selected]
    fields = ("research_problem", "study_design", "sample_size", "data_modalities",
              "behavior_categories", "method", "dataset", "metrics", "results",
              "contributions", "limitations")
    values = {name: [] for name in fields}
    spans, mapping, records = [], {}, []
    calls = 0
    stopped = False
    issues = ["chunk_retrieval:degraded"] if retrieval.degraded else []
    for chunk in chunks:
        check_execution()
        if should_cancel and should_cancel():
            raise InterruptedError("卡片抽取已取消")
        status, reason = "processed", None
        if chunk["excluded"]:
            status, reason = "excluded", "reference_section"
        elif stopped or chunk["chunk_id"] not in selected or calls >= settings.card_extraction_max_chunks:
            status, reason = "skipped", "provider_failure" if stopped else "chunk_call_limit"
        else:
            calls += 1
            try:
                response = llm.complete(CHUNK_FIELD_EXTRACTION_PROMPT.format(
                    title=paper.get("title", ""), text=chunk["text"]), response_format="json")
                check_execution()
                data = _safe_parse_json(response)
                if (not isinstance(data, dict) or not any(name in data for name in fields)
                        or any(not isinstance(data[name], list)
                               or any(not isinstance(q, str) for q in data[name])
                               for name in fields if name in data)):
                    raise ValueError("invalid chunk extraction response")
            except Exception as exc:
                if isinstance(exc, InterruptedError) or is_control_exception(exc):
                    raise
                logger.warning("Chunk extraction failed: %s chunk=%s error=%s",
                               paper.get("paper_id"), chunk["chunk_id"], type(exc).__name__)
                status, reason, stopped = "failed", type(exc).__name__, True
                data = {}
            for field in fields:
                quotes = data.get(field, [])
                if not isinstance(quotes, list):
                    continue
                if len(quotes) > 3:
                    issues.append("extraction:field_quote_limit")
                for quote in quotes[:3]:
                    if not isinstance(quote, str) or not 15 <= len(quote.strip()) <= 1000:
                        issues.append("extraction:rejected_quote")
                        continue
                    quote = quote.strip()
                    start = chunk["text"].find(quote)
                    if start < 0:
                        issues.append("extraction:quote_not_in_source")
                        continue
                    if quote in values[field]:
                        continue
                    values[field].append(quote)
                    evidence_id = f"{paper.get('paper_id')}:chunk:{len(spans)}"
                    spans.append(EvidenceSpan(
                        evidence_id=evidence_id, text=quote, page=chunk["page"],
                        section=chunk["section"], source_type="full_text",
                        char_start=chunk["char_start"] + start,
                        char_end=chunk["char_start"] + start + len(quote),
                        offset_basis=chunk["offset_basis"],
                        provider=paper.get("source"), source_url=paper.get("pdf_url") or paper.get("url"),
                    ))
                    mapping.setdefault(field, []).append(evidence_id)
        records.append(ExtractionChunkRecord(**{
            key: chunk[key] for key in ("chunk_id", "page", "section", "char_start", "char_end", "offset_basis")
        }, status=status, reason=reason))
    scalar = {"research_problem", "study_design", "sample_size", "method", "dataset", "results"}
    data = {field: "\n".join(items) if field in scalar else items for field, items in values.items()}
    # WHY: 元数据只来自检索层；缺失字段保持空，不能让聚合步骤推断或补写。
    card = _dict_to_card(data, paper, "full_text", topic)
    complete = bool(records) and any(r.status == "processed" for r in records) and all(
        r.status in {"processed", "excluded"} for r in records)
    return card.model_copy(update={
        **{key: (value or None) if key in {"sample_size", "dataset", "results"} else value
           for key, value in data.items()},
        "extraction_coverage": ExtractionCoverage(
            status="complete" if complete else "partial",
            chunks=sorted(records, key=lambda record: record.chunk_id), retrieval=retrieval),
        "evidence_spans": spans, "field_evidence": mapping,
        "quality_issues": list(dict.fromkeys(issues + ([] if complete else ["extraction:partial_coverage"]))),
    })


def extract_from_abstract(
    paper: Dict[str, Any],
    llm,
    topic: str = "",
) -> PaperCard:
    """基于摘要用 LLM 生成 PaperCard。"""
    from app.prompt.paper_card import PAPER_CARD_EXTRACTION_PROCTION_PROMPT

    abstract = paper.get("abstract") or ""

    prompt = PAPER_CARD_EXTRACTION_PROCTION_PROMPT.format(
        evidence_label="[仅有摘要，基于摘要抽取]",
        title=paper.get("title", ""),
        full_text_or_json=f"摘要：\n{abstract}",
        paper_id=paper.get("paper_id", "unknown"),
        evidence_source="abstract",
    )

    response = llm.complete(prompt, response_format="json")
    data = _safe_parse_json(response)
    return _dict_to_card(data, paper, "abstract", topic)


def validate_paper_card(card: PaperCard) -> bool:
    """验证身份、证据引用完整性、访问等级和主题关系契约。"""
    if not card.paper_id:
        return False
    if not card.title:
        return False
    if card.evidence_source not in ("abstract", "full_text", "metadata"):
        return False
    if card.relation_type not in (None, "direct", "near", "indirect", "unrelated"):
        return False
    evidence_ids = [span.evidence_id for span in card.evidence_spans]
    if any(not evidence_id for evidence_id in evidence_ids):
        return False
    if len(evidence_ids) != len(set(evidence_ids)):
        return False
    valid_ids = set(evidence_ids)
    if any(
        evidence_id not in valid_ids
        for mapped_ids in card.field_evidence.values()
        for evidence_id in mapped_ids
    ):
        return False
    access_rank = {
        AccessLevel.METADATA_ONLY: 0,
        AccessLevel.TITLE_AND_KEYWORDS: 1,
        AccessLevel.ABSTRACT: 2,
        AccessLevel.PARTIAL_FULL_TEXT: 3,
        AccessLevel.FULL_TEXT: 4,
    }
    card_rank = access_rank.get(card.evidence_state.access_level, 0)
    for claims in card.field_claims.values():
        for claim in claims:
            if claim.evidence_id and claim.evidence_id not in valid_ids:
                return False
            if claim.explicitly_reported and (not claim.evidence_id or not claim.source_text.strip()):
                return False
            if access_rank.get(claim.evidence_level, 0) > card_rank:
                return False
    return True


def resolve_evidence_permissions(
    paper: Dict[str, Any],
    parsed_text: Optional[Dict[str, Any]] = None,
) -> PaperEvidenceState:
    """根据实际可访问内容确定能力边界，不用文献类型冒充访问等级。"""
    abstract = bool(_clean_abstract(str(paper.get("abstract") or "")))
    keywords = bool(paper.get("keywords"))
    full_text = str((parsed_text or {}).get("full_text") or "").strip()
    sections = {
        str(key).lower(): str(value or "")
        for key, value in ((parsed_text or {}).get("sections") or {}).items()
        if str(value or "").strip()
    }
    if full_text:
        complete_markers = {"method", "experiment", "references"}
        access = (
            AccessLevel.FULL_TEXT
            if complete_markers.issubset(sections)
            else AccessLevel.PARTIAL_FULL_TEXT
        )
    elif abstract:
        access = AccessLevel.ABSTRACT
    elif keywords:
        access = AccessLevel.TITLE_AND_KEYWORDS
    else:
        access = AccessLevel.METADATA_ONLY

    stable_source = bool(paper.get("doi") or paper.get("url") or paper.get("arxiv_id"))
    basic_metadata = bool(paper.get("title") and paper.get("year"))
    if stable_source and access in {
        AccessLevel.ABSTRACT, AccessLevel.PARTIAL_FULL_TEXT, AccessLevel.FULL_TEXT
    }:
        verification = VerificationStatus.SOURCE_VERIFIED
    elif stable_source or basic_metadata:
        verification = VerificationStatus.METADATA_VERIFIED
    else:
        verification = VerificationStatus.UNVERIFIED

    available = list(sections)
    if abstract and "abstract" not in available:
        available.append("abstract")
    required = ["abstract", "method", "experiment", "conclusion", "references"]
    missing = [name for name in required if name not in available]
    content_access = access in {
        AccessLevel.ABSTRACT, AccessLevel.PARTIAL_FULL_TEXT, AccessLevel.FULL_TEXT
    }
    return PaperEvidenceState(
        access_level=access,
        verification_status=verification,
        available_sections=available,
        missing_sections=missing,
        can_extract_method=content_access,
        can_extract_results=content_access,
        can_extract_limitations=access == AccessLevel.FULL_TEXT,
        can_compare_metrics=access == AccessLevel.FULL_TEXT,
    )


_PROBLEM_PATTERNS = (
    r"\b(?:this study|this paper|we)\s+(?:aims?|examines?|investigates?|explores?|studies?|evaluates?)\b",
    r"(?:本文|本研究)\s*(?:旨在|研究|探讨|分析|考察|评估)",
    r"研究(?:问题|目的|目标)(?:是|为|在于)",
)
_METHOD_PATTERNS = (
    r"\b(?:we|this (?:paper|study))\s+(?:propose|present|introduce|develop|design|adopt|use|employ)s?\b",
    r"(?:本文|本研究)\s*(?:提出|采用|构建|设计|使用|运用|基于)",
    r"(?:采用|使用|运用).{0,80}(?:方法|模型|框架|数据|访谈|问卷|实验)",
)
_RESULT_PATTERNS = (
    r"\b(?:experimental )?results?\s+(?:show|indicate|demonstrate|reveal)s?\b",
    r"\b(?:we|the (?:model|method|study))\s+(?:achieve|outperform|find|found)s?\b",
    r"\b(?:accuracy|f1|mAP|precision|recall)\b.{0,40}\b\d+(?:\.\d+)?%",
    r"(?:实验|研究|分析)结果(?:表明|显示|发现)",
    r"(?:准确率|召回率|精确率|F1|mAP).{0,20}\d+(?:\.\d+)?%",
)
_CONTRIBUTION_PATTERNS = (
    r"\b(?:we|this paper)\s+(?:propose|present|introduce|contribute)s?\b",
    r"(?:本文|本研究)(?:的主要贡献|提出|构建|设计)",
)
_LIMITATION_PATTERNS = (
    r"\b(?:a|the|our|this study's?)\s+limitations?\b",
    r"\blimitations?\s+(?:include|are|is)\b",
    r"(?:本研究|本文)(?:的)?(?:局限|不足)(?:在于|包括|是)",
)


def _clean_abstract(text: str) -> str:
    """清除中文网页摘要前混入的作者、单位和“摘要”标签。"""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    abstract_marker = re.search(r"(?:^|\s)摘要\s*[:：]", text)
    if abstract_marker:
        text = text[abstract_marker.end():].strip()
    text = re.sub(r"^(?:abstract\s*[:：]\s*)+", "", text, flags=re.IGNORECASE)
    return text


def _contains_affiliation(text: str) -> bool:
    return bool(re.search(
        r"(?:大学|学院|研究院|实验室).{0,30}(?:北京|上海|江苏|广东|中国|省|市)|"
        r"\b(?:university|institute|department|laboratory)\b.{0,80}\b(?:china|usa|uk|city)\b",
        str(text or ""),
        re.IGNORECASE,
    ))


def _is_clean_evidence_sentence(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if re.search(
        r"©|\bcopyright\b|all rights reserved|\bthe author\(s\)\b|"
        r"\bissn\b|\bjournal homepage\b|\bcorresponding author\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"^(?:abstract|introduction|keywords?|highlights?)\b(?:\s*[:：.-]|\s+)",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:for clarity|in the main text|see (?:fig|figure|table)|"
        r"details? (?:are|is) (?:provided|shown)|due to space limitations|"
        r"we present only)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    english_words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if english_words and len(english_words) >= max(2, len(chinese_chars)):
        first_alpha = re.search(r"[A-Za-z]", text)
        if first_alpha and text[first_alpha.start()].islower():
            return False
        if text.rstrip().endswith((",", ":", ";", "-", "–", "—")):
            return False
        if re.search(
            r"\b(?:a|an|and|or|the|for|of|to|with|using|based|validating|"
            r"including|such as|between|from|by|in|on|at|this|our)$",
            text,
            re.IGNORECASE,
        ):
            return False
        # PDF 行切分产生的短标题、指标标签和页眉通常没有完整谓语或句末标点。
        if len(english_words) <= 12 and not re.search(r"[.!?]$", text):
            return False
        if re.search(r"^(?:journal|proceedings|volume|vol\.?|issue)\b", text, re.I):
            return False
    return bool(
        len(text) >= 15
        and not _contains_affiliation(text)
        and "...[truncated]" not in text
        and not re.search(r"(?:^|\s)摘要\s*[:：]", text)
    )


def _normalized_claim_text(text: str) -> str:
    """生成跨字段去重键，忽略空白、标点和大小写差异。"""
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def _same_claim_text(left: str, right: str) -> bool:
    left_key = _normalized_claim_text(left)
    right_key = _normalized_claim_text(right)
    return bool(left_key and right_key and left_key == right_key)


def _deduplicate_card_fields(card: PaperCard) -> PaperCard:
    """同一原句只允许承担一个语义字段。

    摘要原句可能同时被模型写入 research_problem、method 和
    contributions。这里在证据绑定前消除重复，避免同一个 evidence span
    被 Writer 当作三条独立声明使用。
    """
    research_problem = str(card.research_problem or "").strip()
    method = str(card.method or "").strip()
    results = str(card.results or "").strip()
    contributions = [str(item).strip() for item in card.contributions if str(item).strip()]
    limitations = [str(item).strip() for item in card.limitations if str(item).strip()]
    issues = list(card.quality_issues or [])

    problem_is_explicit = any(
        re.search(pattern, research_problem, re.IGNORECASE)
        for pattern in _PROBLEM_PATTERNS
    )
    method_is_explicit = any(
        re.search(pattern, method, re.IGNORECASE)
        for pattern in _METHOD_PATTERNS
    )

    if research_problem and method and _same_claim_text(research_problem, method):
        if method_is_explicit and not problem_is_explicit:
            research_problem = ""
            issues.append("research_problem_method:duplicate_kept_as_method")
        else:
            method = ""
            issues.append("research_problem_method:duplicate_kept_as_problem")

    if research_problem and any(
        _same_claim_text(research_problem, item) for item in contributions
    ):
        if not problem_is_explicit:
            research_problem = ""
            issues.append("research_problem_contributions:duplicate_kept_as_contribution")
        else:
            contributions = [
                item for item in contributions
                if not _same_claim_text(research_problem, item)
            ]
            issues.append("research_problem_contributions:duplicate_kept_as_problem")

    deduped_contributions: list[str] = []
    seen_contributions: set[str] = set()
    for item in contributions:
        key = _normalized_claim_text(item)
        if (
            not key
            or key in seen_contributions
            or (method and _same_claim_text(item, method))
            or (results and _same_claim_text(item, results))
        ):
            issues.append("contributions:cross_field_duplicate")
            continue
        seen_contributions.add(key)
        deduped_contributions.append(item)

    occupied = [research_problem, method, results, *deduped_contributions]
    deduped_limitations: list[str] = []
    seen_limitations: set[str] = set()
    for item in limitations:
        key = _normalized_claim_text(item)
        if (
            not key
            or key in seen_limitations
            or any(value and _same_claim_text(item, value) for value in occupied)
        ):
            issues.append("limitations:cross_field_duplicate")
            continue
        seen_limitations.add(key)
        deduped_limitations.append(item)

    return card.model_copy(update={
        "research_problem": research_problem,
        "method": method,
        "contributions": deduped_contributions,
        "limitations": deduped_limitations,
        "quality_issues": list(dict.fromkeys(issues)),
    })


def _quality_gate(card: PaperCard) -> PaperCard:
    """清空不可靠字段并记录原因；证据不足优先于填满字段。"""
    issues: list[str] = list(card.quality_issues or [])
    updates: dict[str, Any] = {}
    method = str(card.method or "").strip()
    result = str(card.results or "").strip()
    contributions = [
        str(item).strip() for item in card.contributions if str(item).strip()
    ]
    limitations = [str(item).strip() for item in card.limitations if str(item).strip()]
    evidence_state = card.evidence_state
    snapshot = card.model_dump(mode="json")
    field_claims = {
        field: [
            claim.model_copy(update={
                "explicitly_reported": authorized_field_claim(snapshot, claim.model_dump(mode="json")),
            }) for claim in claims
            if field not in {
                "research_problem", "method", "results", "contributions", "limitations"
            }
            or _is_clean_evidence_sentence(str(claim.claim or ""))
        ]
        for field, claims in (card.field_claims or {}).items()
    }
    field_claims = {
        field: claims for field, claims in field_claims.items() if claims
    }
    unsupported = list(card.unsupported_fields or [])

    for field, value in (("research_problem", card.research_problem), ("method", method), ("results", result)):
        if value and _contains_affiliation(str(value)):
            updates[field] = None if field == "results" else ""
            issues.append(f"{field}:contains_affiliation")
        elif value and ("...[truncated]" in str(value) or str(value).rstrip().endswith(("……", "..."))):
            updates[field] = None if field == "results" else ""
            issues.append(f"{field}:truncated_fragment")
        elif value and not _is_clean_evidence_sentence(str(value)):
            updates[field] = None if field == "results" else ""
            issues.append(f"{field}:malformed_evidence_fragment")

    clean_contributions = [
        value for value in contributions if _is_clean_evidence_sentence(value)
    ]
    if len(clean_contributions) != len(contributions):
        updates["contributions"] = clean_contributions
        issues.append("contributions:malformed_evidence_fragment")

    if method and not any(re.search(pattern, method, re.IGNORECASE) for pattern in _METHOD_PATTERNS) and re.search(
        r"^(?:however|although|despite|然而|但是|尽管)|(?:traditional|existing|传统|现有).{0,80}(?:problem|limitation|不足|问题|困难)",
        method,
        re.IGNORECASE,
    ):
        updates["method"] = ""
        issues.append("method:not_method_statement")
    if result and not any(re.search(pattern, result, re.IGNORECASE) for pattern in _RESULT_PATTERNS):
        updates["results"] = None
        issues.append("results:not_result_statement")
    if method and result and _match_score(method, result) >= 0.85:
        updates["results"] = None
        issues.append("method_results:duplicate")

    clean_limitations = [
        value for value in limitations
        if any(re.search(pattern, value, re.IGNORECASE) for pattern in _LIMITATION_PATTERNS)
        and not _contains_affiliation(value)
    ]
    if len(clean_limitations) != len(limitations):
        updates["limitations"] = clean_limitations
        issues.append("limitations:not_author_stated")

    def explicitly_supported(field: str) -> bool:
        return any(claim.explicitly_reported for claim in field_claims.get(field, []))

    if method and (not evidence_state.can_extract_method or not explicitly_supported("method")):
        updates["method"] = ""
        unsupported.append("method")
        issues.append("method:insufficient_explicit_evidence")
    if result and (not evidence_state.can_extract_results or not explicitly_supported("results")):
        updates["results"] = None
        unsupported.append("results")
        issues.append("results:insufficient_explicit_evidence")
    if limitations and (
        not evidence_state.can_extract_limitations or not explicitly_supported("limitations")
    ):
        updates["limitations"] = []
        unsupported.append("limitations")
        issues.append("limitations:access_level_too_weak")

    if card.dataset and not explicitly_supported("dataset"):
        updates["dataset"] = None
        unsupported.append("dataset")
    if card.metrics and not explicitly_supported("metrics"):
        updates["metrics"] = []
        unsupported.append("metrics")

    if card.evidence_source == "metadata":
        updates.update({
            "research_problem": "", "method": "", "results": None,
            "limitations": [], "dataset": None, "metrics": [],
        })
        issues.append("metadata_only:no_content_claims")
        unsupported.extend([
            "research_problem", "method", "results", "limitations", "dataset",
            "metrics", "sample_size", "experiment_setup", "dataset_split",
            "ablation_results", "metric_comparison",
        ])
    elif evidence_state.access_level == AccessLevel.ABSTRACT:
        unsupported.extend([
            "limitations", "experiment_setup", "dataset_split", "ablation_results",
            "fair_baseline_comparison", "detailed_model_structure",
        ])
    elif evidence_state.access_level == AccessLevel.PARTIAL_FULL_TEXT:
        unsupported.extend([
            "experiment_setup", "dataset_split", "ablation_results",
            "fair_baseline_comparison",
        ])

    effective_research_problem = updates.get("research_problem", card.research_problem)
    meaningful = bool(
        (updates.get("method", method) or "")
        or (updates.get("results", result) or "")
        or effective_research_problem
    )
    status = "partial" if meaningful else "invalid"
    if meaningful and not issues and card.evidence_source == "full_text":
        status = "valid"
    cleared_fields = {
        field for field, value in updates.items()
        if field in field_claims and value in ("", None, [])
    }
    for field in cleared_fields:
        field_claims.pop(field, None)
    field_evidence = {
        field: evidence_ids
        for field, evidence_ids in card.field_evidence.items()
        if field not in cleared_fields
    }
    updates.update({
        "field_evidence": field_evidence,
        "field_claims": field_claims,
        "unsupported_fields": list(dict.fromkeys(unsupported)),
        "quality_status": status,
        "quality_issues": list(dict.fromkeys(issues)),
    })
    return card.model_copy(update=updates)


def batch_extract_paper_cards(
    papers: List[Dict[str, Any]],
    parsed_texts: Dict[str, Dict[str, Any]],
    llm=None,
    topic: str = "",
    max_workers: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[PaperCard]:
    """受控并发批量抽取 PaperCard，保持原始论文顺序。"""
    if not papers:
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from app.agent.execution_budget import submit_with_context
    from app.core.config import get_settings

    settings = get_settings()
    workers = max_workers or getattr(settings, "card_extraction_max_workers", 5)
    workers = max(1, min(workers, len(papers)))

    def _extract_single(idx_and_paper: tuple[int, Dict[str, Any]]) -> tuple[int, PaperCard]:
        idx, paper = idx_and_paper
        paper_id = str(paper.get("paper_id") or "")
        parsed = parsed_texts.get(paper_id)
        try:
            card = extract_paper_card(paper, parsed, llm, topic, should_cancel=should_cancel)
            return idx, card
        except InterruptedError:
            # 协作式取消必须向上传播，不能被降级为规则抽取。
            raise
        except Exception as e:
            from app.agent.execution_budget import is_control_exception
            if isinstance(e, (VectorIntegrityError, VectorStoreUnavailable)) or is_control_exception(e):
                raise
            logger.warning("Failed to extract card for %s: %s", paper_id, e)
            if isinstance(e, PaperCardExtractionError):
                raise
            return idx, _unavailable_card(paper, f"batch_failure:{type(e).__name__}", topic)


    if should_cancel and should_cancel():
        raise InterruptedError("卡片抽取已取消")
    if workers <= 1 or len(papers) <= 1:
        results = [_extract_single((i, p)) for i, p in enumerate(papers)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                submit_with_context(executor, _extract_single, (i, p))
                for i, p in enumerate(papers)
            ]
            results = []
            for future in as_completed(futures):
                if should_cancel and should_cancel():
                    for pending_future in futures:
                        pending_future.cancel()
                    raise InterruptedError("卡片抽取已取消")
                results.append(future.result())

    results.sort(key=lambda x: x[0])
    cards = [r[1] for r in results]
    logger.info("Extracted %d PaperCards with concurrency %d", len(cards), workers)
    return cards


def _coerce_year(value) -> Optional[int]:
    """年份字段防御性转换：LLM 可能返回 "2023"、"2023年" 或非数字文本。"""
    if value is None:
        return None
    text = str(value).strip().rstrip("年")
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _dict_to_card(
    data: dict,
    paper: dict,
    evidence_source: str,
    topic: str,
) -> PaperCard:
    """将 LLM 输出的 JSON 映射为 PaperCard。

    身份与权威书目元数据（paper_id、title、authors、year、venue、doi、url、
    publication_status）只来自检索层，LLM 回显值一律不覆盖，否则幻觉作者或
    DOI 会直接污染最终参考文献。文献类型、同行评审状态与证据等级一律由可
    验证的元数据/文本规则确定性派生，不接受 LLM 覆盖——防止模型把自己升级
    为 systematic_review 高等级证据。
    """
    source_text = " ".join(
        str(value or "")
        for value in (paper.get("title"), paper.get("abstract"), data.get("method"))
    )
    publication = _infer_publication_profile(paper, source_text)
    return PaperCard(
        paper_id=str(paper.get("paper_id") or ""),
        title=paper.get("title") or "",
        authors=_coerce_to_list(paper.get("authors") or []),
        year=_coerce_year(paper.get("year")),
        venue=paper.get("venue"),
        doi=paper.get("doi"),
        url=paper.get("url"),
        source=str(paper.get("source") or "unknown"),
        arxiv_id=paper.get("arxiv_id"),
        publication_status=_coerce_publication_status(paper.get("publication_status")),
        publication_type=publication["publication_type"],
        peer_review_status=publication["peer_review_status"],
        evidence_level=publication["evidence_level"],
        research_problem=data.get("research_problem", ""),
        study_design=data.get("study_design") or "",
        sample_size=_coerce_optional_text(data.get("sample_size")),
        data_modalities=_coerce_to_list(data.get("data_modalities")),
        # WHY: 内容字段只来自模型绑定的原文，缺失时不得用词表推测类别。
        behavior_categories=_coerce_to_list(data.get("behavior_categories")),
        method=data.get("method", ""),
        dataset=data.get("dataset"),
        metrics=_coerce_to_list(data.get("metrics", [])),
        results=_coerce_optional_text(data.get("results")),
        contributions=_coerce_to_list(data.get("contributions", [])),
        limitations=_coerce_to_list(data.get("limitations", [])),
        relevance_reason=data.get("relevance_reason", f"与「{topic}」相关"),
        evidence_source=data.get("evidence_source", evidence_source),
        relation_type=data.get("relation_type"),
    )


def _coerce_to_list(value) -> list:
    """将 LLM 返回的标量/None/列表统一为 list，避免 pydantic 校验失败。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_optional_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        text = "; ".join(str(item).strip() for item in value if str(item).strip())
        return text or None
    text = str(value).strip()
    return text or None


def _infer_publication_profile(paper: Dict[str, Any], text: str = "") -> Dict[str, str]:
    """根据可验证元数据保守推断文献类型与证据等级。"""
    from app.tools.venue_tiers import is_conference_venue, is_preprint_record

    combined = f"{paper.get('title') or ''} {text or ''}".lower()
    source = str(paper.get("source") or "").lower()
    venue = str(paper.get("venue") or "").lower()
    doi = str(paper.get("doi") or "")
    # 预印本判据集中在 venue_tiers：平台自有 DOI 前缀充分；平台 venue（含 SSRN 的
    # 假刊名 "SSRN Electronic Journal"）与 arxiv_id 只在没有出版方 DOI 时充分——
    # 否则 CVPR / ACM MM / IJCV 的正式论文会因为同时挂着 arXiv 预印本而被全部标成
    # [EB/OL]。此处先算出唯一判决，同行评审状态复用它，不再各自看 source。
    preprint = is_preprint_record(
        venue=venue, doi=doi, source=source, arxiv_id=str(paper.get("arxiv_id") or ""),
    )
    if any(term in combined for term in ("systematic review", "系统综述", "系统评价")):
        return _publication_profile("systematic_review", preprint)
    if any(term in combined for term in ("meta-analysis", "meta analysis", "元分析")):
        return _publication_profile("meta_analysis", preprint)
    if preprint:
        return _publication_profile("preprint", preprint)
    if any(term in f"{venue} {combined}" for term in ("poster", "late-breaking", "late breaking", "short paper")):
        return _publication_profile("conference_short_paper", preprint)
    # 会议判定不能只看 "conference/proceedings" 字面词：S2 / OpenAlex 给
    # CVPR 的 venue 是裸名 "Computer Vision and Pattern Recognition"。
    if is_conference_venue(venue=venue, doi=doi):
        return _publication_profile("conference_paper", preprint)
    if paper.get("doi") or venue:
        return _publication_profile("journal_article", preprint)
    return _publication_profile("unknown", preprint)


def _publication_profile(publication_type: str, is_preprint: bool) -> Dict[str, str]:
    evidence_map = {
        "meta_analysis": "meta_analysis",
        "systematic_review": "systematic_review",
        "preprint": "preprint",
        "conference_paper": "conference_paper",
        "conference_short_paper": "conference_short_paper",
        "journal_article": "journal_article",
        "unknown": "unknown",
    }
    # 同行评审状态以"是否预印本"为准，不能以 source == "arxiv" 为准：source 只
    # 记录哪个接口返回了该记录，而 arXiv 接口对已正式发表的论文会回填出版方 DOI。
    # 用 source 定罪会让 CVPR / IJCV 的正式论文按未评审计入 global_evidence_gate
    # 的同行评审占比——与 [EB/OL] 错标同源，只是错在另一个字段。
    if is_preprint or publication_type == "preprint":
        review_status = "not_peer_reviewed"
    elif publication_type == "unknown":
        review_status = "unknown"
    else:
        review_status = "likely_peer_reviewed"
    return {
        "publication_type": publication_type,
        "peer_review_status": review_status,
        "evidence_level": evidence_map[publication_type],
    }


def _split_sentences_with_offsets(text: str) -> List[tuple[str, int, int]]:
    """中英文句子切分，同时保留原文字符位置。"""
    if not text:
        return []
    results: List[tuple[str, int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[。！？!?])\s*|(?<=[.;])\s+(?=[A-Z0-9])|\n+", text):
        end = match.start()
        sentence = text[start:end].strip()
        if len(sentence) >= 15:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            results.append((sentence, start + left_trim, start + left_trim + len(sentence)))
        start = match.end()
    sentence = text[start:].strip()
    if len(sentence) >= 15:
        left_trim = len(text[start:]) - len(text[start:].lstrip())
        results.append((sentence, start + left_trim, start + left_trim + len(sentence)))
    return results


def _evidence_tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    english = set(re.findall(r"[a-z][a-z0-9_-]{2,}", lowered))
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = {"".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))}
    return english | bigrams


def _match_score(query: str, evidence: str) -> float:
    query_tokens = _evidence_tokens(query)
    if not query_tokens:
        return 0.0
    evidence_tokens = _evidence_tokens(evidence)
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def _balanced_evidence_sample(
    candidates: List[Dict[str, Any]],
    *,
    limit: int,
    group_field: str,
) -> List[Dict[str, Any]]:
    """按页或章节轮询取样，避免证据预算被论文前部耗尽。"""
    if len(candidates) <= limit:
        return candidates
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for candidate in candidates:
        key = str(candidate.get(group_field) or "unknown")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(candidate)

    sampled: List[Dict[str, Any]] = []
    offset = 0
    while len(sampled) < limit:
        added = False
        for key in order:
            group = groups[key]
            if offset < len(group):
                sampled.append(group[offset])
                added = True
                if len(sampled) >= limit:
                    break
        if not added:
            break
        offset += 1
    return sampled


def _candidate_evidence(
    paper: Dict[str, Any],
    parsed_text: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    full_text_candidates: List[Dict[str, Any]] = []
    auxiliary_candidates: List[Dict[str, Any]] = []
    pages = (parsed_text or {}).get("pages") or []
    for page_data in pages:
        page_text = str(page_data.get("text") or "")
        for sentence, start, end in _split_sentences_with_offsets(page_text):
            if not _is_clean_evidence_sentence(sentence):
                continue
            full_text_candidates.append({
                "text": sentence,
                "section": "full_text",
                "page": page_data.get("page"),
                "source_type": "full_text",
                "char_start": start,
                "char_end": end,
                "confidence": 0.95,
            })

    if not full_text_candidates and parsed_text:
        sections = (parsed_text.get("sections") or {}) if parsed_text else {}
        for section_name, section_text in sections.items():
            if str(section_name).lower() == "references":
                continue
            for sentence, start, end in _split_sentences_with_offsets(str(section_text or "")):
                if not _is_clean_evidence_sentence(sentence):
                    continue
                full_text_candidates.append({
                    "text": sentence,
                    "section": str(section_name),
                    "page": None,
                    "source_type": "full_text",
                    "char_start": start,
                    "char_end": end,
                    "confidence": 0.9,
                })

        if not full_text_candidates and parsed_text.get("full_text"):
            full_text = str(parsed_text.get("full_text") or "")
            for sentence, start, end in _split_sentences_with_offsets(full_text):
                if not _is_clean_evidence_sentence(sentence):
                    continue
                full_text_candidates.append({
                    "text": sentence,
                    "section": "full_text",
                    "page": None,
                    "source_type": "full_text",
                    "char_start": start,
                    "char_end": end,
                    "confidence": 0.9,
                })

    abstract = _clean_abstract(str(paper.get("abstract") or ""))
    for sentence, start, end in _split_sentences_with_offsets(abstract):
        if not _is_clean_evidence_sentence(sentence):
            continue
        auxiliary_candidates.append({
            "text": sentence,
            "section": "abstract",
            "page": None,
            "source_type": "abstract",
            "char_start": start,
            "char_end": end,
            "confidence": 0.85,
        })

    title = str(paper.get("title") or "").strip()
    if title:
        auxiliary_candidates.append({
            "text": title,
            "section": "title",
            "page": None,
            "source_type": "title",
            "char_start": 0,
            "char_end": len(title),
            "confidence": 1.0,
            "provider": str(paper.get("source") or "") or None,
            "source_url": paper.get("url"),
            "_metadata_field": "title",
        })
    year = paper.get("year")
    venue = str(paper.get("venue") or "").strip()
    if year or venue:
        metadata_text = "；".join(
            part for part in (
                f"发表年份：{year}" if year else "",
                f"发表来源：{venue}" if venue else "",
            )
            if part
        )
        auxiliary_candidates.append({
            "text": metadata_text,
            "section": "metadata",
            "page": None,
            "source_type": "metadata",
            "char_start": None,
            "char_end": None,
            "confidence": 1.0,
            "provider": str(paper.get("source") or "") or None,
            "source_url": paper.get("url"),
            "_metadata_field": "publication",
        })
    # WHY: 书目事实必须与内容证据同等可追溯。作者、年份、来源、DOI 和出版
    # 状态各自保留独立 span，最终验收才能区分"检索层已核验"与"写作阶段推断"。
    for field, text in (
        ("authors", "作者：" + "、".join(
            str(author).strip() for author in (paper.get("authors") or []) if str(author).strip()
        ) if paper.get("authors") else ""),
        ("year", f"发表年份：{year}" if year else ""),
        ("venue", f"发表来源：{venue}" if venue else ""),
        ("doi", f"DOI：{paper.get('doi')}" if paper.get("doi") else ""),
        ("publication_status", f"出版状态：{paper.get('publication_status')}" if paper.get("publication_status") else ""),
    ):
        if not text:
            continue
        auxiliary_candidates.append({
            "text": text,
            "section": "metadata",
            "page": None,
            "source_type": "metadata",
            "char_start": None,
            "char_end": None,
            "confidence": 1.0,
            "provider": str(paper.get("source") or "") or None,
            "source_url": paper.get("url"),
            "_metadata_field": field,
        })
    # 为摘要、标题和元数据预留预算；全文按页（或章节）轮询取样。
    full_text_budget = max(0, 200 - len(auxiliary_candidates))
    group_field = "page" if pages else "section"
    sampled_full_text = _balanced_evidence_sample(
        full_text_candidates,
        limit=full_text_budget,
        group_field=group_field,
    )
    return [*sampled_full_text, *auxiliary_candidates][:200]


def _attach_evidence(
    card: PaperCard,
    paper: Dict[str, Any],
    parsed_text: Optional[Dict[str, Any]],
) -> PaperCard:
    """为结构化字段匹配原文证据，并生成稳定的 field_evidence 映射。"""
    candidates = _candidate_evidence(paper, parsed_text)
    evidence_state = resolve_evidence_permissions(paper, parsed_text)
    # WHY: 新路径只接受能按原坐标回读的摘录；不能再被 200 条词项候选采样挤掉。
    if card.extraction_coverage is not None:
        span_by_id = {span.evidence_id: span for span in card.evidence_spans}
        verified = []
        for field, ids in card.field_evidence.items():
            value = getattr(card, field, None)
            value_text = "\n".join(value) if isinstance(value, list) else str(value or "")
            for evidence_id in ids:
                span = span_by_id.get(evidence_id)
                if span is None or span.char_start is None or span.char_end is None:
                    continue
                if span.page is not None:
                    source = next((str(page.get("text") or "") for page in (parsed_text or {}).get("pages", [])
                                   if page.get("page") == span.page), "")
                else:
                    source = str((parsed_text or {}).get("full_text") or "")
                if source[span.char_start:span.char_end] != span.text or span.text not in value_text:
                    continue
                candidate = span.model_dump(exclude={"evidence_id"})
                candidate["_field"] = field
                verified.append(candidate)
        candidates = verified + candidates
    if not candidates:
        return card.model_copy(update={
            "evidence_state": evidence_state,
            "unsupported_fields": [
                "research_problem", "method", "results", "limitations", "dataset", "metrics"
            ],
        })

    field_values: Dict[str, str] = {
        "research_problem": card.research_problem,
        "method": card.method,
        "dataset": card.dataset or "",
        "metrics": " ".join(card.metrics),
        "results": card.results or "",
        "contributions": " ".join(card.contributions),
        "limitations": " ".join(card.limitations),
    }
    if card.extraction_coverage is not None:
        for name in ("study_design", "sample_size", "data_modalities", "behavior_categories"):
            value = getattr(card, name)
            field_values[name] = "\n".join(value) if isinstance(value, list) else str(value or "")
    selected_indexes: List[int] = []
    field_indexes: Dict[str, List[int]] = {}
    field_scores: Dict[str, Dict[int, float]] = {}
    for field, value in field_values.items():
        if not value:
            continue
        ranked = sorted(
            ((index, _match_score(value, item["text"])) for index, item in enumerate(candidates)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if card.extraction_coverage is not None:
            matches = [index for index, item in enumerate(candidates) if item.get("_field") == field]
            ranked = [(index, 1.0) for index in matches]
        else:
            matches = [index for index, score in ranked[:2] if score >= 0.08]
        if matches:
            field_indexes[field] = matches
            field_scores[field] = {index: score for index, score in ranked if index in matches}
            selected_indexes.extend(matches)

    metadata_indexes = [
        index for index, item in enumerate(candidates)
        if item.get("source_type") == "metadata"
    ]
    if metadata_indexes:
        field_indexes["publication"] = metadata_indexes[:1]
        selected_indexes.extend(metadata_indexes[:1])

    # WHY: 书目字段的 evidence 必须与内容字段一起进入 selected_indexes，
    # 否则 field_evidence 会指向未被保留的 span，validate_paper_card 判失败。
    bibliographic_indexes: Dict[str, List[int]] = {}
    for index, item in enumerate(candidates):
        field = str(item.get("_metadata_field") or "")
        if field and field != "publication":
            bibliographic_indexes.setdefault(field, []).append(index)
    for field, indexes in bibliographic_indexes.items():
        field_indexes[field] = indexes[:1]
        selected_indexes.extend(indexes[:1])

    if not selected_indexes:
        selected_indexes = [0]
    # WHY: 上限从 12 提高到 20，为新增的书目 evidence 留出预算；内容字段仍
    # 排在前面，因此不会被书目 span 挤出。
    selected_indexes = list(dict.fromkeys(selected_indexes))
    if card.extraction_coverage is None:
        selected_indexes = selected_indexes[:20]
    paper_id = str(card.paper_id or paper.get("paper_id") or "paper")
    index_to_id = {
        index: f"{paper_id}:e{position + 1:03d}"
        for position, index in enumerate(selected_indexes)
    }
    spans = [
        EvidenceSpan(
            evidence_id=index_to_id[index],
            **{
                key: value for key, value in candidates[index].items()
                if not key.startswith("_")
            },
        )
        for index in selected_indexes
    ]
    from app.schemas.retrieval_schema import fingerprint
    for span in spans:
        if span.source_type not in {"abstract", "full_text", "table"}:
            continue
        if span.source_type == "abstract":
            source = _clean_abstract(str(paper.get("abstract") or ""))
        elif span.page is not None:
            source = next((str(page.get("text") or "") for page in (parsed_text or {}).get("pages", [])
                           if page.get("page") == span.page), "")
        elif span.offset_basis == "full_text" or span.section == "full_text":
            source = str((parsed_text or {}).get("full_text") or "")
        else:
            source = str(((parsed_text or {}).get("sections") or {}).get(span.section) or "")
        anchored = (span.char_start is not None and span.char_end is not None
                    and 0 <= span.char_start < span.char_end <= len(source)
                    and source[span.char_start:span.char_end] == span.text)
        span.source_status = "source_verified" if anchored else "source_unverified"
        span.source_text_hash = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
        span.source_document_hash = fingerprint(source) if anchored else None
    field_evidence = {
        field: [index_to_id[index] for index in indexes if index in index_to_id]
        for field, indexes in field_indexes.items()
    }
    access_level = evidence_state.access_level
    field_claims: Dict[str, List[EvidenceClaim]] = {}
    verified_span_ids = {span.evidence_id for span in spans if span.source_status == "source_verified"}
    bibliographic_fields = {
        "publication", "title", "authors", "year", "venue", "doi", "publication_status",
    }
    for field, indexes in field_indexes.items():
        # WHY: 书目字段只提供可核验的元数据溯源，不构成可写入正文的内容主张，
        # 因此不进入 field_claims，避免 writer 把书目串当成研究结论。
        if field in bibliographic_fields:
            continue
        value = field_values.get(field, "")
        claims: List[EvidenceClaim] = []
        for index in indexes:
            if index not in index_to_id:
                continue
            candidate = candidates[index]
            score = float(field_scores.get(field, {}).get(index, 0.0))
            claim_value = str(candidate["text"]) if candidate.get("_field") else value
            explicit = bool(
                literal_support(claim_value,candidate["text"], fragment=field in {"dataset","metrics"})
                and index_to_id[index] in verified_span_ids
                and candidate.get("source_type") not in ("metadata", "title")
            )
            source_type = str(candidate.get("source_type") or "metadata")
            claim_level = {
                "metadata": AccessLevel.METADATA_ONLY,
                "title": AccessLevel.TITLE_AND_KEYWORDS,
                "abstract": AccessLevel.ABSTRACT,
            }.get(source_type, access_level)
            claims.append(EvidenceClaim(
                claim=claim_value,
                source_text=str(candidate["text"]),
                source_section=str(candidate.get("section") or candidate.get("source_type") or "unknown"),
                evidence_id=index_to_id[index],
                evidence_level=claim_level,
                confidence=max(0.0, min(1.0, score)),
                explicitly_reported=explicit,
            ))
        if claims:
            field_claims[field] = claims
    # 排序阶段的逐篇语义筛选是关系类型的单一事实来源。卡片抽取模型只
    # 负责论文内容，不能自行把一篇已判为间接相关的论文升级为直接证据。
    relation_type = _infer_relation_type(card, paper)
    return card.model_copy(update={
        "evidence_spans": spans,
        "field_evidence": field_evidence,
        "evidence_state": evidence_state,
        "field_claims": field_claims,
        "relation_type": relation_type,
        "anchor_low_confidence": bool(paper.get("_anchor_low_confidence")),
        "eligible_deliverables": [
            str(value) for value in paper.get("_eligible_deliverables") or []
            if str(value).strip()
        ],
    })


def _infer_relation_type(card: PaperCard, paper: Dict[str, Any]) -> str:
    """优先采用语义筛选关系；缺失时执行保守的确定性兜底。"""
    relation_type = str(paper.get("_topic_relation") or "").strip().lower()
    relation_type = {
        "method_related": "near",
        "topic_related": "near",
        "background": "indirect",
        "analogy": "indirect",
    }.get(relation_type, relation_type)
    if relation_type in {"direct", "near", "indirect", "unrelated"}:
        return relation_type

    relevance_score = paper.get("_llm_semantic_score")
    if not isinstance(relevance_score, (int, float)):
        relevance_score = paper.get("_relevance_score")
    if isinstance(relevance_score, (int, float)):
        if relevance_score >= 0.75:
            return "direct"
        if relevance_score >= 0.5:
            return "near"
    topic_match = re.search(r"「([^」]+)」", card.relevance_reason or "")
    topic_text = topic_match.group(1) if topic_match else card.relevance_reason
    paper_text = " ".join(
        str(value or "")
        for value in (
            paper.get("title"),
            paper.get("abstract"),
            card.research_problem,
            card.method,
        )
    )
    topic_tokens = _evidence_tokens(topic_text)
    if not topic_tokens:
        return "indirect"
    overlap = len(topic_tokens & _evidence_tokens(paper_text)) / len(topic_tokens)
    if overlap >= 0.5:
        return "direct"
    if overlap >= 0.25:
        return "near"
    return "indirect"


from app.core.json_utils import parse_json_object as _safe_parse_json  # noqa: E402
