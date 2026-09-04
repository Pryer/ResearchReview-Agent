"""论文文本匹配原语。

词法过滤、相关性打分与筛选协议共用的术语匹配器：兼容美式/英式拼写、
连字符/空格变体、精确单词边界与 CJK 连续词组。纯函数，无 IO、无 LLM。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


_SCOPE_STOPWORDS = {
    "about", "analysis", "approach", "based", "research", "study", "studies",
    "method", "methods", "system", "using", "with", "from", "into", "and", "the",
}


def _as_mapping(value: Any) -> dict[str, Any]:
    """将普通字典/Pydantic 语义帧转换为只读编译输入。"""
    if isinstance(value, Mapping):
        return dict(value)
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            return dict(dumper(mode="json"))
        except TypeError:
            return dict(dumper())
    return {}


def _clean_scope_terms(values: Any) -> list[str]:
    """稳定去重术语，拒绝空值但不猜测缺失元数据。"""
    if isinstance(values, str):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values if isinstance(values, Sequence) else []:
        if isinstance(value, Mapping):
            value = value.get("term") or value.get("label") or value.get("surface_text") or value.get("name")
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _semantic_item_terms(item: Any) -> list[str]:
    data = _as_mapping(item)
    return _clean_scope_terms([
        data.get("surface_text"), data.get("label"), data.get("name"),
        *(data.get("aliases") or []), *(data.get("context_aliases") or []),
    ])


def _canonicalize_for_fingerprint(value: Any) -> Any:
    """为 fingerprint 生成与输入列表顺序无关的 JSON 结构。"""
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_for_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        normalized = [_canonicalize_for_fingerprint(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    return value


def _scope_group(group_id: str, role: str, source: str, terms: Any, **extra: Any) -> dict[str, Any] | None:
    aliases = _clean_scope_terms(terms)
    if not aliases:
        return None
    languages = {
        "zh": [term for term in aliases if re.search(r"[\u4e00-\u9fff]", term)],
        "en": [term for term in aliases if not re.search(r"[\u4e00-\u9fff]", term)],
    }
    return {
        "group_id": str(group_id), "role": str(role), "source": str(source),
        "aliases": aliases, "languages": languages,
        "tokens": {
            "en": sorted(set(token for term in languages["en"] for token in normalized_latin_tokens(term))),
            "zh": sorted(set(re.findall(r"[一-鿿]+", " ".join(languages["zh"])))),
        },
        **extra,
    }


def compile_scope(
    selected_scope: Mapping[str, Any] | None = None,
    semantic_frame: Any = None,
    screening_protocol: Mapping[str, Any] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    topic_anchors: Sequence[Sequence[str]] | None = None,
    search_branches: Sequence[Mapping[str, Any]] | None = None,
    excluded_title_terms: Sequence[str] | None = None,
    topic: str = "",
    research_mode: str = "",
) -> dict[str, Any]:
    """编译所有范围来源为稳定、可审计的匹配上下文。

    该函数只做确定性规范化，不执行检索或语义推断。``groups`` 保存来源/角色
    与双语 aliases，``tokens`` 保存可复用的英文词元和 CJK 短语；fingerprint
    由规范化内容计算，因此同一范围在初筛和详情复核中必然一致。
    """
    scope = _as_mapping(selected_scope)
    frame = _as_mapping(semantic_frame)
    protocol = _as_mapping(screening_protocol)
    groups: list[dict[str, Any]] = []

    def add(group_id: str, role: str, source: str, terms: Any, **extra: Any) -> None:
        item = _scope_group(group_id, role, source, terms, **extra)
        if item:
            groups.append(item)

    add("scope_include", "scope_include", "selected_scope", scope.get("include_terms"))
    add("scope_exclude", "scope_exclude", "selected_scope", scope.get("exclude_terms"))
    for index, branch in enumerate(scope.get("branches") or []):
        branch_data = _as_mapping(branch)
        add(
            f"scope_branch_{index}", "scope_branch", "selected_scope",
            branch_data.get("seed_queries"), label=branch_data.get("label"),
            scope_id=branch_data.get("scope_id"),
            seed_queries=_clean_scope_terms(branch_data.get("seed_queries")),
        )
    add("scope_seed", "scope_seed", "selected_scope", scope.get("seed_queries"))
    for index, branch in enumerate(search_branches or []):
        branch_data = _as_mapping(branch)
        add(
            f"search_branch_{index}", "search_branch", "search_branches",
            [*(branch_data.get("queries") or []),
             *(term for group in (branch_data.get("required_concepts") or []) for term in (group or []))],
            branch_type=branch_data.get("branch_type"),
            constraint_level=branch_data.get("constraint_level"),
            required_concepts=[
                _clean_scope_terms(group)
                for group in (branch_data.get("required_concepts") or [])
                if _clean_scope_terms(group)
            ],
        )

    for role, key in (("object", "research_objects"), ("method", "methods"),
                      ("context", "application_domains"), ("context", "analysis_targets")):
        for index, item in enumerate(frame.get(key) or []):
            add(f"semantic_{key}_{index}", role, "semantic_frame", _semantic_item_terms(item))
    add("semantic_focus", "focus", "semantic_frame", frame.get("required_focuses"))
    for index, item in enumerate(frame.get("evidence_requirements") or []):
        data = _as_mapping(item)
        add(f"evidence_requirement_{index}", "requirement", "semantic_frame",
            [data.get("label"), *(data.get("aliases") or []), *(data.get("context_aliases") or [])],
            requirement_id=data.get("requirement_id"), evidence_role=data.get("evidence_role"))

    # Hard/soft protocol terms remain grouped so language_filter can select the
    # language-specific aliases without reinterpreting the protocol.
    for kind in ("hard_include_criteria", "soft_include_criteria"):
        for index, criterion in enumerate(protocol.get(kind) or []):
            data = _as_mapping(criterion)
            add(
                f"protocol_{kind}_{index}", kind, "screening_protocol",
                [*(data.get("terms") or []), *(data.get("terms_zh") or []), *(data.get("terms_en") or [])],
                terms_by_language={
                    "zh": _clean_scope_terms(data.get("terms_zh")),
                    "en": _clean_scope_terms(data.get("terms_en")),
                }, applies_to_each_paper=bool(data.get("applies_to_each_paper", True)),
                source_name=data.get("source"), label=data.get("label"),
            )
    for index, route in enumerate(protocol.get("routes") or []):
        data = _as_mapping(route)
        add(
            f"protocol_route_{index}", "route", "screening_protocol",
            [*(data.get("terms") or []), *(data.get("terms_zh") or []), *(data.get("terms_en") or [])],
            route_id=data.get("route_id"), label=data.get("label"),
        )
    add("protocol_exclude", "protocol_exclude", "screening_protocol", protocol.get("hard_exclude_title_terms"))
    add("title_exclude", "title_exclude", "ranking_input", excluded_title_terms)

    concept_groups = [
        ("topic_anchors", group) for group in (topic_anchors or [])
    ] + [
        ("required_concepts", group) for group in (required_concepts or [])
    ]
    seen_concept_groups: set[tuple[str, ...]] = set()
    for index, (source_name, concept_group) in enumerate(concept_groups):
        normalized_group = tuple(sorted(
            term.casefold() for term in _clean_scope_terms(concept_group)
        ))
        if not normalized_group or normalized_group in seen_concept_groups:
            continue
        seen_concept_groups.add(normalized_group)
        add(f"topic_anchor_{index}", "topic_anchor", source_name, concept_group)
    if topic:
        add("topic", "topic", "topic", [topic])

    group_by_role: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        group_by_role.setdefault(group["role"], []).append(group)
    aliases: dict[str, list[str]] = {}
    tokens: dict[str, list[str]] = {"en": [], "zh": []}
    for role, role_groups in group_by_role.items():
        aliases[role] = list(dict.fromkeys(term for group in role_groups for term in group["aliases"]))
    for group in groups:
        for language in ("en", "zh"):
            tokens[language].extend(group["tokens"][language])
    tokens = {language: sorted(set(values)) for language, values in tokens.items()}
    canonical = {
        "schema": "compiled_scope.v1", "research_mode": str(research_mode or frame.get("research_mode") or ""),
        "scope": scope, "groups": groups, "aliases": aliases, "tokens": tokens,
    }
    fingerprint = hashlib.sha256(json.dumps(
        _canonicalize_for_fingerprint(canonical),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "version": "compiled_scope.v1", "fingerprint": fingerprint,
        "scope": scope, "groups": groups, "by_role": group_by_role,
        "aliases": aliases, "tokens": tokens,
        "include_terms": aliases.get("scope_include", []),
        "exclude_terms": aliases.get("scope_exclude", []),
        "seed_queries": aliases.get("scope_seed", []),
        "branches": [group for group in groups if group["role"] == "scope_branch"],
        "protocol": protocol, "research_mode": canonical["research_mode"],
    }


def _latin_stem(token: str) -> str:
    """返回仅用于学术检索匹配的保守英文词干。

    不引入完整 NLP 词形还原依赖，只处理本项目检索式中常见的复数和
    ``interaction/interactive`` 一类派生变化。短词和 ``analysis`` 等
    易被误截断的词保持原样。
    """
    value = str(token or "").casefold()
    if len(value) <= 4 or value.endswith(("ss", "us", "is")):
        return value
    if value.endswith("ies") and len(value) > 5:
        return value[:-3] + "y"
    if value.endswith("ations") and len(value) > 8:
        return value[:-6]
    if value.endswith("ation") and len(value) > 7:
        return value[:-5]
    if value.endswith("ions") and len(value) > 6:
        return value[:-4]
    if value.endswith("ion") and len(value) > 5:
        return value[:-3]
    if value.endswith("ives") and len(value) > 6:
        return value[:-4]
    if value.endswith("ive") and len(value) > 5:
        return value[:-3]
    if value.endswith("ing") and len(value) > 6:
        return value[:-3]
    if value.endswith("ed") and len(value) > 5:
        return value[:-2]
    if value.endswith("es") and len(value) > 5:
        return value[:-2]
    if value.endswith("s") and len(value) > 4:
        return value[:-1]
    return value


def normalized_latin_tokens(text: str) -> list[str]:
    """提取顺序稳定的英文规范词元，供所有范围与锚点规则共用。"""
    normalized = re.sub(
        r"behaviour", "behavior",
        re.sub(r"[-_/]+", " ", str(text or "").casefold()),
    )
    return [_latin_stem(token) for token in re.findall(r"[a-z0-9]+", normalized)]


def term_matches_haystack(term: str, haystack: str) -> bool:
    """匹配同义术语，兼容美式/英式拼写（behavior/behaviour）、精确单词边界与 CJK 切词。"""
    term_clean = str(term or "").strip().lower()
    if not term_clean:
        return False
    haystack_clean = re.sub(
        r"behaviour", "behavior",
        re.sub(r"[-_/]+", " ", str(haystack or "").lower())
    )
    term_clean = re.sub(r"behaviour", "behavior", re.sub(r"[-_/]+", " ", term_clean))

    # 1. 英文全短语精确匹配（使用单词边界 \b 防止 action 误匹配 interaction）
    raw_tokens = re.findall(r"[a-z0-9]+", term_clean)
    tokens = normalized_latin_tokens(term_clean)
    if raw_tokens:
        phrase_pattern = r"\b" + r"\s+".join(re.escape(t) for t in raw_tokens) + r"\b"
        if re.search(phrase_pattern, haystack_clean):
            return True

    # 2. CJK 中文短语匹配。带空格的检索式按连续中文词组拆分，
    #    避免旧实现逐字切分后把所有单字丢弃，导致中文组合词永远无法命中。
    cjk_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", term_clean)
    if len(cjk_phrases) >= 2:
        matched_cjk = sum(phrase in haystack_clean for phrase in cjk_phrases)
        if matched_cjk / len(cjk_phrases) >= 0.75:
            return True
    elif cjk_phrases:
        if term_clean in haystack_clean:
            return True

    # 3. 英文多词组词汇共现匹配（例如 "few shot action recognition" 要求 75% 单词以独立单词形式出现）
    if len(tokens) >= 2:
        haystack_tokens = set(normalized_latin_tokens(haystack_clean))
        matched = sum(token in haystack_tokens for token in tokens)
        return matched / len(tokens) >= 0.75

    return False


def hard_anchor_matches_haystack(term: str, haystack: str) -> bool:
    """使用通用短语规则匹配由 LLM 协议提供的逐篇核心对象。"""
    return term_matches_haystack(term, haystack)


def excluded_term_matches_title(term: str, title: str) -> bool:
    """排除词只做严格短语匹配，避免 zero-shot 误杀 few-shot。"""
    term_clean = str(term or "").strip().lower()
    if not term_clean:
        return False
    normalized_title = re.sub(r"[-_/]+", " ", str(title or "").lower())
    normalized_term = re.sub(r"[-_/]+", " ", term_clean)
    tokens = re.findall(r"[a-z0-9]+", normalized_term)
    if tokens:
        phrase_pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
        return bool(re.search(phrase_pattern, normalized_title))
    return normalized_term in normalized_title


def matches_seed_context(haystack: str, seed_queries: list[str]) -> tuple[bool, str]:
    """要求种子检索式的大部分语义成分共同出现。"""
    for query in seed_queries:
        cjk_phrases = [
            part for part in re.split(r"\s+", str(query).strip())
            if re.search(r"[\u4e00-\u9fff]", part)
        ]
        if cjk_phrases:
            matched = sum(term_matches_haystack(part, haystack) for part in cjk_phrases)
            required = len(cjk_phrases) if len(cjk_phrases) <= 2 else max(2, int(len(cjk_phrases) * 0.6 + 0.5))
            if matched >= required:
                return True, f"命中范围种子语境 ({matched}/{len(cjk_phrases)})"
            continue
        tokens = [
            token for token in re.findall(r"[a-z][a-z0-9-]{2,}", query.lower())
            if token not in _SCOPE_STOPWORDS
        ]
        if not tokens:
            continue
        matched = sum(term_matches_haystack(token, haystack) for token in tokens)
        required = 1 if len(tokens) == 1 else max(2, int(len(tokens) * 0.6 + 0.5))
        if matched >= required:
            return True, f"命中范围种子语境 ({matched}/{len(tokens)})"
    return False, "未命中范围种子语境"
