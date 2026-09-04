"""主张语义聚簇：把各论文摘要句归并为跨文献论点。

根因回归：原始主张是各论文摘要的整句，措辞差异极大（实测同一路线内 873 对
同类型主张的 3-gram 相似度最大仅 0.125），而字面层合并要求前 80 字符全等，
因此几乎从不触发，support_level 恒为 single（实测 227/233）。
"""

import json

import pytest

from app.agent.claim_plan import (
    _MAX_THESIS_MEMBERS,
    _majority_claim_type,
    build_claim_plans,
    cluster_claims_into_theses,
)


class _ScriptedLLM:
    """按预设 payload 回应，并记录调用次数。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        payload = self.payload(prompt) if callable(self.payload) else self.payload
        return json.dumps(payload, ensure_ascii=False)


def _claims(count: int, *, one_paper: bool = False) -> dict:
    """构造 count 条字面层主张，默认每条来自不同论文。"""
    merged = {}
    for i in range(count):
        pid = "p0" if one_paper else f"p{i}"
        merged[f"k{i}"] = {
            "claim_text": f"第{i}条主张的具体内容陈述，来自某篇论文摘要",
            "claim_type": "finding",
            "evidence_ids": [f"{pid}:results{i}"],
            "paper_ids": [pid],
        }
    return merged


def _cards(count: int, *, one_paper: bool = False) -> dict:
    ids = ["p0"] if one_paper else [f"p{i}" for i in range(count)]
    return {
        pid: {"paper_id": pid, "authors": [f"作者{pid}"], "doi": f"10.1000/{pid}"}
        for pid in ids
    }


def _grouping(groups: list[list[int]]) -> dict:
    return {
        "theses": [
            {
                "thesis_text": "多篇文献共同支持的论点表述" if len(g) > 1 else "",
                "claim_type": "finding",
                "member_indices": g,
                "merge_basis": "同一论点" if len(g) > 1 else "独立论点",
            }
            for g in groups
        ]
    }


def _parse_claims_payload(prompt: str) -> list[dict]:
    """从提示词中取回 claims_json 数组。

    提示词尾部的 JSON 返回格式示例里也有方括号，用 rindex 会截到那里，
    所以按 claims_json 的起始位置做括号配平解析。
    """
    start = prompt.index("[")
    depth = 0
    for offset, char in enumerate(prompt[start:], start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(prompt[start : offset + 1])
    raise AssertionError("claims_json array not found in prompt")


# ---------- 正常聚簇 ----------

def test_cross_paper_claims_merge_into_one_thesis():
    """讲同一件事的多篇论文主张合并为一个论点，证据随之累积。"""
    merged = _claims(6)
    llm = _ScriptedLLM(_grouping([[0, 1, 2], [3], [4], [5]]))

    result = cluster_claims_into_theses(
        merged, route_name="学习参与度分析", topic="课堂行为分析",
        card_map=_cards(6), llm=llm,
    )

    assert llm.calls == 1
    assert len(result) == 4
    multi = [v for v in result.values() if v.get("thesis_member_count", 1) > 1]
    assert len(multi) == 1
    assert sorted(multi[0]["paper_ids"]) == ["p0", "p1", "p2"]
    assert len(multi[0]["evidence_ids"]) == 3
    assert multi[0]["claim_text"] == "多篇文献共同支持的论点表述"


def test_single_member_thesis_keeps_original_wording():
    """独立论点保留原文表述，不被 LLM 概括替换——原文更可核验。"""
    merged = _claims(4)
    original = merged["k3"]["claim_text"]
    result = cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(4),
        llm=_ScriptedLLM(_grouping([[0, 1], [2], [3]])),
    )

    assert any(v["claim_text"] == original for v in result.values())


# ---------- 确定性护栏 ----------

@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"theses": []}, "空分组"),
        ({"theses": [{"thesis_text": "x", "member_indices": [0, 1]}]}, "索引遗漏"),
        (
            {"theses": [
                {"thesis_text": "a", "member_indices": [0, 1]},
                {"thesis_text": "b", "member_indices": [1, 2, 3, 4, 5]},
            ]},
            "索引重复",
        ),
        (
            {"theses": [
                {"thesis_text": "a", "member_indices": [0, 99]},
                {"thesis_text": "b", "member_indices": [1, 2, 3, 4, 5]},
            ]},
            "索引越界",
        ),
        (
            {"theses": [{"thesis_text": "a", "member_indices": ["0", 1]}]},
            "索引非整数",
        ),
        ({"theses": [{"thesis_text": "a", "member_indices": []}]}, "空成员"),
        ({"theses": "not-a-list"}, "结构错误"),
    ],
)
def test_invalid_grouping_falls_back_to_input(payload, reason):
    """任一护栏不通过即整体回退，不接受部分可用的分组。"""
    merged = _claims(6)
    result = cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(6),
        llm=_ScriptedLLM(payload),
    )
    assert result is merged, reason


def test_whole_route_collapsed_into_one_thesis_is_rejected():
    """整条路线折叠成一个论点等于凭空制造共识，必须拒绝。"""
    merged = _claims(10)
    result = cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(10),
        llm=_ScriptedLLM(_grouping([list(range(10))])),
    )
    assert result is merged


def test_member_cap_scales_with_route_size():
    """成员上限随路线规模缩放：大路线允许更大论点，但不超过绝对上限。"""
    merged = _claims(20)
    # 9 个成员超过 _MAX_THESIS_MEMBERS，即使占比未过半也应拒绝。
    groups = [list(range(9))] + [[i] for i in range(9, 20)]
    result = cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(20),
        llm=_ScriptedLLM(_grouping(groups)),
    )
    assert result is merged
    assert _MAX_THESIS_MEMBERS == 8

    # 8 个成员刚好在上限内，应被接受。
    groups_ok = [list(range(8))] + [[i] for i in range(8, 20)]
    accepted = cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(20),
        llm=_ScriptedLLM(_grouping(groups_ok)),
    )
    assert accepted is not merged
    assert max(v.get("thesis_member_count", 1) for v in accepted.values()) == 8


def test_same_paper_members_do_not_forge_cross_paper_support():
    """同篇论文的多个字段主张合并后仍只有一篇论文，须保留原文而非概括。"""
    merged = {
        "a": {
            "claim_text": "该文提出的问题陈述",
            "claim_type": "problem",
            "evidence_ids": ["p0:research_problem"],
            "paper_ids": ["p0"],
        },
        "b": {
            "claim_text": "该文采用的方法陈述",
            "claim_type": "method_progression",
            "evidence_ids": ["p0:method"],
            "paper_ids": ["p0"],
        },
        "c": {
            "claim_text": "另一篇论文的发现陈述",
            "claim_type": "finding",
            "evidence_ids": ["p1:results"],
            "paper_ids": ["p1"],
        },
    }
    result = cluster_claims_into_theses(
        merged, route_name="R", topic="T",
        card_map={"p0": {"paper_id": "p0"}, "p1": {"paper_id": "p1"}},
        llm=_ScriptedLLM(_grouping([[0, 1], [2]])),
    )

    texts = {v["claim_text"] for v in result.values()}
    assert "该文提出的问题陈述" in texts
    assert "该文采用的方法陈述" in texts
    assert "多篇文献共同支持的论点表述" not in texts
    assert all(len(v["paper_ids"]) == 1 for v in result.values())


def test_no_merge_benefit_falls_back():
    """分组数等于输入数时没有合并收益，不用 LLM 概括覆盖原文。"""
    merged = _claims(5)
    result = cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(5),
        llm=_ScriptedLLM(_grouping([[0], [1], [2], [3], [4]])),
    )
    assert result is merged


# ---------- 回退与降级 ----------

def test_without_llm_returns_input_unchanged():
    merged = _claims(5)
    assert cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(5), llm=None,
    ) is merged


def test_llm_failure_falls_back_silently():
    class Boom:
        def complete(self, *a, **kw):
            raise RuntimeError("upstream unavailable")

    merged = _claims(5)
    assert cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(5), llm=Boom(),
    ) is merged


def test_single_claim_route_skips_llm():
    merged = _claims(1)
    llm = _ScriptedLLM(_grouping([[0]]))
    assert cluster_claims_into_theses(
        merged, route_name="R", topic="T", card_map=_cards(1), llm=llm,
    ) is merged
    assert llm.calls == 0


# ---------- claim_type 多数票 ----------

def test_majority_claim_type_picks_dominant():
    members = [
        {"claim_type": "method_progression"},
        {"claim_type": "method_progression"},
        {"claim_type": "finding"},
    ]
    assert _majority_claim_type(members) == "method_progression"


def test_majority_claim_type_ignores_llm_mixed_on_tie():
    """平票时取已知类型的首个成员，不采纳 LLM 的 mixed。"""
    members = [{"claim_type": "problem"}, {"claim_type": "finding"}]
    assert _majority_claim_type(members, fallback="mixed") == "problem"


# ---------- 端到端：支持级别真的提升 ----------

def _card(pid: str, problem: str, method: str) -> dict:
    return {
        "paper_id": pid,
        "title": f"论文 {pid}",
        "authors": [f"作者{pid}"],
        "doi": f"10.1000/{pid}",
        "year": 2025,
        "venue": "某期刊",
        "quality_status": "valid",
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract"},
        "research_problem": problem,
        "method": method,
        "field_claims": {
            "research_problem": [
                {"claim": problem, "evidence_id": f"{pid}:problem",
                 "explicitly_reported": True},
            ],
            "method": [
                {"claim": method, "evidence_id": f"{pid}:method",
                 "explicitly_reported": True},
            ],
        },
    }


def test_clustering_raises_support_level_above_single():
    """核心目的：语义聚簇后应出现 moderate/strong，而非全部 single。"""
    cards = [
        _card("p1", "课堂行为难以人工持续观察", "基于YOLOv8的课堂行为检测"),
        _card("p2", "教师难以及时掌握学习状态", "基于姿态估计的行为识别"),
        _card("p3", "传统观察覆盖不全", "深度学习课堂行为分类"),
        _card("p4", "评价缺乏客观依据", "多模态融合的行为分析"),
    ]
    routes = [{
        "route_id": "R1",
        "name": "视觉行为识别",
        "core_paper_ids": ["p1", "p2", "p3", "p4"],
        "paper_ids": ["p1", "p2", "p3", "p4"],
        "status": "KEEP",
    }]

    # 未启用聚簇：每条主张仅一篇论文证据。
    baseline = build_claim_plans(routes, cards, llm=None, topic="课堂行为分析")
    assert baseline
    assert baseline[0]["strong_plus_claims"] == 0
    assert baseline[0]["single_evidence_claims"] == baseline[0]["total_claims"]

    # 启用聚簇：把 4 篇的方法类主张归为一个论点。
    def payload(prompt: str) -> dict:
        data = _parse_claims_payload(prompt)
        method_idx = [
            item["claim_index"] for item in data
            if item["claim_type"] == "method_progression"
        ]
        other_idx = [
            item["claim_index"] for item in data
            if item["claim_index"] not in method_idx
        ]
        return _grouping([method_idx] + [[i] for i in other_idx])

    clustered = build_claim_plans(
        routes, cards, llm=_ScriptedLLM(payload), topic="课堂行为分析",
    )

    assert clustered
    # 4 篇方法类主张合并为一个论点后拿到 4 个独立来源，达到 strong。
    assert clustered[0]["strong_plus_claims"] >= 1
    strong = [
        c for c in clustered[0]["claims"]
        if c["support_level"] in ("strong", "established")
    ]
    assert strong[0]["independent_source_count"] == 4
    # 单证据主张占比从 100% 降下来，这是本次改动的核心目的。
    baseline_ratio = (
        baseline[0]["single_evidence_claims"] / baseline[0]["total_claims"]
    )
    clustered_ratio = (
        clustered[0]["single_evidence_claims"] / clustered[0]["total_claims"]
    )
    assert baseline_ratio == 1.0
    assert clustered_ratio < baseline_ratio
