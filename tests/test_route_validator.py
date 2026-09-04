"""Focused tests for Route Validator v2 integration details."""

from app.agent.route_validator import merge_weak_routes_for_writing, prepare_route_anchors


class AnchorExpansionLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        assert kwargs["operation"] == "expand_route_semantic_anchors"
        return """{
          "routes": [{
            "route_id": "R1",
            "anchor_expansions": [
              {
                "text": "temporal alignment",
                "anchor_type": "semantic",
                "supports": "时序对齐"
              },
              {
                "text": "few-shot video recognition",
                "anchor_type": "task",
                "supports": "少样本视频识别"
              }
            ]
          }]
        }"""


def _route() -> dict:
    return {
        "route_id": "R1",
        "name": "基于时序对齐的识别方法",
        "research_question": "如何通过时序对齐完成少样本视频识别？",
        "route_role": "interpretation",
        "core_concepts": ["时序对齐", "少样本视频识别", "支持集与查询集匹配"],
        "semantic_anchors": [],
        "method_concepts": [],
        "task_anchors": [],
        "negative_anchors": ["fully supervised localization"],
        "search_queries": ["temporal alignment few-shot video recognition"],
        "inclusion_criteria": ["使用时序对齐处理支持集与查询集"],
        "exclusion_criteria": ["排除全监督动作定位"],
        "boundary_note": "必须包含少样本 support-query 设置。",
    }


def test_missing_bilingual_anchors_are_expanded_once_and_persisted():
    llm = AnchorExpansionLLM()

    first, reports = prepare_route_anchors([_route()], llm=llm)
    second, _ = prepare_route_anchors(first, llm=llm)

    assert llm.calls == 1
    assert "temporal alignment" in first[0]["semantic_anchors"]
    assert first[0]["anchor_expansions"] == []
    assert reports["R1"]["accepted"][0]["supports"] == "时序对齐"
    assert second[0]["semantic_anchors"] == first[0]["semantic_anchors"]


def test_route_paper_assignments_have_unique_primary_theme(monkeypatch):
    """跨路线成员不得获得多个 primary_theme_id。

    回归 2026-08-22 实测缺陷：validated_routes 的 paper_ids 含 cross_route
    成员，逐路线写入 assignments 会让同一论文在每条路线各得一个
    "primary" 归属，synthesize_themes 按该字段聚合后主题互相吞并
    （实测 5 个主题各 72-81 篇、两两重叠 65-77 篇），主题节授权面铺满
    全池并造成跨小节重复引用。
    """
    from app.agent.nodes.synthesis import validate_routes_node
    from app.tools.synthesize_themes import synthesize_themes

    # p_shared 同时出现在两条路线（cross_route 成员），且其逐篇最佳匹配为 R2
    fake_result = {
        "validated_routes": [
            {
                "route_id": "R1", "name": "行为编码与观察",
                "research_question": "如何自动完成课堂行为编码",
                "core_paper_ids": ["p_a"],
                "paper_ids": ["p_a", "p_shared"],
            },
            {
                "route_id": "R2", "name": "互动序列分析",
                "research_question": "如何分析师生互动序列",
                "core_paper_ids": ["p_b", "p_shared"],
                "paper_ids": ["p_b", "p_shared"],
            },
        ],
        "decisions": [],
        "coverage": {},
        "assignment_map": {
            "p_a": {"type": "single_route", "primary_route": "R1"},
            "p_b": {"type": "single_route", "primary_route": "R2"},
            "p_shared": {"type": "cross_route", "primary_route": "R2"},
        },
    }
    monkeypatch.setattr(
        "app.agent.provisional_routes.validate_routes_against_evidence",
        lambda *a, **k: fake_result,
    )

    cards = [
        {"paper_id": pid, "title": f"Paper {pid}", "quality_status": "valid",
         "research_problem": "课堂行为编码成本高", "method": "自动编码"}
        for pid in ("p_a", "p_b", "p_shared")
    ]
    state = {
        "paper_cards": cards,
        "provisional_framework": {"provisional_routes": [{"route_id": "R1"}, {"route_id": "R2"}]},
    }

    validate_routes_node(state, llm=None)

    assignments = (state.get("dynamic_taxonomy") or {}).get("assignments") or []
    owners = {}
    for item in assignments:
        pid = str(item["paper_id"])
        assert pid not in owners, f"{pid} 出现多条 primary 归属"
        owners[pid] = item["primary_theme_id"]
    # 覆盖率不因去重下降：三篇论文全部有归属
    assert set(owners) == {"p_a", "p_b", "p_shared"}
    # 归属遵循 assignment_map 的逐篇最佳匹配，而非路线书写顺序
    theme_by_name = {
        t["name"]: t["theme_id"] for t in (state["dynamic_taxonomy"]["themes"] or [])
    }
    assert owners["p_shared"] == theme_by_name["互动序列分析"]

    # 主题综合层随之互斥：任意两个主题的论文集合不再重叠
    syntheses = synthesize_themes(cards, state["dynamic_taxonomy"])
    id_sets = [{str(x) for x in s.get("paper_ids") or []} for s in syntheses]
    for left in range(len(id_sets)):
        for right in range(left + 1, len(id_sets)):
            assert not (id_sets[left] & id_sets[right])


def _behavior_card(paper_id: str, topic_text: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": topic_text,
        "abstract": topic_text,
        "research_problem": topic_text,
        "method": topic_text,
        "quality_status": "valid",
    }


def test_oversized_route_is_split_into_balanced_sub_routes():
    """独占成员份额过高的路线按证据子聚类拆分，而不是坍缩成巨型小节。

    回归 2026-08-22 缺陷：诊断只有下限（core >= N 即 STRONG_ROUTE），
    一条独占半数文献的路线也被判健康，写作层随之出现 51 篇挤一节。
    """
    from app.agent.route_validator import _split_oversized_routes

    dominant = [f"d{i}" for i in range(1, 13)]
    small = ["s1", "s2"]
    cards = {
        **{
            pid: _behavior_card(
                pid,
                "课堂行为编码 coding scheme 时间窗 annotation"
                if index % 2 == 0
                else "师生互动序列 interaction sequence 滞后分析 lag",
            )
            for index, pid in enumerate(dominant)
        },
        **{pid: _behavior_card(pid, "学习投入度 engagement 量表") for pid in small},
    }
    routes = [
        {"route_id": "R1", "name": "课堂行为分析",
         "core_paper_ids": list(dominant), "supporting_paper_ids": [],
         "paper_ids": list(dominant)},
        {"route_id": "R2", "name": "学习投入度",
         "core_paper_ids": list(small), "supporting_paper_ids": [],
         "paper_ids": list(small)},
    ]
    primary_owner = {**{pid: "R1" for pid in dominant}, **{pid: "R2" for pid in small}}
    decisions: list[dict] = []

    reassigned = _split_oversized_routes(routes, decisions, cards, primary_owner)

    # R1 被替换为多条子路线，R2 保持原状
    route_ids = [route["route_id"] for route in routes]
    assert "R1" not in route_ids
    assert "R2" in route_ids
    sub_ids = [rid for rid in route_ids if rid.startswith("R1_S")]
    assert 2 <= len(sub_ids) <= 3

    # 子路线规模相近，不出现“一条吞掉全部”的退化划分
    sizes = [
        len(route["core_paper_ids"]) for route in routes
        if route["route_id"].startswith("R1_S")
    ]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == len(dominant)  # 证据不流失

    # 每篇论文只归入一条子路线，供写作层作为唯一主题归属
    assert set(reassigned) == set(dominant)
    assert len(set(reassigned.values())) == len(sub_ids)

    split_decisions = [d for d in decisions if d.get("action") == "SPLIT_INTO"]
    assert len(split_decisions) == 1
    assert split_decisions[0]["diagnosis"] == "OVERSIZED_ROUTE"
    assert split_decisions[0]["route_id"] == "R1"


def test_balanced_routes_are_not_split():
    """规模均衡的路线不得被误拆——判据是相对份额，不是绝对篇数。"""
    from app.agent.route_validator import _split_oversized_routes

    group_a = [f"a{i}" for i in range(1, 9)]
    group_b = [f"b{i}" for i in range(1, 9)]
    cards = {
        **{pid: _behavior_card(pid, "课堂行为编码 coding") for pid in group_a},
        **{pid: _behavior_card(pid, "师生互动序列 interaction") for pid in group_b},
    }
    routes = [
        {"route_id": "R1", "name": "行为编码", "core_paper_ids": list(group_a),
         "supporting_paper_ids": [], "paper_ids": list(group_a)},
        {"route_id": "R2", "name": "互动序列", "core_paper_ids": list(group_b),
         "supporting_paper_ids": [], "paper_ids": list(group_b)},
    ]
    primary_owner = {
        **{pid: "R1" for pid in group_a}, **{pid: "R2" for pid in group_b},
    }
    decisions: list[dict] = []

    reassigned = _split_oversized_routes(routes, decisions, cards, primary_owner)

    assert reassigned == {}
    assert [route["route_id"] for route in routes] == ["R1", "R2"]
    assert not [d for d in decisions if d.get("action") == "SPLIT_INTO"]


def test_sub_route_name_rejects_topic_restating_and_setting_drift():
    """子路线名必须是区分点：复述主题或改写任务设定的名称一律拒收。

    回归 2026-08-22 会话：LLM 在无上下文时产出「跨模态小样本视频动作识别」
    「时空对齐的零样本动作识别」——把主题整句搬进小节标题，并把主题的
    少样本设定擅自改成零样本。判据是"不得包含主题核心词"，因此不需要
    维护任何设定词对照表。
    """
    from app.agent.route_validator import _is_valid_sub_route_name

    topic = "少样本动作识别"
    for bad in (
        "跨模态小样本视频动作识别",   # 复述主题
        "少样本骨架动作识别研究",     # 复述主题
        "时空对齐的零样本动作识别",   # 复述主题 + 设定漂移
        "",
    ):
        assert not _is_valid_sub_route_name(bad, topic=topic, sibling_names=[])

    for good in ("跨模态原型匹配", "骨架图结构建模", "时序对齐与节奏"):
        assert _is_valid_sub_route_name(good, topic=topic, sibling_names=[])

    # 与同级重名同样拒收
    assert not _is_valid_sub_route_name(
        "跨模态原型匹配", topic=topic, sibling_names=["跨模态原型匹配"],
    )


def test_deterministic_sub_route_labels_are_distinctive_and_unique():
    """兜底名取"相对其他子簇最具区分度"的词项，不得选中各簇共有的泛词。

    仅按簇内词频会让三条子路线都叫 recognition（各簇共有），因此改用
    簇内占比减簇外占比排序。
    """
    from app.agent.route_validator import _distinctive_cluster_labels

    def card(pid: str, text: str) -> dict:
        return {"paper_id": pid, "title": text, "research_problem": text, "method": text}

    card_map = {}
    clusters = []
    for prefix, distinctive in (("s", "skeleton graph"), ("m", "metric prototype"), ("t", "temporal alignment")):
        ids = [f"{prefix}{i}" for i in range(1, 5)]
        for pid in ids:
            # recognition 为三簇共有的泛词，必须不被选中
            card_map[pid] = card(pid, f"few-shot action recognition with {distinctive}")
        clusters.append(ids)

    labels = _distinctive_cluster_labels(
        clusters, card_map, parent_name="度量学习与匹配", topic="少样本动作识别",
    )

    assert len(labels) == 3
    assert len(set(labels)) == 3          # 互不重复
    assert all("recognition" not in item for item in labels)
    assert all(item.startswith("度量学习与匹配") for item in labels)


def test_name_overlap_detects_cross_stage_concept_sharing():
    """跨阶段的名称概念重叠判定：与既有路线共享概念即拒收。

    回归 2026-08-22 会话：拆分子路线被命名为「跨模态原型匹配」，而检索前
    生成的既有路线已叫「多模态与自监督」——两者共享「模态」概念，正文出
    现两个读者无法区分的小节。判据为纯词法（剥 1 修饰字取公共前缀、词级
    包含、≥3 字子串），不维护同义词表。
    """
    from app.agent.route_validator import _name_overlaps

    overlaps = [
        ("跨模态原型匹配", "多模态与自监督"),
        ("跨模态匹配", "多模态与自监督"),
        ("多模态原型融合", "多模态与自监督"),
        ("自监督对比预训练", "多模态与自监督"),
        ("骨架数据增强", "数据增强与生成"),
        ("时序对齐建模", "时序对齐的零样本动作识别"),
    ]
    distinct = [
        ("骨架图结构建模", "多模态与自监督"),
        ("时序对齐与节奏", "多模态与自监督"),
        ("迁移学习方法", "度量学习与匹配"),   # 共用领域后缀但机制不同
        ("原型度量", "度量学习与匹配"),
        ("temporal alignment", "metric matching"),
    ]
    for left, right in overlaps:
        assert _name_overlaps(left, right), f"{left} 应与 {right} 判重叠"
    for left, right in distinct:
        assert not _name_overlaps(left, right), f"{left} 不应与 {right} 判重叠"


def test_sub_route_name_rejected_when_overlapping_existing_route():
    """与既有（非同级）路线重叠的子路线名必须拒收并退回兜底名。"""
    from app.agent.route_validator import _is_valid_sub_route_name

    reserved = ["多模态与自监督", "跨域与泛化"]
    valid = _is_valid_sub_route_name(
        "跨模态原型匹配", topic="少样本动作识别",
        sibling_names=[], reserved_names=reserved,
    )
    assert valid is False

    # 与既有路线无概念共享的名称正常通过
    ok = _is_valid_sub_route_name(
        "骨架图结构建模", topic="少样本动作识别",
        sibling_names=[], reserved_names=reserved,
    )
    assert ok is True


def test_name_overlap_catches_tail_shared_concepts():
    """规则 4 扫描窗口必须覆盖左串全部起始位置。

    回归：窗口曾按短串长度截断（min(len)-2），位于左串后部的共享概念
    漏检——「跨模态原型时序对齐」与「多视角时序对齐」共享的「时序对」
    在第 5 位，两个同概念子路线名同时存活。
    """
    from app.agent.route_validator import _name_overlaps

    assert _name_overlaps("跨模态原型时序对齐", "多视角时序对齐")   # 共享"时序对齐"
    # 仅共享 2 字领域词（时序）不判重叠，与"度量学习/迁移学习"口径一致
    assert not _name_overlaps("混合关系时序匹配", "多视角时序对齐")
    # 前部共享的既有用例保持不回归
    assert _name_overlaps("时序对齐建模", "时序对齐的零样本动作识别")
    assert not _name_overlaps("迁移学习方法", "度量学习与匹配")


def test_weak_routes_are_merged_before_writing_without_mutating_validation():
    routes = [
        {
            "route_id": "R1", "name": "时序对齐与匹配", "status": "KEEP",
            "core_concepts": ["时序对齐", "原型匹配"],
            "paper_ids": ["p1", "p2"], "core_paper_ids": ["p1", "p2"],
        },
        {
            "route_id": "R2", "name": "时序原型技术", "status": "WEAK",
            "core_concepts": ["时序对齐", "原型"],
            "paper_ids": ["p3"], "core_paper_ids": ["p3"],
        },
        {
            "route_id": "R3", "name": "跨域泛化", "status": "KEEP",
            "core_concepts": ["领域适应", "泛化"],
            "paper_ids": ["p4", "p5"], "core_paper_ids": ["p4", "p5"],
        },
    ]

    writing_routes = merge_weak_routes_for_writing(routes)

    assert [route["route_id"] for route in writing_routes] == ["R1", "R3"]
    assert writing_routes[0]["paper_ids"] == ["p1", "p2", "p3"]
    assert writing_routes[0]["merged_route_ids"] == ["R2"]
    assert routes[0]["paper_ids"] == ["p1", "p2"]
    assert routes[1]["status"] == "WEAK"


def _route_with(route_id: str, name: str, question: str, concepts: list[str]) -> dict:
    return {
        "route_id": route_id,
        "name": name,
        "research_question": question,
        "core_concepts": concepts,
        "semantic_anchors": concepts,
        "method_concepts": [],
        "task_anchors": concepts,
        "negative_anchors": [],
        "inclusion_criteria": [question],
        "exclusion_criteria": [],
    }


def test_overlapping_core_members_go_to_the_best_fitting_route():
    """core 成员在多条路线重叠时，独占归属按契合度而非路线书写顺序分配。

    回归 2026-08-29 实测缺陷：``primary_owner`` 取 ``core_routes[0]``，60 篇
    证据里 42 篇同时是 5 条路线的 core，于是首条路线独吞 42 篇，其余 4 条
    各剩 1-4 篇，写作层四个小节同时退化为「本节纳入 1 篇文献」的罗列。
    """
    from app.agent.provisional_routes import validate_routes_against_evidence

    routes = [
        _route_with(
            "R1", "课堂行为识别方法",
            "如何用视觉模型识别课堂行为？",
            ["课堂行为识别", "视觉模型", "行为检测"],
        ),
        _route_with(
            "R2", "师生互动序列分析",
            "如何分析师生互动的时序结构？",
            ["师生互动", "互动序列", "时序结构"],
        ),
    ]
    # 这篇论文同时命中两条路线的全部锚点（两侧都判 core），但正文主体是
    # 互动序列分析，识别结果只是输入——契合度更高的一侧应拿到 primary。
    shared_text = (
        "课堂行为识别 视觉模型 行为检测 师生互动 互动序列 时序结构 "
        "本研究以师生互动 互动序列 时序结构为核心，分析课堂师生互动的时序结构，"
        "并辅以课堂行为识别结果作为输入"
    )
    cards = [_behavior_card("p_shared", shared_text)]

    result = validate_routes_against_evidence(routes, cards, llm=None, topic="课堂行为分析")
    assignment = result["assignment_map"]["p_shared"]

    # 两条路线都判 core，说明这正是顺序取值会出错的场景
    assert assignment["type"] == "cross_route"
    # 归属落在契合度更高的 R2，而不是列表里的第一条 R1
    assert assignment["primary_route"] == "R2"
    assert assignment["secondary_routes"] == ["R1"]


def test_primary_owner_is_not_biased_by_route_order():
    """把路线顺序颠倒后，跨路线成员的独占归属保持不变。

    这是"归属不依赖书写顺序"的直接判据：只要匹配特征相同，同一篇论文
    必须落到同一条路线，而不是每次都落到列表里的第一条。
    """
    from app.agent.provisional_routes import validate_routes_against_evidence

    forward = [
        _route_with(
            "R1", "课堂行为识别方法",
            "如何用视觉模型识别课堂行为？",
            ["课堂行为识别", "视觉模型", "行为检测"],
        ),
        _route_with(
            "R2", "师生互动序列分析",
            "如何分析师生互动的时序结构？",
            ["师生互动", "互动序列", "时序结构"],
        ),
    ]
    cards = [_behavior_card(
        "p_shared",
        "课堂行为识别 视觉模型 行为检测 师生互动 互动序列 时序结构 "
        "本研究以师生互动 互动序列 时序结构为核心，分析课堂师生互动的时序结构，"
        "并辅以课堂行为识别结果作为输入",
    )]

    first = validate_routes_against_evidence(
        forward, cards, llm=None, topic="课堂行为分析",
    )["assignment_map"]["p_shared"]["primary_route"]
    second = validate_routes_against_evidence(
        list(reversed(forward)), cards, llm=None, topic="课堂行为分析",
    )["assignment_map"]["p_shared"]["primary_route"]

    assert first == second == "R2"


def _status_card(paper_id: str, text: str) -> dict:
    """带证据等级与显式声明的卡片：可进入写作计划的授权池。"""
    return {
        **_behavior_card(paper_id, text),
        "year": 2025,
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract"},
        "field_claims": {
            "research_problem": [{
                "claim": text[:40],
                "evidence_id": f"{paper_id}:e1",
                "explicitly_reported": True,
            }],
        },
    }


def test_unique_owner_chain_keeps_every_route_section_above_the_evidence_floor():
    """重叠证据的独占归属必须一路传到写作计划，且不留单篇正式小节。

    这是 2026-08-30 实测缺陷的完整链路回归：路线验证阶段某路线统计有十余
    篇证据，独占归属后主题只剩 1 篇，最终正文写成"本节纳入 1 篇文献"。
    链路为：真实 overlap 计算 → 唯一 primary owner → dynamic_taxonomy →
    synthesize_themes → build_writing_plan。
    """
    from app.agent.nodes.synthesis import validate_routes_node
    from app.agent.writing_plan import build_writing_plan
    from app.tools.synthesize_themes import synthesize_themes

    recognition = "课堂行为识别 视觉模型 行为检测 学生动作检测"
    interaction = "师生互动 互动序列 时序结构 滞后序列分析"
    # 该论文同时命中两条路线的锚点，但正文主体是互动序列分析
    shared = (
        f"{recognition} {interaction} 本研究以师生互动 互动序列 时序结构为核心，"
        "分析课堂师生互动的时序结构，并辅以课堂行为识别结果作为输入"
    )
    cards = [
        *[_status_card(f"a{index}", recognition) for index in range(1, 4)],
        *[_status_card(f"b{index}", interaction) for index in range(1, 4)],
        _status_card("p_shared", shared),
    ]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "required_reference_count": 20,
        "provisional_framework": {"provisional_routes": [
            _route_with(
                "R1", "课堂行为识别方法",
                "如何用视觉模型识别课堂行为？",
                ["课堂行为识别", "视觉模型", "行为检测"],
            ),
            _route_with(
                "R2", "师生互动序列分析",
                "如何分析师生互动的时序结构？",
                ["师生互动", "互动序列", "时序结构"],
            ),
        ]},
    }

    validate_routes_node(state, llm=None)
    taxonomy = state["dynamic_taxonomy"]
    owners: dict[str, str] = {}
    for item in taxonomy.get("assignments") or []:
        paper_id = str(item["paper_id"])
        assert paper_id not in owners, f"{paper_id} 出现多条 primary 归属"
        owners[paper_id] = str(item["primary_theme_id"])
    assert set(owners) == {c["paper_id"] for c in cards}

    theme_by_name = {
        item["name"]: item["theme_id"] for item in taxonomy.get("themes") or []
    }
    # 重叠论文按契合度归入互动序列路线，而不是列表里的第一条
    assert owners["p_shared"] == theme_by_name["师生互动序列分析"]

    state["theme_synthesis"] = synthesize_themes(cards, taxonomy)
    plan = build_writing_plan("research_status", state)
    theme_sections = [s for s in plan.sections if s.id.startswith("theme_")]

    assert len(theme_sections) == 2
    for section in theme_sections:
        # 正式路线小节必须能做路线内比较：既有章节契约也有实际证据
        assert section.minimum_unique_references == 2, section.title
        assert len(section.supporting_paper_ids) >= 2, section.title
    interaction_section = next(
        s for s in theme_sections
        if s.id == f"theme_{theme_by_name['师生互动序列分析']}"
    )
    assert "p_shared" in interaction_section.supporting_paper_ids
    recognition_section = next(
        s for s in theme_sections
        if s.id == f"theme_{theme_by_name['课堂行为识别方法']}"
    )
    # 独占归属：重叠论文不同时进入两个小节的授权集合
    assert "p_shared" not in recognition_section.supporting_paper_ids
