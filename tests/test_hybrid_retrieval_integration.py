"""混合检索主链接入回归。

覆盖 plan.md 第 10 节测试矩阵中工具级测试之外的主链行为：
shadow 不得改变规则选择、会话 profile 固定、远程模型降级不伪造资格、
取消传播、语言/定向恢复窗口保护、检索循环统一语义准入与无 LLM 安全态。
所有模型调用均使用固定向量与 fake LLM，不访问外部网络。
"""

from __future__ import annotations

import json

import pytest

from app.agent.nodes import rank_node
from app.agent.retrieval_loop import search_rank_with_refinement
from app.clients.retrieval_http import RetrievalProviderError
from app.core import config as config_module
from tests.test_hybrid_retrieval import _FakeEmbedding, _FakeReranker, _settings


@pytest.fixture
def hybrid_settings(tmp_path, monkeypatch):
    """把全进程 settings 固定为小窗口 hybrid，其他字段保持真实默认。"""
    base = config_module.get_settings()
    patched = base.model_copy(update={
        "retrieval_ranking_mode": "hybrid",
        "retrieval_embedding_dimension": 2,
        "retrieval_cache_dir": str(tmp_path),
        "retrieval_bm25_top_k": 8,
        "retrieval_dense_top_k": 8,
        "retrieval_cross_encoder_initial_k": 4,
        "retrieval_cross_encoder_max_k": 4,
        "search_refinement_max_rounds": 0,
    })
    fake = lambda *args, **kwargs: patched
    monkeypatch.setattr(config_module, "get_settings", fake)
    # graph/nodes 通过 `from app.core.config import get_settings` 把函数绑定到
    # 了各自模块命名空间，patch config 模块本身影响不到这些绑定，必须分别
    # patch；否则 profile 决策读到的仍是 conftest 固定的 rules 单例。
    from app.agent import graph as graph_module
    from app.agent.nodes import retrieval as retrieval_node_module
    from app.agent import retrieval_loop as retrieval_loop_module

    monkeypatch.setattr(graph_module, "get_settings", fake)
    monkeypatch.setattr(retrieval_node_module, "get_settings", fake)
    monkeypatch.setattr(retrieval_loop_module, "get_settings", fake)
    return patched


def _patch_clients(monkeypatch, embedding=None, reranker=None):
    from app.services import retrieval_ranking_service as service

    monkeypatch.setattr(
        service, "EmbeddingClient",
        lambda settings, session=None: embedding or _FakeEmbedding(),
    )
    monkeypatch.setattr(
        service, "RerankerClient",
        lambda settings, session=None: reranker or _FakeReranker(),
    )


def _base_state(**overrides):
    state = {
        "topic": "课堂互动",
        "keywords": ["课堂互动"],
        "candidate_papers": [],
        "max_papers": 4,
        "retrieval_target": 4,
        "selected_scope": {},
        "steps": [],
        "errors": [],
    }
    state.update(overrides)
    return state


class _BoomEmbedding:
    def embed(self, texts, *, text_type, should_cancel=None):
        raise RetrievalProviderError("dense 通道不可用")


class _BoomReranker:
    def score(self, query, documents, *, should_cancel=None):
        raise RetrievalProviderError("精排服务不可用")


class _IncludeAllLLM:
    """对构造时给定的每篇论文都返回合规 include 的假 LLM。"""

    def __init__(self, paper_ids):
        self.paper_ids = list(paper_ids)
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        # WHY: 分数严格按输入顺序递减，区分"保留融合顺序"与"随机并列"。
        return json.dumps({"results": [
            {
                "paper_id": paper_id,
                "topic_relevance": 9 - index, "scope_alignment": 9, "method_alignment": 8,
                "decision": "include", "confidence": 0.95,
                "route_id": None, "relation_type": "direct",
                "eligible_deliverables": ["related_work"],
                "required_condition_results": [],
            }
            for index, paper_id in enumerate(self.paper_ids)
        ]})


def test_shadow_mode_keeps_rules_selection_and_records_diagnostics(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    state = _base_state(candidate_papers=[
        {"paper_id": "zh1", "title": "课堂互动行为分析", "abstract": "师生互动编码观察", "year": 2025},
        {"paper_id": "zh2", "title": "课堂互动编码体系", "abstract": "行为序列研究", "year": 2025},
    ], retrieval_profile={"mode": "hybrid_shadow"})

    rank_node(state)

    shadow = state["hybrid_shadow_report"]
    assert shadow["candidate_count"] == 2
    assert shadow["fused_count"] >= 1
    ranked = state["ranked_papers"]
    assert {paper["paper_id"] for paper in ranked} == {"zh1", "zh2"}
    # shadow 只做旁路诊断，规则链路的论文不得携带混合检索特征。
    assert all("_retrieval_features" not in paper for paper in ranked)


def test_hybrid_rank_node_returns_only_llm_confirmed_candidates(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    state = _base_state(candidate_papers=[
        {"paper_id": "p1", "title": "课堂互动行为观察", "abstract": "师生互动编码分析", "year": 2025},
        {"paper_id": "p2", "title": "课堂互动编码体系", "abstract": "行为序列研究", "year": 2025},
    ], retrieval_profile={"mode": "hybrid"})

    rank_node(state, llm=_IncludeAllLLM(["p1", "p2"]))

    assert [paper["paper_id"] for paper in state["ranked_papers"]] == ["p1", "p2"]
    assert state["screening_report"]["mode"] == "hybrid"
    assert state["screening_report"]["hybrid"]["screening"]["confirmed_count"] == 2
    assert state["retrieval_requirement_met"] is False  # 2 篇未达到 target=4


def test_dense_and_rerank_failure_degrades_without_faking_eligibility(hybrid_settings, monkeypatch):
    from pathlib import Path

    ns = _settings(Path(hybrid_settings.retrieval_cache_dir))
    state = _base_state(candidate_papers=[
        # 与 build_queries 生成的整体查询有真实词项交集：BM25 单独可召回。
        {"paper_id": "lex", "title": "课堂互动行为观察", "abstract": "师生互动编码", "year": 2025},
        # 词面完全不同：dense 失败后不得凭身份顺序进入融合结果。
        {"paper_id": "semantic", "title": "师生对话记录", "abstract": "课间交流方式", "year": 2025},
    ], retrieval_profile={"mode": "hybrid"})

    from app.services.retrieval_ranking_service import rank_candidates

    ranked, report = rank_candidates(
        state, settings=ns, include_rerank=True,
        embedding_client=_BoomEmbedding(), reranker_client=_BoomReranker(),
    )

    assert [paper["paper_id"] for paper in ranked] == ["lex"]
    assert ranked[0]["_retrieval_features"]["bm25"] is not None
    assert ranked[0]["_retrieval_features"]["cosine"] is None
    assert ranked[0]["_retrieval_stage"] == "fusion"
    assert any(item.startswith("dense:") for item in report["degraded"])
    assert any(item.startswith("rerank:") for item in report["degraded"])


def test_reranker_failure_keeps_fusion_order_for_llm_screen(hybrid_settings, monkeypatch):
    from app.services.retrieval_ranking_service import rank_candidates, screen_candidates

    _patch_clients(monkeypatch, reranker=_BoomReranker())
    state = _base_state(candidate_papers=[
        {"paper_id": "p1", "title": "课堂互动行为观察", "abstract": "师生互动编码", "year": 2025},
        {"paper_id": "p2", "title": "课堂互动编码体系", "abstract": "行为序列研究", "year": 2025},
    ], retrieval_profile={"mode": "hybrid"})

    ranked, report = rank_candidates(
        state, settings=hybrid_settings, include_rerank=True,
        embedding_client=_FakeEmbedding(), reranker_client=_BoomReranker(),
    )
    assert any(item.startswith("rerank:") for item in report["degraded"])
    fusion_order = [paper["paper_id"] for paper in ranked]
    confirmed, screening = screen_candidates(
        state, ranked, _IncludeAllLLM(fusion_order), target=4,
    )
    # 精排降级时按 RRF 融合顺序进入 LLM 准入，准入通过的论文仍获正式候选资格。
    assert [paper["paper_id"] for paper in confirmed] == fusion_order
    assert screening["confirmed_count"] == 2


def test_cancellation_propagates_before_any_model_call(hybrid_settings):
    from app.services.retrieval_ranking_service import rank_candidates

    state = _base_state(candidate_papers=[
        {"paper_id": "p1", "title": "课堂互动行为观察", "year": 2025},
    ], retrieval_profile={"mode": "hybrid"})
    with pytest.raises(InterruptedError):
        rank_candidates(
            state, settings=hybrid_settings,
            embedding_client=_FakeEmbedding(), reranker_client=_FakeReranker(),
            should_cancel=lambda: True,
        )


def test_language_window_protects_english_from_chinese_fusion_majority(hybrid_settings, monkeypatch):
    from app.services.retrieval_ranking_service import rank_candidates

    _patch_clients(monkeypatch)
    settings = hybrid_settings.model_copy(update={
        "language_branch_min_zh": 0, "language_branch_min_en": 1,
        "retrieval_cross_encoder_initial_k": 3,
        "retrieval_cross_encoder_max_k": 3,
    })
    state = _base_state(candidate_papers=[
        {"paper_id": "zh1", "title": "课堂互动观察一", "abstract": "课堂互动行为分析", "year": 2025},
        {"paper_id": "zh2", "title": "课堂互动观察二", "abstract": "课堂互动编码研究", "year": 2025},
        {"paper_id": "zh3", "title": "课堂互动观察三", "abstract": "师生行为序列", "year": 2025},
        {"paper_id": "en1", "title": "Pupil dialogue in lessons",
         "abstract": "Ethnographic observation of talk in classrooms", "year": 2025},
    ], retrieval_profile={"mode": "hybrid"})

    ranked, _ = rank_candidates(
        state, settings=settings, include_rerank=False,
        embedding_client=_FakeEmbedding(), reranker_client=_FakeReranker(),
    )
    assert "en1" in [paper["paper_id"] for paper in ranked]


def test_targeted_recovery_branch_paper_keeps_its_window_slot(hybrid_settings, monkeypatch):
    from app.services.retrieval_ranking_service import rank_candidates

    _patch_clients(monkeypatch)
    settings = hybrid_settings.model_copy(update={
        "language_branch_min_zh": 0, "language_branch_min_en": 0,
        "retrieval_cross_encoder_initial_k": 4,
        "retrieval_cross_encoder_max_k": 4,
    })
    state = _base_state(
        candidate_papers=[
            {"paper_id": f"zh{index}", "title": f"课堂互动观察{index}",
             "abstract": "课堂互动行为编码分析", "year": 2025}
            for index in range(1, 5)
        ] + [
            # 仅靠 dense 低相似度落在融合尾部，但携带定向恢复分支标签。
            {"paper_id": "rec1", "title": "远距离背景材料", "abstract": "相邻主题资料",
             "year": 2025, "_search_branches": ["quality_focus_recovery"]},
        ],
        search_branches=[{
            "branch_type": "quality_focus_recovery",
            "constraint_level": "targeted_recovery",
            "queries": ["课堂互动 焦点补证"],
        }],
        retrieval_profile={"mode": "hybrid"},
    )

    ranked, report = rank_candidates(
        state, settings=settings, include_rerank=False,
        embedding_client=_FakeEmbedding(), reranker_client=_FakeReranker(),
    )
    assert "rec1" in [paper["paper_id"] for paper in ranked]
    assert report["protected_count"] >= 1


def test_rank_node_rejects_changed_model_config_within_session(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    state = _base_state(candidate_papers=[
        {"paper_id": "p1", "title": "课堂互动", "year": 2025},
    ], retrieval_profile={
        "mode": "hybrid", "config_fingerprint": "stale-fingerprint",
    })
    rank_node(state, llm=None)
    # 节点按既有约定显式失败并可诊断，绝不静默回退到 rules 继续出结果。
    assert not state.get("ranked_papers")
    assert any("配置已变化" in error for error in state.get("errors", []))
    assert state["steps"][-1]["step_name"] == "rank"
    assert state["steps"][-1]["status"] == "failed"


def test_retrieval_loop_runs_single_final_semantic_admission(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    papers = [
        {"paper_id": "p1", "title": "课堂互动行为观察", "abstract": "师生互动编码分析", "year": 2025},
        {"paper_id": "p2", "title": "课堂互动编码体系", "abstract": "行为序列研究", "year": 2025},
    ]
    monkeypatch.setattr(
        "app.agent.retrieval_loop.search_node",
        lambda state, should_cancel=None: state.update({"candidate_papers": [dict(p) for p in papers]}),
    )
    state = _base_state(retrieval_profile={"mode": "hybrid"})
    llm = _IncludeAllLLM(["p1", "p2"])

    search_rank_with_refinement(state, llm=llm)

    assert [paper["paper_id"] for paper in state["ranked_papers"]] == ["p1", "p2"]
    assert state["retrieval_eligible_count"] == 2
    assert state["retrieval_requirement_met"] is False  # target=4 未满足，不提前宣布成功
    assert state["screening_report"]["hybrid"]["screening"]["confirmed_count"] == 2
    # 精化轮 rank_node(llm=None) 未做 CE，末尾统一精排只发生一次。
    assert llm.calls >= 1


@pytest.fixture
def hybrid_deepen_settings(tmp_path, monkeypatch):
    """允许 2 轮精化、大窗口，用于验证 hybrid 循环的提前停止/继续加深判定。"""
    base = config_module.get_settings()
    patched = base.model_copy(update={
        "retrieval_ranking_mode": "hybrid",
        "retrieval_embedding_dimension": 2,
        "retrieval_cache_dir": str(tmp_path),
        "retrieval_bm25_top_k": 20,
        "retrieval_dense_top_k": 20,
        "retrieval_cross_encoder_initial_k": 20,
        "retrieval_cross_encoder_max_k": 20,
        "search_refinement_max_rounds": 2,
    })
    fake = lambda *args, **kwargs: patched
    monkeypatch.setattr(config_module, "get_settings", fake)
    from app.agent import graph as graph_module
    from app.agent.nodes import retrieval as retrieval_node_module
    from app.agent import retrieval_loop as retrieval_loop_module

    monkeypatch.setattr(graph_module, "get_settings", fake)
    monkeypatch.setattr(retrieval_node_module, "get_settings", fake)
    monkeypatch.setattr(retrieval_loop_module, "get_settings", fake)
    return patched


_TOPIC_PAPER_VARIANTS = [
    ("基于FIAS编码体系的中学课堂师生互动话语序列分析",
     "使用弗兰德斯互动分析系统对课堂录像的言语行为编码并做滞后序列分析"),
    ("面向学习分析的课堂观察协议构建与多评分者信度检验",
     "构建课堂教学行为观察量规，报告编码员一致性与信效度验证结果"),
    ("智能教室多模态学习分析中的学生参与行为自动识别",
     "融合音视频与日志信号自动识别学生课堂参与度并预测学习投入"),
    ("教师课堂提问行为编码框架及其对高阶思维的影响研究",
     "对教师提问认知层级编码，分析开放性问题与学生高阶回应的关系"),
    ("师范生课堂教学行为分析工具的本土化修订与应用",
     "修订国外课堂观察工具并在本土教师教育课堂开展编码应用"),
    ("课堂社会网络分析：同伴互动结构与学业表现的关联",
     "以课堂互动网络中心性指标刻画同伴结构并回归学业成绩"),
    ("基于机器学习的课堂讲授与小组讨论行为时序模式挖掘",
     "采集可穿戴设备数据，用序列模型挖掘教学组织行为的时间模式"),
    ("幼儿课堂师幼互动质量的CLASS评估与行为编码研究",
     "用CLASS量表评估师幼互动情感与教学支持并进行行为编码"),
    ("混合式教学中大学生在线线下课堂参与行为对比分析",
     "对比混合课程线上讨论与面授课堂的学生参与编码频次差异"),
    ("课堂反馈行为的微观分析：即时回应与学习坚持性的关系",
     "对教师即时反馈话语逐句编码，追踪其与学生坚持性的纵向关联"),
]


def _topic_paper(index: int):
    title, abstract = _TOPIC_PAPER_VARIANTS[index % len(_TOPIC_PAPER_VARIANTS)]
    return {
        "paper_id": f"p{index}",
        "title": f"{title}（研究 {index}）",
        "abstract": abstract,
        "year": 2024 + (index % 3),
    }


def test_hybrid_loop_skips_refinement_once_window_covers_reserve_target(
    hybrid_deepen_settings, monkeypatch,
):
    """融合窗口已达 1.5× 引用储备时不得再跑精化轮（事故中每轮多花数分钟）。"""
    _patch_clients(monkeypatch)
    papers = [_topic_paper(i) for i in range(7)]
    monkeypatch.setattr(
        "app.agent.retrieval_loop.search_node",
        lambda state, should_cancel=None: state.update({
            "candidate_papers": [dict(p) for p in papers],
            "last_search_new_results": len(papers),
        }),
    )
    refine_calls: list[int] = []
    monkeypatch.setattr(
        "app.agent.retrieval_loop.refine_search_node",
        lambda state, llm=None: refine_calls.append(1),
    )
    state = _base_state(
        max_papers=7,
        retrieval_target=7,
        retrieval_profile={"mode": "hybrid"},
        required_reference_count=4,
    )

    search_rank_with_refinement(state, llm=_IncludeAllLLM([p["paper_id"] for p in papers]))

    # 储备线 ceil(4*1.5)=6；首轮融合窗口 7 篇，一次精化都不应发生。
    # retrieval_target 与 planner 派生一致（required+safety，不小于 1.5×），
    # 末尾 LLM 准入不会把 7 篇确认候选截回 required=4。
    assert refine_calls == []
    assert len(state["ranked_papers"]) == 7
    assert state["retrieval_requirement_met"] is True
    assert "1.5 倍储备" in state["retrieval_stop_reason"]


def test_hybrid_loop_keeps_deepening_while_window_below_reserve_target(
    hybrid_deepen_settings, monkeypatch,
):
    """窗口不足 1.5× 储备时精化加深照常发生，防止提前停止饿死召回。"""
    from app.agent.nodes.base import append_step

    _patch_clients(monkeypatch)
    counter = {"n": 2}

    def _fake_search(state, should_cancel=None):
        # 循环结构：首轮 search_node 先于任何精化判定执行（取第 1 批），
        # 随后每次精化各触发一次增量检索；不得预置候选，否则首批会被双计。
        added = [_topic_paper(100 + counter["n"] + k) for k in range(2)]
        counter["n"] += 2
        state["candidate_papers"] = [*(state.get("candidate_papers") or []), *added]
        state["last_search_new_results"] = 2

    refine_calls: list[int] = []

    def _fake_refine(state, llm=None):
        refine_calls.append(1)
        state["keywords"] = [*(state.get("keywords") or []), f"kw-{len(refine_calls)}"]
        append_step(state, "refine_search", "success",
                    output_data={"keywords": state["keywords"]})

    monkeypatch.setattr("app.agent.retrieval_loop.search_node", _fake_search)
    monkeypatch.setattr("app.agent.retrieval_loop.refine_search_node", _fake_refine)
    state = _base_state(
        max_papers=7,
        retrieval_target=7,
        retrieval_profile={"mode": "hybrid"},
        required_reference_count=4,
    )
    # 首轮 p102/p103，两轮精化分别补 p104/p105、p106/p107；窗口 2→4→6。
    llm = _IncludeAllLLM(["p102", "p103", "p104", "p105", "p106", "p107"])

    search_rank_with_refinement(state, llm=llm)

    # 窗口 2/4 均 < 储备线 6：两轮精化都应执行；窗口到 6 后跳出，准入 6 篇达标。
    assert len(refine_calls) == 2
    assert len(state["ranked_papers"]) == 6
    assert state["retrieval_requirement_met"] is True


def test_retrieval_loop_without_llm_keeps_pending_candidates(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    papers = [
        {"paper_id": "p1", "title": "课堂互动行为观察", "abstract": "师生互动编码分析", "year": 2025},
    ]
    monkeypatch.setattr(
        "app.agent.retrieval_loop.search_node",
        lambda state, should_cancel=None: state.update({"candidate_papers": [dict(p) for p in papers]}),
    )
    state = _base_state(retrieval_profile={"mode": "hybrid"})

    search_rank_with_refinement(state, llm=None)

    assert [paper["paper_id"] for paper in state["ranked_papers"]] == ["p1"]
    assert state["ranked_papers"][0]["_pending_semantic_check"] is True
    assert state["retrieval_requirement_met"] is False
    assert "等待 LLM 语义准入" in state["retrieval_stop_reason"]
    assert state["screening_report"]["hybrid"]["screening"]["mode"] == "pending_no_llm"


def test_pending_checkpoint_is_screened_without_empty_repeat_search(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    papers = [{"paper_id": "p1", "title": "课堂互动行为观察",
               "abstract": "师生互动编码分析", "year": 2025}]
    calls = []

    def search(state, should_cancel=None):
        calls.append("search")
        state["candidate_papers"] = [dict(p) for p in papers]

    monkeypatch.setattr("app.agent.retrieval_loop.search_node", search)
    state = _base_state(retrieval_profile={"mode": "hybrid"})
    search_rank_with_refinement(state, llm=None)
    assert len(calls) == 1
    state["searched_keywords"] = list(state.get("keywords") or [])
    search_rank_with_refinement(state, llm=_IncludeAllLLM(["p1"]))
    assert len(calls) == 1
    assert [paper["paper_id"] for paper in state["ranked_papers"]] == ["p1"]
    assert state["ranked_papers"][0]["_screening_decision"] == "include"


def test_pending_checkpoint_rechecks_changed_hard_year_before_admission(hybrid_settings, monkeypatch):
    _patch_clients(monkeypatch)
    monkeypatch.setattr("app.agent.retrieval_loop.search_node", lambda state, should_cancel=None:
                        state.update(candidate_papers=[{
                            "paper_id": "p1", "title": "课堂互动行为观察",
                            "abstract": "师生互动编码分析", "year": 2025,
                        }]))
    state = _base_state(retrieval_profile={"mode": "hybrid"})
    search_rank_with_refinement(state, llm=None)
    state["searched_keywords"] = list(state.get("keywords") or [])
    state["start_year"] = 2026
    search_rank_with_refinement(state, llm=_IncludeAllLLM(["p1"]))
    assert state["ranked_papers"] == []
    assert state["retrieval_eligible_count"] == 0


def test_final_editable_state_keeps_hybrid_strategy_for_resume():
    from app.agent.graph import _build_output

    output = _build_output({
        "intent": "generate_review", "result_status": "blocked",
        "retrieval_profile": {"mode": "hybrid"}, "steps": [], "errors": [],
        "review": "", "paper_cards": [], "paper_details": [],
    })
    assert output["research_state"]["retrieval_profile"] == {"mode": "hybrid"}


def test_blocked_public_diagnostic_separates_cards_from_usable_and_cited():
    from app.agent.graph import _build_output

    output = _build_output({
        "intent": "generate_review", "result_status": "blocked",
        "required_reference_count": 3, "max_papers_explicit": True,
        "paper_cards": [{"paper_id": "a"}, {"paper_id": "b"}],
        "generation_readiness": {"usable_reference_count": 1},
        "quality_gate": {"passed": False, "phase": "pre_generation",
                         "blocking_issues": [{"code": "minimum_references_not_met"}]},
        "errors": [{"code": "agent_no_progress"}], "review": "",
    })
    assert "2 张证据卡" in output["answer"]
    assert "可用 1 篇" in output["answer"]
    assert "有效引用 0 篇" in output["answer"]
    assert "最终引用 3 篇" in output["answer"]


class _DecisionLLM:
    """最小主控决策假机：search_and_rank 后 request_finish。"""

    native_tools_enabled = True

    def __init__(self):
        self.actions = iter(["search_and_rank", ("request_finish", {})])

    def complete_tool_call(self, messages, **kwargs):
        action = next(self.actions)
        if isinstance(action, tuple):
            return {"name": action[0], "arguments": action[1]}
        return {"name": action, "arguments": {}}


def _run_graph_and_capture_profile(monkeypatch, initial_state):
    from app.agent import graph

    captured: list[str] = []

    def fake_search(current, **kwargs):
        captured.append(str((current.get("retrieval_profile") or {}).get("mode")))
        current["ranked_papers"] = [{"paper_id": "p1", "title": "论文"}]

    monkeypatch.setattr(graph, "_get_llm", lambda: _DecisionLLM())
    monkeypatch.setattr(
        graph, "plan_node",
        lambda current, **kwargs: current.update({"intent": "search_papers", "topic": "主题"}),
    )
    monkeypatch.setattr(graph, "_search_rank_with_refinement", fake_search)
    output = graph.run_research_agent("检索主题论文", initial_state=initial_state)
    return output, captured


def test_new_session_uses_configured_hybrid_profile(hybrid_settings, monkeypatch):
    output, captured = _run_graph_and_capture_profile(monkeypatch, None)
    assert output["status"] == "success"
    assert captured == ["hybrid"]


def test_previous_work_session_is_pinned_to_rules(hybrid_settings, monkeypatch):
    initial = {"candidate_papers": [{"paper_id": "old1", "title": "既有候选"}]}
    output, captured = _run_graph_and_capture_profile(monkeypatch, initial)
    assert output["status"] == "success"
    assert captured == ["rules"]
