"""句子级主张—证据验证测试。"""

from __future__ import annotations

import pytest

from app.tools.extract_paper_card import extract_paper_card
from app.tools.training_data import (
    build_controlled_training_records,
    build_qlora_records,
)
from app.tools.verify_claims import verify_review_claims


def test_failure_baselines_keep_log_and_persisted_snapshots_separate():
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "claim_quality_failure_scenarios.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    scenarios = {item["id"]: item for item in data["scenarios"]}

    log_replay = scenarios["log_replay_2026_09_06_1902"]
    persisted = scenarios["persisted_state_2026_09_06_1711"]
    assert log_replay["evidence_card_count"] == 153
    assert log_replay["has_sentence_level_snapshot"] is False
    assert persisted["evidence_card_count"] == 72
    assert persisted["has_sentence_level_snapshot"] is True
    assert log_replay["id"] != persisted["id"]


def _card() -> dict:
    paper = {
        "paper_id": "p1",
        "title": "Evidence Verification for RAG",
        "authors": ["Alice"],
        "year": 2024,
        "venue": "ACL",
        "abstract": (
            "We propose an evidence verification method for retrieval augmented generation. "
            "The method achieves 80% accuracy on the FEVER dataset."
        ),
    }
    return extract_paper_card(paper, llm=None, topic="RAG hallucination").model_dump()


def _second_card() -> dict:
    paper = {
        "paper_id": "p2",
        "title": "Another Verification Study",
        "authors": ["Bob"],
        "year": 2025,
        "venue": "EMNLP",
        "abstract": (
            "We study a different verification pipeline. "
            "The pipeline achieves 95% accuracy on the SciFact dataset."
        ),
    }
    return extract_paper_card(paper, llm=None, topic="RAG hallucination").model_dump()


def test_rendered_citations_keep_the_same_evidence_on_repeated_checks():
    """首次出现编号不得在重验时变成证据卡片顺序。"""
    from app.agent.nodes.verification import citation_check_node, verify_claims_node

    first, second = _card(), _second_card()
    state = {
        "review": "## 研究现状\n\n该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        "paper_cards": [second, first],
        "paper_details": [second, first],
        "writing_plans": [],
        "citation_style": "gbt7714",
        "required_reference_count": 1,
        "steps": [],
        "errors": [],
    }
    citation_check_node(state)
    first_reference = state["reference_papers"][0]["paper_id"]
    assert first_reference == "p1"
    assert state["review"].endswith("[1]。")
    citation_check_node(state)
    assert state["reference_papers"][0]["paper_id"] == first_reference
    state["writing_plans"] = [{}]
    verify_claims_node(state, llm=None)
    claim = next(item for item in state["claim_verification"]["claims"] if item["factual"])
    assert claim["citations"] == ["p1"]
    assert claim["evidence_snippets"]
    assert "citation_not_found" not in claim["issues"]


def test_serialized_rendered_citations_survive_card_reordering():
    import json
    from app.agent.nodes.verification import citation_check_node, verify_claims_node

    first, second = _card(), _second_card()
    state = {
        "review": "## 研究现状\n\n该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        "paper_cards": [second, first], "paper_details": [second, first],
        "writing_plans": [], "citation_style": "gbt7714",
        "required_reference_count": 1, "steps": [], "errors": [],
    }
    citation_check_node(state)
    resumed = json.loads(json.dumps(state, ensure_ascii=False))
    resumed["paper_cards"].reverse()
    resumed["paper_details"].reverse()
    citation_check_node(resumed)
    verify_claims_node(resumed, llm=None)
    assert resumed["reference_papers"][0]["paper_id"] == "p1"
    assert resumed["claim_verification"]["claims"][0]["citations"] == ["p1"]


def test_legacy_numeric_draft_without_trusted_source_fails_closed():
    from app.agent.nodes.verification import citation_check_node, verify_claims_node

    state = {
        "review": "## 研究现状\n\n该方法达到 80% accuracy [1]。",
        "paper_cards": [_card(), _second_card()],
        "paper_details": [_card(), _second_card()],
        "writing_plans": [], "citation_style": "gbt7714",
        "required_reference_count": 1, "steps": [], "errors": [],
        "claim_verification": {"valid": True, "supported": 1},
    }
    citation_check_node(state)
    verify_claims_node(state, llm=None)
    assert state["citation_validation"]["reason"] == "citation_identity_unverified"
    assert state["citation_map"] == {}
    assert state["generation_quality"]["reason"] == "citation_identity_unverified"
    assert state["claim_verification"]["prior_report"]["supported"] == 1


def test_numbers_from_two_papers_cannot_be_unioned_into_support():
    """两篇论文各自报告一个数字时，合并引用不得让复合句判为已支持。"""
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy，并在 SciFact 数据集上达到 95% accuracy [p1][p2]。",
        [_card(), _second_card()],
    )

    claim = report["claims"][0]
    assert claim["support_status"] != "supported"
    assert claim["atomic_claims"]
    # 每个原子单元只允许使用自己那一篇论文的证据。
    for atomic in claim["atomic_claims"]:
        assert len(atomic["citations"]) == 1
        for evidence_id in atomic["evidence_ids"]:
            assert evidence_id.startswith(f"{atomic['citations'][0]}:")


def test_clause_local_citations_bind_to_their_own_paper():
    """分句局部引用应各自绑定到本句论文，不共享证据池。"""
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]；"
        "另一流程在 SciFact 数据集上达到 95% accuracy [p2]。",
        [_card(), _second_card()],
    )

    claim = report["claims"][0]
    atomic = claim["atomic_claims"]
    assert len(atomic) == 2
    assert atomic[0]["citations"] == ["p1"]
    assert atomic[1]["citations"] == ["p2"]
    assert all(eid.startswith("p1:") for eid in atomic[0]["evidence_ids"])
    assert all(eid.startswith("p2:") for eid in atomic[1]["evidence_ids"])


def test_llm_entailment_payload_carries_one_atomic_claim_per_item():
    """LLM 蕴含验证的每个项目只能携带该原子主张自己的证据。"""
    payloads: list[list[dict]] = []

    class RecordingLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            import json
            import re

            match = re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S)
            if match:
                payloads.append(json.loads(match.group(1)))
            return '{"results":[{"claim_id":"c001","label":"entailed","confidence":0.95,"reason":"支持"}]}'

    verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]；"
        "另一流程在 SciFact 数据集上达到 95% accuracy [p2]。",
        [_card(), _second_card()],
        llm=RecordingLLM(),
    )

    assert payloads
    for payload in payloads:
        assert 1 <= len(payload) <= 12
        for item in payload:
            # 批量请求允许携带多个项目，但每个项目的证据仍只能来自自己的原子单元。
            joined = " ".join(item["evidence"])
            if "80%" in item["claim"]:
                assert "95%" not in joined
            if "95%" in item["claim"]:
                assert "80%" not in joined


def test_entailment_cache_reuses_only_matching_claim_evidence_fingerprint():
    calls = 0

    class EchoEntailmentLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            nonlocal calls
            import json
            import re

            calls += 1
            payload = json.loads(re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S).group(1))
            return json.dumps({"results": [
                {
                    "claim_id": item["claim_id"],
                    "label": "entailed",
                    "confidence": 0.95,
                    "reason": "支持",
                }
                for item in payload
            ]})

    cache: dict = {}
    text = "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
    first = verify_review_claims(text, [_card()], llm=EchoEntailmentLLM(), entailment_cache=cache)
    second = verify_review_claims(text, [_card()], llm=EchoEntailmentLLM(), entailment_cache=cache)
    changed = _card()
    changed["evidence_spans"][0]["text"] += " Additional validation evidence."
    third = verify_review_claims(text, [changed], llm=EchoEntailmentLLM(), entailment_cache=cache)

    assert calls == 2
    assert first["entailment_cache_stats"] == {"reused": 0, "computed": 1}
    assert second["entailment_cache_stats"] == {"reused": 1, "computed": 0}
    assert third["entailment_cache_stats"] == {"reused": 0, "computed": 1}
    assert first["entailment_llm_batch_stats"] == {"attempted": 1, "failed": 0, "claims_submitted": 1}
    assert second["entailment_llm_batch_stats"] == {"attempted": 0, "failed": 0, "claims_submitted": 0}
    assert third["semantic_verification"]["llm_claims_submitted"] == 1


def test_supported_claim_links_to_evidence_span():
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
    )

    assert report["supported"] == 1
    claim = report["claims"][0]
    assert claim["support_status"] == "supported"
    assert claim["evidence_ids"]
    assert claim["evidence_snippets"][0]["source_type"] == "abstract"


def test_fullwidth_citation_is_not_skipped_as_non_factual():
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy〔p1〕。",
        [_card()],
    )

    assert report["supported"] == 1
    claim = report["claims"][0]
    assert claim["factual"] is True
    assert claim["citations"] == ["p1"]
    assert claim["support_status"] == "supported"


def test_changed_number_is_unsupported():
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 95% accuracy [p1]。",
        [_card()],
    )

    claim = report["claims"][0]
    assert claim["support_status"] == "unsupported"
    assert "numeric_value_not_found_in_evidence" in claim["issues"]


def test_llm_entailment_rejects_opposite_claim_despite_lexical_overlap():
    class ContradictionLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            assert kwargs["operation"] == "verify_claim_entailment"
            return (
                '{"results":[{"claim_id":"c001u01","label":"contradicted",'
                '"confidence":0.98,"reason":"证据报告达到80%，主张却否定该结果"}]}'
            )

    report = verify_review_claims(
        "该方法并未在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
        llm=ContradictionLLM(),
    )

    claim = report["claims"][0]
    assert claim["support_status"] == "unsupported"
    assert "claim_contradicted_by_evidence" in claim["issues"]


def test_cross_language_paraphrase_uses_semantic_entailment_after_hard_checks():
    class EntailedLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return (
                '{"results":[{"claim_id":"c001u01","label":"entailed",'
                '"confidence":0.96,"reason":"中文主张与英文证据语义一致"}]}'
            )

    report = verify_review_claims(
        "该研究提出了一种用于检索增强生成的证据核验方法[p1]。",
        [_card()],
        llm=EntailedLLM(),
    )

    claim = report["claims"][0]
    assert claim["support_status"] == "supported"
    assert claim["verification_status"] == "verified"
    assert claim["verification_method"] == "semantic"


def test_string_confidence_from_llm_does_not_crash_verification():
    class StringConfidenceLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return (
                '{"results":[{"claim_id":"c001u01","label":"entailed",'
                '"confidence":"high","reason":"证据直接支持该主张"}]}'
            )

    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
        llm=StringConfidenceLLM(),
    )

    claim = report["claims"][0]
    # 非数字置信度按低置信度降级，报告照常产出
    assert claim["support_status"] == "partially_supported"
    assert "low_entailment_confidence" in claim["issues"]


def test_missing_semantic_result_is_reported_as_unverified_not_proven_unsupported():
    class EmptyResultLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return '{"results": []}'

    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
        llm=EmptyResultLLM(),
    )

    claim = report["claims"][0]
    assert claim["support_status"] == "unsupported"
    assert claim["verification_status"] == "not_completed"
    assert claim["verification_method"] == "semantic"
    assert report["unverified"] == 1
    assert report["verification_coverage"] == 0.0
    assert "entailment_not_verified" in claim["issues"]


def test_semantic_cache_fingerprint_includes_access_and_source_provenance():
    calls = 0

    class EchoLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            nonlocal calls
            import json
            import re

            calls += 1
            payload = json.loads(re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S).group(1))
            return json.dumps({"results": [
                {"claim_id": item["claim_id"], "label": "entailed", "confidence": 0.95}
                for item in payload
            ]})

    cache = {}
    card = _card()
    text = "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
    verify_review_claims(text, [card], llm=EchoLLM(), entailment_cache=cache)
    changed = _card()
    changed["evidence_state"]["access_level"] = "partial_full_text"
    verify_review_claims(text, [changed], llm=EchoLLM(), entailment_cache=cache)

    assert calls == 2


def test_model_names_do_not_trigger_numeric_mismatch():
    card = _card()
    report = verify_review_claims(
        "基于 GPT-4 与 3D 卷积的方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [card],
    )

    claim = report["claims"][0]
    assert "numeric_value_not_found_in_evidence" not in claim["issues"]
    assert claim["support_status"] in ("supported", "partially_supported")


def test_overstated_language_is_flagged_and_weakened():
    report = verify_review_claims(
        "该方法在 FEVER 数据集上显著提升了 accuracy [p1]。",
        [_card()],
    )

    claim = report["claims"][0]
    assert claim["support_status"] in ("partially_supported", "unsupported")
    assert "overstated_language" in claim["issues"]
    assert "显著" not in claim["suggested_revision"]


def test_missing_citation_is_unsupported():
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy。",
        [_card()],
    )

    claim = report["claims"][0]
    assert claim["support_status"] == "unsupported"
    assert "factual_claim_without_citation" in claim["issues"]


def test_uncited_literature_synthesis_is_factual_not_a_free_transition():
    report = verify_review_claims(
        "现有研究共同强调课堂行为分析对教学评估具有关键作用。",
        [_card()],
    )

    claim = report["claims"][0]
    assert claim["factual"] is True
    assert claim["support_status"] == "unsupported"
    assert "factual_claim_without_citation" in claim["issues"]


def test_template_prefix_does_not_create_an_implicit_citation():
    report = verify_review_claims(
        "跨论文综合判断：该方法在 FEVER 数据集上达到 80% accuracy。",
        [_card()],
    )

    claim = report["claims"][0]
    assert claim["support_status"] == "unsupported"
    assert claim["citations"] == []
    assert claim["evidence_ids"] == []
    assert "factual_claim_without_citation" in claim["issues"]


def test_abstract_cannot_support_ablation_or_detailed_experiment_claim():
    report = verify_review_claims(
        "消融实验表明该模块在所有基线上均贡献了性能提升 [p1]。",
        [_card()],
    )
    claim = report["claims"][0]
    assert claim["required_access_level"] == "full_text"
    assert claim["actual_access_level"] == "abstract"
    assert claim["support_status"] == "unsupported"
    assert "access_level_too_weak_for_claim" in claim["issues"]


def test_quality_report_counts_abstract_only_papers():
    report = verify_review_claims("该方向仍需进一步研究。", [_card()])
    assert report["evidence_summary"]["abstract"] == 1
    assert report["evidence_limitations"]


def test_report_can_be_exported_as_qlora_records():
    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
    )
    records = build_qlora_records(report)

    assert len(records) == 1
    assert records[0]["label"] == "supported"
    assert "80%" in records[0]["evidence"]


def test_controlled_training_records_cover_three_labels():
    first = _card()
    second = _card()
    second["paper_id"] = "p2"
    second["title"] = "A Different RAG Method"
    for span in second["evidence_spans"]:
        span["evidence_id"] = span["evidence_id"].replace("p1:", "p2:")
    second["field_evidence"] = {
        field: [evidence_id.replace("p1:", "p2:") for evidence_id in evidence_ids]
        for field, evidence_ids in second["field_evidence"].items()
    }

    records = build_controlled_training_records([first, second])
    labels = {record["label"] for record in records}

    assert labels == {"supported", "partially_supported", "unsupported"}
    assert any(record["mutation_type"] == "cross_paper_evidence_swap" for record in records)


def test_dataset_builder_splits_by_paper_without_cross_split_leakage(tmp_path):
    import json

    from scripts.build_claim_verifier_dataset import build_dataset

    cards = []
    for index in range(30):
        card = _card()
        old_id = card["paper_id"]
        paper_id = f"paper-{index}"
        card["paper_id"] = paper_id
        for span in card["evidence_spans"]:
            span["evidence_id"] = span["evidence_id"].replace(f"{old_id}:", f"{paper_id}:")
        card["field_evidence"] = {
            field: [evidence_id.replace(f"{old_id}:", f"{paper_id}:") for evidence_id in evidence_ids]
            for field, evidence_ids in card["field_evidence"].items()
        }
        cards.append(card)

    input_path = tmp_path / "agent.json"
    input_path.write_text(json.dumps({"paper_cards": cards}, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "dataset"
    counts = build_dataset(input_path, output_dir)

    split_papers = {}
    for split in ("train", "dev", "test"):
        records = [
            json.loads(line)
            for line in (output_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        split_papers[split] = {
            paper_id
            for record in records
            for paper_id in (record.get("paper_id"), record.get("evidence_paper_id"))
            if paper_id
        }
        assert len(records) == counts[split]

    assert split_papers["train"].isdisjoint(split_papers["dev"])
    assert split_papers["train"].isdisjoint(split_papers["test"])
    assert split_papers["dev"].isdisjoint(split_papers["test"])


def test_rewrite_and_weaken_unsupported_claims_with_llm():
    from app.agent.nodes.verification import _rewrite_and_weaken_unsupported_claims
    from unittest.mock import MagicMock

    card = _card()
    unsupported = [
        {
            "claim_id": "c001",
            "sentence": "该方法在消融实验中显著优于基线并且准确率达到99.8%[p001]。",
            "citations": ["p001"],
            "issues": ["numeric_value_not_found_in_evidence", "access_level_too_weak_for_claim"],
        }
    ]
    orig_text = "一、研究现状\n\n该方法在消融实验中显著优于基线并且准确率达到99.8%[p001]。其他工作在进一步探索中。"

    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"results": [{"claim_id": "c001", "original": "该方法在消融实验中显著优于基线并且准确率达到99.8%[p001]。", "revised": "该方法采用跨视角表示学习以增强判别稳定性[p001]。"}]}'

    revised_text, records = _rewrite_and_weaken_unsupported_claims(
        orig_text,
        unsupported,
        [card],
        llm=mock_llm,
    )

    assert len(records) == 1
    assert records[0]["method"] == "llm_rewrite"
    assert "[p001]" in revised_text
    assert "99.8%" not in revised_text
    assert "跨视角表示学习" in revised_text


def test_rule_weakening_fallback():
    from app.agent.nodes.verification import _rewrite_and_weaken_unsupported_claims

    card = _card()
    unsupported = [
        {
            "claim_id": "c001",
            "sentence": "该方法首次提出全新框架并且准确率达到95.5%[p001]。",
            "citations": ["p001"],
            "issues": ["numeric_value_not_found_in_evidence", "overstated_language"],
        }
    ]
    orig_text = "该方法首次提出全新框架并且准确率达到95.5%[p001]。"

    revised_text, records = _rewrite_and_weaken_unsupported_claims(
        orig_text,
        unsupported,
        [card],
        llm=None,
    )

    assert len(records) == 1
    assert records[0]["method"] == "rule_weakening"
    assert "[p001]" in revised_text
    assert "首次提出" not in revised_text
    assert "提出全新框架" in revised_text


def test_rewrite_cannot_add_citations_to_an_uncited_sentence():
    """原判据 req_cits <= rev_cits 在原句无引用时恒真，改写可凭空补引用。"""
    from app.agent.nodes.verification import _rewrite_and_weaken_unsupported_claims
    from unittest.mock import MagicMock

    sentence = "课堂行为编码体系在近年研究中得到持续细化。"
    unsupported = [{
        "claim_id": "c001",
        "sentence": sentence,
        "citations": [],
        "issues": ["missing_citation"],
    }]
    orig_text = f"一、研究现状\n\n{sentence}其他工作在进一步探索中。"

    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        '{"results": [{"claim_id": "c001", "original": "' + sentence +
        '", "revised": "课堂行为编码体系在近年研究中得到持续细化[p001]。"}]}'
    )

    revised_text, records = _rewrite_and_weaken_unsupported_claims(
        orig_text, unsupported, [_card()], llm=mock_llm,
    )

    assert all(record["method"] != "llm_rewrite" for record in records)
    assert "[p001]" not in revised_text
    assert sentence in revised_text


def test_rewrite_is_rejected_when_it_restates_a_neighbour_sentence():
    """改写不得把无支持句变成上下文已有句子的复述。"""
    from app.agent.nodes.verification import _rewrite_and_weaken_unsupported_claims
    from unittest.mock import MagicMock

    neighbour = "多模态融合为课堂行为识别提供了新的证据来源[p001]。"
    sentence = "该模型在课堂场景中的识别准确率达到98.6%[p001]。"
    orig_text = f"一、研究现状\n\n{sentence}{neighbour}"

    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        '{"results": [{"claim_id": "c001", "original": "' + sentence +
        '", "revised": "' + neighbour + '"}]}'
    )

    revised_text, records = _rewrite_and_weaken_unsupported_claims(
        orig_text,
        [{
            "claim_id": "c001",
            "sentence": sentence,
            "citations": ["p001"],
            "issues": ["numeric_value_not_found_in_evidence"],
        }],
        [_card()],
        llm=mock_llm,
    )

    assert all(record["method"] != "llm_rewrite" for record in records)
    assert revised_text.count("多模态融合为课堂行为识别提供了新的证据来源") == 1


def test_zero_improvement_rewrite_batch_is_not_persisted(monkeypatch):
    """改写未减少未支持主张时整批不落盘，转由 section_rewrite_required 报告。"""
    from app.agent.nodes import verification

    original = (
        "一、研究现状\n\n"
        "该方法在消融实验中的准确率达到99.8%[p001]。"
        "该模型在跨校样本上的召回率达到97.3%[p001]。"
        "该框架在真实课堂中的部署成本下降四成[p001]。"
        "其他工作在进一步探索中。"
    )
    cosmetic = original.replace("其他工作在进一步探索中。", "其余研究仍在持续推进之中。")
    monkeypatch.setattr(
        verification,
        "_rewrite_and_weaken_unsupported_claims",
        lambda text, claims, cards, llm=None: (cosmetic, [{
            "claim_id": "c001", "original": "原句", "revised": "改写句",
            "method": "llm_rewrite",
        }]),
    )
    state = {
        "intent": "generate_review",
        "review": original,
        "body": original,
        "paper_cards": [_card()],
        "writing_plans": [{"sections": [{"title": "研究现状"}]}],
        "steps": [],
        "errors": [],
    }

    verification.verify_claims_node(state, llm=None)

    assert state["claim_repairs"]["strategy"] == "section_rewrite_required"
    assert state["claim_repairs"]["unsupported_after_rewrite"] >= 3
    assert state["review"] == original
    assert "其余研究仍在持续推进之中" not in state["review"]


def test_post_rewrite_verification_keeps_semantic_verifier(monkeypatch):
    """重写后的候选必须沿用初次验证的 LLM 语义核验契约。"""
    from app.agent.nodes import verification

    seen_llms = []
    calls = 0

    def fake_verify(text, cards, **kwargs):
        nonlocal calls
        calls += 1
        seen_llms.append(kwargs.get("llm"))
        unsupported = 6 if calls == 1 else 5
        claims = [
            {
                "claim_id": f"c{index:03d}",
                "sentence": f"事实主张{index}[p1]。",
                "citations": ["p1"],
                "factual": True,
                "support_status": "unsupported",
                "verification_status": "verified",
            }
            for index in range(1, unsupported + 1)
        ]
        return {
            "valid": False,
            "total_sentences": unsupported,
            "factual_claims": unsupported,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": unsupported,
            "unverified": 0,
            "verification_coverage": 1.0,
            "support_rate": 0.0,
            "claims": claims,
        }

    monkeypatch.setattr("app.tools.verify_claims.verify_review_claims", fake_verify)
    monkeypatch.setattr(
        verification,
        "_rewrite_and_weaken_unsupported_claims",
        lambda text, claims, cards, llm=None: (
            text.replace("原始内容", "修复内容"),
            [{"claim_id": "c001", "original": "原始内容", "revised": "修复内容", "method": "llm_rewrite"}],
        ),
    )
    semantic_llm = object()
    state = {
        "review": "## 研究现状\n\n原始内容。",
        "body": "## 研究现状\n\n原始内容。",
        "paper_cards": [_card()],
        "writing_plans": [{"sections": [{"title": "研究现状"}]}],
        "steps": [],
        "errors": [],
    }

    verification.verify_claims_node(state, llm=semantic_llm)

    assert calls >= 2
    assert all(item is semantic_llm for item in seen_llms)



def test_local_verification_reuses_untargeted_claims_and_reports_stats():
    """局部模式只重算目标句，保留上一轮未受影响句及其证据结果。"""
    class RecordingLLM:
        def __init__(self):
            self.claim_ids = []

        def complete(self, prompt: str, **kwargs) -> str:
            import json
            import re
            payload = json.loads(re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S).group(1))
            self.claim_ids.extend(item["claim_id"] for item in payload)
            return json.dumps({"results": [
                {"claim_id": item["claim_id"], "label": "entailed", "confidence": 0.95, "reason": "支持"}
                for item in payload
            ]})

    llm = RecordingLLM()
    text = (
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
        "另一流程在 SciFact 数据集上达到 95% accuracy [p2]。"
    )
    cache = {}
    first = verify_review_claims(text, [_card(), _second_card()], llm=llm, entailment_cache=cache)
    llm.claim_ids.clear()
    second = verify_review_claims(
        text,
        [_card(), _second_card()],
        llm=llm,
        entailment_cache=cache,
        target_sentence_indices=[2],
        verification_scope={"mode": "local", "previous_report": first},
    )

    # 目标句重新解析，但其指纹命中既有 cache，因此无需再次调用 LLM。
    assert llm.claim_ids == []
    assert second["entailment_cache_stats"] == {"reused": 1, "computed": 0}
    assert second["verification_scope"]["mode"] == "local"
    assert second["verification_scope"]["reused_sentences"] == 1
    assert second["verification_scope"]["recomputed_sentences"] == 1
    assert [claim["claim_id"] for claim in second["claims"]] == ["c001", "c002"]
    assert second["claims"][0] == first["claims"][0]


def test_local_verification_accepts_target_claim_id():
    """原子 claim ID 可以精确选择其所属句子进行局部验证。"""
    text = (
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
        "另一流程在 SciFact 数据集上达到 95% accuracy [p2]。"
    )
    first = verify_review_claims(text, [_card(), _second_card()])
    second = verify_review_claims(
        text,
        [_card(), _second_card()],
        target_claim_ids=["c002u01"],
        verification_scope={"mode": "local", "previous_report": first},
    )
    assert second["verification_scope"]["target_claim_ids"] == ["c002u01"]
    assert second["verification_scope"]["recomputed_sentences"] == 1
    assert second["verification_scope"]["reused_sentences"] == 1


def test_local_verification_reuses_unchanged_content_after_sentence_deletion():
    """句序移动后按完整句子与引用复用，不能把旧序号的结论套给新句。"""
    text = (
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
        "另一流程在 SciFact 数据集上达到 95% accuracy [p2]。"
    )
    first = verify_review_claims(text, [_card(), _second_card()])
    second = verify_review_claims(
        "另一流程在 SciFact 数据集上达到 95% accuracy [p2]。",
        [_card(), _second_card()],
        target_sentence_indices=[2],
        verification_scope={"mode": "local", "previous_report": first},
    )

    assert second["verification_scope"]["mode"] == "local"
    assert second["verification_scope"]["reused_sentences"] == 1
    assert second["verification_scope"]["recomputed_sentences"] == 0
    assert second["claims"][0]["claim_id"] == "c001"
    assert "SciFact" in second["claims"][0]["sentence"]


def test_legacy_report_without_verification_contract_forces_full_revalidation():
    first = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
    )
    for claim in first["claims"]:
        claim.pop("verification_status", None)
        claim.pop("verification_method", None)

    second = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        [_card()],
        target_sentence_indices=[1],
        verification_scope={"mode": "local", "previous_report": first},
    )

    assert second["verification_scope"]["mode"] == "full"
    assert second["verification_scope"]["recomputed_sentences"] == 1


def test_number_tokens_do_not_truncate_unit_suffixes():
    """M12 回归：数值 token 不因回溯截断，"70B" 不得被抽成 "7"。

    旧正则会在 "7" 后停止匹配导致 "70B"→"7" 的假数值相等，把
    "参数量 7B" 的证据误判为支撑 "70B" 的主张。
    """
    from app.tools.verify_claims import _numbers

    # 单位后缀紧贴数字：整串不作为数值 token 抽出
    assert "70" not in _numbers("模型规模为 70B 参数")
    assert "7" not in _numbers("GPT-4 与 3D 卷积")
    # 独立数字、百分比、小数、中文后缀（"2023年"）不受影响
    assert "85%" in _numbers("准确率达到 85%.")
    assert "3.14" in _numbers("增益为 3.14 倍")
    assert "2023" in _numbers("2023年提出")
    # URL/时间戳里的数字不抽出
    assert "12345" not in _numbers("https://arxiv.org/abs/2401.12345")


def _numbered_cards(count: int) -> list[dict]:
    """构造 count 张各自独立、含可匹配数值的证据卡，用于跨批次蕴含测试。"""
    cards: list[dict] = []
    for index in range(count):
        paper = {
            "paper_id": f"p{index}",
            "title": f"Verification Study {index}",
            "authors": ["Alice"],
            "year": 2024,
            "venue": "ACL",
            "abstract": (
                f"We propose method {index}. "
                f"The method achieves {80 + index}% accuracy on the FEVER dataset."
            ),
        }
        cards.append(extract_paper_card(paper, llm=None, topic="RAG").model_dump())
    return cards


def _numbered_text(count: int) -> str:
    return "".join(
        f"方法{index}在 FEVER 数据集上达到 {80 + index}% accuracy [p{index}]。"
        for index in range(count)
    )


def test_entailment_control_exception_propagates_and_stops_later_batches():
    """T01：蕴含批处理遇预算停止信号立即传播，后续批次调用数为 0。"""
    from app.agent.execution_budget import AgentBudgetExceeded

    calls = {"n": 0}

    class BudgetStopLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            calls["n"] += 1
            raise AgentBudgetExceeded("token reservation exceeds remaining budget")

    # 13 条事实主张 => 2 个批次（12 + 1）；第一批即触发停止信号。
    with pytest.raises(AgentBudgetExceeded):
        verify_review_claims(_numbered_text(13), _numbered_cards(13), llm=BudgetStopLLM())
    assert calls["n"] == 1


def test_entailment_lease_stale_signal_propagates():
    """T01：租约失权（AgentExecutionStale）同样不得被蕴含批处理吞掉。"""
    from app.agent.execution_budget import AgentExecutionStale

    class StaleLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise AgentExecutionStale("execution lease expired")

    with pytest.raises(AgentExecutionStale):
        verify_review_claims(_numbered_text(13), _numbered_cards(13), llm=StaleLLM())


def test_entailment_ordinary_provider_failure_is_recoverable_and_recorded():
    """T02：普通 provider 失败记为未完成、保留原因，不伪造否定判定。"""
    from app.core.exceptions import LLMInvocationError

    class TimeoutLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise LLMInvocationError("LLM 调用失败: request timed out")

    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。", [_card()], llm=TimeoutLLM(),
    )
    claim = report["claims"][0]
    assert claim["verification_status"] == "not_completed"
    assert "entailment_not_verified" in claim["issues"]
    assert "claim_contradicted_by_evidence" not in claim["issues"]
    stats = report["entailment_llm_batch_stats"]
    assert stats["failed"] == 1
    assert stats["claims_submitted"] == 1
    assert stats.get("failure_reasons")


def test_verify_claims_node_propagates_budget_stop_instead_of_degrading():
    """T01：验证节点遇预算停止信号必须传播，不得降级成 skip_verification。"""
    from app.agent.execution_budget import AgentBudgetExceeded
    from app.agent.nodes.verification import verify_claims_node

    class StopLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise AgentBudgetExceeded("token reservation exceeds remaining budget")

    state = {
        "review": "该方法在 FEVER 数据集上达到 80% accuracy [p1]。",
        "paper_cards": [_card()],
        "writing_plans": [{"section": "related_work"}],
    }
    with pytest.raises(AgentBudgetExceeded):
        verify_claims_node(state, llm=StopLLM())
    # 停止信号不得被写成普通验证失败/降级标记。
    assert not any("verify_claims" in str(e) for e in state.get("errors") or [])


def test_uncited_structural_failure_is_verified_unsupported_not_reverify_eligible():
    """T03：无引用主张是确定性结构失败，不再标 not_completed，也不送进蕴含。"""
    class NoopLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise AssertionError("structural failure must not be submitted for entailment")

    report = verify_review_claims(
        "该方法在 FEVER 数据集上达到 80% accuracy。",  # 无引用
        [_card()],
        llm=NoopLLM(),
    )
    claim = report["claims"][0]
    assert claim["support_status"] == "unsupported"
    assert claim["verification_status"] == "verified"
    assert claim["verification_method"] == "deterministic"
    assert "entailment_not_verified" not in claim["issues"]
    assert report["unverified"] == 0
    semantic = report["semantic_verification"]
    assert semantic["structural_failures"] == 1
    assert semantic["not_completed"] == 0
    assert semantic["llm_claims_submitted"] == 0


def test_mixed_batch_submits_only_semantically_verifiable_claims():
    """T04：混合批次只提交有引用有片段的主张，结构性失败不进入蕴含请求。"""
    import json
    import re

    submitted: list[str] = []

    class RecordingLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            match = re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S)
            payload = json.loads(match.group(1)) if match else []
            submitted.extend(str(item["claim_id"]) for item in payload)
            return json.dumps({"results": [
                {"claim_id": item["claim_id"], "label": "entailed", "confidence": 0.95}
                for item in payload
            ]})

    text = (
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
        "另一项方法在 SciFact 数据集上达到 95% accuracy。"  # 无引用，结构性失败
    )
    report = verify_review_claims(text, [_card()], llm=RecordingLLM())
    semantic = report["semantic_verification"]
    assert semantic["llm_claims_submitted"] == 1
    assert semantic["structural_failures"] == 1
    assert len(submitted) == 1 and submitted[0].startswith("c001")
    claims = {c["claim_id"]: c for c in report["claims"]}
    assert claims["c002"]["verification_status"] == "verified"
    assert claims["c002"]["support_status"] == "unsupported"


def test_cached_negative_verdict_is_reused_and_not_auto_cleared():
    """T04：已缓存的 contradicted 判定按指纹复用，重试不自动清除也不重复提交。"""
    import json
    import re

    calls = {"n": 0}

    class ContradictLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            calls["n"] += 1
            payload = json.loads(re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S).group(1))
            return json.dumps({"results": [
                {"claim_id": item["claim_id"], "label": "contradicted", "confidence": 0.9}
                for item in payload
            ]})

    cache: dict = {}
    text = "该方法在 FEVER 数据集上达到 80% accuracy [p1]。"
    first = verify_review_claims(text, [_card()], llm=ContradictLLM(), entailment_cache=cache)
    second = verify_review_claims(text, [_card()], llm=ContradictLLM(), entailment_cache=cache)
    assert calls["n"] == 1  # 第二次复用缓存，不再提交
    assert first["claims"][0]["support_status"] == "unsupported"
    assert second["claims"][0]["support_status"] == "unsupported"
    assert "claim_contradicted_by_evidence" in second["claims"][0]["issues"]
    assert second["entailment_cache_stats"]["reused"] == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_access_failure_never_requests_entailment_or_becomes_pending(legacy):
    from app.tools.verify_claims import _verify_review_claims_legacy
    from app.agent.generation_recovery import reverify_eligible_claim_ids

    calls = []

    class LLM:
        def complete(self, *args, **kwargs):
            calls.append(kwargs)
            return '{"results": []}'

    verifier = _verify_review_claims_legacy if legacy else verify_review_claims
    report = verifier(
        "消融实验表明该模块在所有基线上均贡献了性能提升 [p1]。",
        [_card()], llm=LLM(),
    )
    assert calls == []
    assert report["unverified"] == 0
    claim = report["claims"][0]
    assert claim["verification_status"] == "verified"
    assert claim["verification_method"] == "deterministic"
    assert claim["support_status"] == "unsupported"
    assert "access_level_too_weak_for_claim" in claim["issues"]
    assert reverify_eligible_claim_ids({"claim_verification": report}) == []


@pytest.mark.parametrize("label", [None, "invalid"])
@pytest.mark.parametrize("legacy", [False, True])
def test_invalid_entailment_label_is_pending_and_not_cached(label, legacy):
    import json
    import re
    from app.tools.verify_claims import _verify_review_claims_legacy

    class LLM:
        def complete(self, prompt, **kwargs):
            payload = json.loads(re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S).group(1))
            return json.dumps({"results": [
                dict({"claim_id": item["claim_id"], "confidence": 0.95},
                     **({} if label is None else {"label": label}))
                for item in payload
            ]})

    cache = {}
    verifier = _verify_review_claims_legacy if legacy else verify_review_claims
    kwargs = {} if legacy else {"entailment_cache": cache}
    report = verifier(
        "该方法在 FEVER 数据集上达到 80% accuracy [p1]。", [_card()], llm=LLM(), **kwargs,
    )
    claim = report["claims"][0]
    assert claim["verification_status"] == "not_completed"
    assert "entailment_not_verified" in claim["issues"]
    assert "claim_not_entailed_by_evidence" not in claim["issues"]
    assert report["unverified"] == 1
    assert cache == {}
    if not legacy:
        assert report["semantic_verification"]["completed"] == 0
        assert report["entailment_llm_batch_stats"]["failure_reasons"]


def test_malformed_result_does_not_discard_other_valid_verdicts():
    import json
    import re

    class LLM:
        def complete(self, prompt, **kwargs):
            payload = json.loads(re.search(r"待验证项目：(\[.*?\])\n", prompt, re.S).group(1))
            return json.dumps({"results": [
                {"claim_id": payload[0]["claim_id"], "confidence": 0.95},
                {"claim_id": payload[1]["claim_id"], "label": "contradicted", "confidence": 0.95},
            ]})
    cache = {}
    report = verify_review_claims(_numbered_text(2), _numbered_cards(2), llm=LLM(), entailment_cache=cache)
    assert report["claims"][0]["verification_status"] == "not_completed"
    assert report["claims"][1]["verification_status"] == "verified"
    assert "claim_contradicted_by_evidence" in report["claims"][1]["issues"]
    assert report["semantic_verification"]["completed"] == 1
    assert report["semantic_verification"]["not_completed"] == 1
    assert len(cache) == 1
