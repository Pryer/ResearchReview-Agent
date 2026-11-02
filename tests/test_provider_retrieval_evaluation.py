"""付费评测恢复的截止时间约束；不调用 Provider。"""
from datetime import datetime
import json

import pytest

from scripts.e2e_reporting import digest
from scripts.run_provider_retrieval_evaluation import original_round_deadline


def test_first_round_deadline_is_derived_from_original_start():
    record = {"started_at": "2026-10-29T23:56:04+08:00", "limits": {"deadline_seconds": 1800}}
    assert original_round_deadline(record) == datetime.fromisoformat("2026-10-30T00:26:04+08:00")


def test_explicit_deadline_survives_later_resume_start():
    record = {"started_at": "2026-10-30T00:20:00+08:00", "limits": {"deadline_seconds": 1800},
              "round_deadline_at": "2026-10-30T00:26:04+08:00"}
    assert original_round_deadline(record) == datetime.fromisoformat("2026-10-30T00:26:04+08:00")


def test_legacy_resume_follows_immutable_parent_instead_of_resetting_clock(tmp_path):
    parent = tmp_path / "data" / "parent"
    parent.mkdir(parents=True)
    original = {"started_at": "2026-10-29T23:56:04+08:00", "limits": {"deadline_seconds": 1800}}
    (parent / "run.json").write_text(json.dumps(original))
    child = {"started_at": "2026-10-30T00:20:00+08:00", "limits": {"deadline_seconds": 1800},
             "parent_run": "data/parent", "parent_manifest_sha256": digest(original)}
    assert original_round_deadline(child, tmp_path) == datetime.fromisoformat("2026-10-30T00:26:04+08:00")
    original["limits"]["deadline_seconds"] = 3600
    (parent / "run.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="changed"):
        original_round_deadline(child, tmp_path)


def test_parent_chain_cannot_read_outside_data(tmp_path):
    with pytest.raises(ValueError, match="invalid parent"):
        original_round_deadline({"parent_run": "private"}, tmp_path)
