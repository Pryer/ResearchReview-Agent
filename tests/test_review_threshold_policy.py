"""集中阈值策略的默认值、覆盖和诊断快照测试。"""

from app.core.config import ReviewThresholdPolicy, Settings


def test_review_threshold_policy_preserves_defaults():
    policy = ReviewThresholdPolicy()
    assert policy.version == "review-thresholds-v1"
    assert policy.route_min_core_evidence == 3
    assert policy.claim_support_similarity == 0.35
    assert policy.synthesis_fulltext_support_rate == 0.80
    assert policy.snapshot()["version"] == policy.version


def test_settings_can_override_review_thresholds(monkeypatch):
    monkeypatch.setenv("ROUTE_VALIDATOR_MIN_CORE_EVIDENCE", "5")
    monkeypatch.setenv("CLAIM_SUPPORT_SIMILARITY", "0.45")
    monkeypatch.setenv("SYNTHESIS_FULLTEXT_SUPPORT_RATE", "0.9")
    settings = Settings()
    assert settings.route_validator_min_core_evidence == 5
    assert settings.claim_support_similarity == 0.45
    assert settings.synthesis_fulltext_support_rate == 0.9
