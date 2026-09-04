"""并发卡片抽取与受控线程池测试。"""

from app.tools.extract_paper_card import batch_extract_paper_cards
from app.schemas.paper_schema import PaperCard

def test_batch_extract_paper_cards_concurrency_and_ordering():
    papers = [
        {
            "paper_id": f"p{i}",
            "title": f"Test Title {i}",
            "authors": [f"Author {i}"],
            "year": 2024,
            "abstract": f"Abstract for paper {i} solving problem {i}.",
            "venue": "Test Venue",
        }
        for i in range(10)
    ]
    parsed_texts = {}

    cards = batch_extract_paper_cards(papers, parsed_texts, llm=None, topic="测试主题", max_workers=4)

    assert len(cards) == 10
    # 验证顺序完全保持
    for i, card in enumerate(cards):
        assert card.paper_id == f"p{i}"
        assert card.title == f"Test Title {i}"
        assert isinstance(card, PaperCard)

def test_batch_extract_paper_cards_handles_worker_exception():
    papers = [
        {"paper_id": "good_1", "title": "Good Paper 1", "abstract": "Good abstract 1"},
        {"paper_id": "bad_1", "title": None, "abstract": None},  # Missing minimal fields
        {"paper_id": "good_2", "title": "Good Paper 2", "abstract": "Good abstract 2"},
    ]
    parsed_texts = {}

    cards = batch_extract_paper_cards(papers, parsed_texts, llm=None, topic="测试主题", max_workers=2)

    assert len(cards) == 3
    assert cards[0].paper_id == "good_1"
    assert cards[2].paper_id == "good_2"
    # Bad card still extracted safely via fallback rule-based
    assert cards[1].paper_id == "bad_1"
