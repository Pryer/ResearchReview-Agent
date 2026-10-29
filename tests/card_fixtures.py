"""验证器测试的显式来源夹具，不调用已删除的规则抽卡。"""
from app.tools.extract_paper_card import _attach_evidence, _dict_to_card, _quality_gate


def source_card(paper, parsed_text=None, **fields):
    source = "full_text" if parsed_text and parsed_text.get("full_text") else "abstract"
    data = fields or {"method": paper.get("abstract") or ""}
    card = _dict_to_card(data, paper, source, "fixture")
    return _quality_gate(_attach_evidence(card, paper, parsed_text))
