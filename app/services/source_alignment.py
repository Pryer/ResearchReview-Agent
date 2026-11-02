"""只规范空白以定位原文；大小写、标点、连字、断词和否定关系均不改写。"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QuoteLocation:
    start: int
    end: int
    text: str
    method: str


def _spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(source: str, quote: str) -> QuoteLocation | None:
    quote = quote.strip()
    if not quote:
        return None
    first = source.find(quote)
    if first >= 0:
        # WHY: 重复摘录无法唯一确认位置；不能默选第一次出现作为已验证的来源坐标。
        if source.find(quote, first + 1) >= 0:
            return None
        return QuoteLocation(first, first + len(quote), source[first:first + len(quote)], "exact")
    chars, starts, ends = [], [], []
    for match in re.finditer(r"\s+|[^\s]", source):
        chars.append(" " if match.group().isspace() else match.group())
        starts.append(match.start())
        ends.append(match.end())
    normalized = "".join(chars)
    target = _spaces(quote)
    first = normalized.find(target)
    if first < 0 or normalized.find(target, first + 1) >= 0:
        return None
    start, end = starts[first], ends[first + len(target) - 1]
    original = source[start:end]
    if _spaces(original) != target:
        return None
    # WHY: 回传的是原文连续区间，不是模型改写；后续仍按原始字符Offset和原文哈希回读。
    return QuoteLocation(start, end, original, "whitespace")
