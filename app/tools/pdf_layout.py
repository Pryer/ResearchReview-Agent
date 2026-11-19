"""可选 PDF 版面检测，保留可由原页面读取的坐标。"""

from collections import Counter
import re
from threading import Lock

from app.utils.text_cleaner import clean_parsed_text

_TABLE_ANALYSIS_LOCK = Lock()


def _cell_anchor(page, page_text, text, box, textpage):
    if not text or "\n" in text:
        return None
    # WHY: 同页重复单元格没有唯一字符锚点，仍可用其原页矩形回查。
    matches = list(re.finditer(r"(?m)^" + re.escape(text) + r"$", page_text))
    if len(matches) == 1:
        hits = page.search_for(text, textpage=textpage)
        if len(hits) == 1:
            x0, y0, x1, y1 = box
            hit = hits[0]
            if x0 <= hit.x0 and y0 <= hit.y0 and x1 >= hit.x1 and y1 >= hit.y1:
                return {"char_start": matches[0].start(), "char_end": matches[0].end()}
    return None


def inspect_tables(page, page_text):
    # WHY: PyMuPDF 1.24 的表格检测及单元格坐标计算共享模块级缓冲区。
    with _TABLE_ANALYSIS_LOCK:
        tables = []
        textpage = page.get_textpage()
        for table in page.find_tables(strategy="lines_strict").tables:
            rows = []
            for row in table.rows:
                cells = []
                for box in row.cells:
                    if box is None:
                        cells.append(None)
                        continue
                    text = clean_parsed_text(page.get_textbox(box))
                    cells.append({
                        "text": text,
                        "bbox": list(box),
                        "anchor": _cell_anchor(page, page_text, text, box, textpage),
                    })
                rows.append(cells)
            tables.append({"bbox": list(table.bbox), "rows": rows})
        return tables


def two_column_text(page, page_text):
    """在明确的双栏区间内按栏读取。全宽块保持在各区间的边界。"""
    blocks = [block for block in page.get_text("blocks") if block[6] == 0]
    mid = (page.rect.x0 + page.rect.x1) / 2
    gutter = page.rect.width * 0.015
    ordered, band = [], []
    changed = False

    def flush():
        left = [block for block in band if block[2] <= mid - gutter]
        right = [block for block in band if block[0] >= mid + gutter]
        # 两栏至少各有两行且垂直跨度重叠。局部边注或错开的标题沿用原顺序。
        if (not left or not right
                or any(sum(len(block[4].strip().splitlines()) for block in side) < 2
                       for side in (left, right))
                or min(max(block[3] for block in side) for side in (left, right))
                <= max(min(block[1] for block in side) for side in (left, right))):
            ordered.extend(sorted(band, key=lambda block: block[5]))
            return False
        ordered.extend(sorted(left, key=lambda block: block[1]))
        ordered.extend(sorted(right, key=lambda block: block[1]))
        return True

    for block in sorted(blocks, key=lambda item: item[1]):
        if block[2] <= mid - gutter or block[0] >= mid + gutter:
            band.append(block)
            continue
        # WHY: 与两栏块垂直重叠的跨栏对象可能是浮动图注，无法确定阅读顺序。
        if any(item[3] > block[1] for item in band) or any(
            item[1] < block[3] and item[1] >= block[1] and item is not block for item in blocks
        ):
            return None
        changed = flush() or changed
        band.clear()
        ordered.append(block)
    changed = flush() or changed
    if not changed:
        return None
    candidate = clean_parsed_text("\n".join(block[4].strip("\n") for block in ordered))
    # 只有内容逐行守恒时才采用坐标排序。来源偏移由最终页面文本生成。
    if Counter(filter(None, candidate.splitlines())) != Counter(filter(None, page_text.splitlines())):
        return None
    return candidate


def analyze_layout(page, page_text):
    tables = inspect_tables(page, page_text)
    # WHY: 跨列的表格可能打断正文。检测到表格的页保持原文本，只提供独立网格坐标。
    reordered = None if tables else two_column_text(page, page_text)
    return reordered or page_text, {
        "reading_order": "two_columns" if reordered is not None else "native",
        "tables": tables,
    }
