"""为评测子集中无摘要的真实候选补全摘要（P0 元数据补全）。

来源策略（均为无需登录的开放 API）：
1. OpenAlex 按 DOI 直查，重建 abstract_inverted_index（覆盖 crossref 缺口）；
2. Semantic Scholar 按 DOI 回退（有速率限制，串行 + 退避）；
3. CNKI 与无 DOI 记录跳过，绝不编造摘要。

结果缓存到 data/retrieval_eval/enrichment_cache.json，重跑不重复请求。
输出补全后的子集快照（候选身份不变，只新增 abstract 及 provenance 标记）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE_PATH = Path("data/retrieval_eval/enrichment_cache.json")
HEADERS = {"User-Agent": "research-review-agent-eval/1.0 (mailto:eval@example.org)"}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _openalex_abstract(doi: str) -> str | None:
    url = f"https://api.openalex.org/works/https://doi.org/{requests.utils.quote(doi)}"
    resp = requests.get(url, headers=HEADERS, timeout=20,
                        params={"mailto": "eval@example.org"})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    inverted = (resp.json() or {}).get("abstract_inverted_index")
    if not isinstance(inverted, dict) or not inverted:
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted.items():
        for index in indices:
            positions[index] = word
    return " ".join(positions[index] for index in sorted(positions)) or None


def _s2_abstract(doi: str) -> str | None:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    resp = requests.get(url, headers=HEADERS, timeout=20,
                        params={"fields": "abstract,externalIds,title"})
    if resp.status_code in (404, 400):
        return None
    resp.raise_for_status()
    abstract = (resp.json() or {}).get("abstract")
    return abstract.strip() if isinstance(abstract, str) and abstract.strip() else None


def enrich_abstract(paper: dict, cache: dict) -> tuple[str | None, str]:
    doi = str(paper.get("doi") or "").strip()
    if paper.get("source") == "cnki":
        return None, "skipped_cnki_requires_login"
    if not doi:
        return None, "skipped_no_doi"
    if doi in cache:
        entry = cache[doi]
        return (entry.get("abstract") or None), entry.get("source") or "cached_miss"
    last_error = ""
    for name, getter, delay in (("openalex", _openalex_abstract, 0.2),
                                ("semantic_scholar", _s2_abstract, 3.2)):
        try:
            abstract = getter(doi)
            if abstract:
                cache[doi] = {"abstract": abstract[:3000], "source": name}
                _save_cache(cache)
                time.sleep(delay)
                return abstract[:3000], name
            last_error = f"{name}:no_abstract"
        except requests.RequestException as exc:
            last_error = f"{name}:{type(exc).__name__}_{getattr(exc.response, 'status_code', '')}"
            time.sleep(delay)
    cache[doi] = {"abstract": None, "source": f"miss:{last_error}"}
    _save_cache(cache)
    return None, f"miss:{last_error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    cache = _load_cache()
    stats = {"filled": 0, "miss": 0, "skipped": 0, "by_source": {}}
    for task in snapshot.get("tasks") or []:
        for paper in task["candidate_papers"]:
            if str(paper.get("abstract") or "").strip():
                continue
            abstract, source = enrich_abstract(paper, cache)
            if abstract:
                paper["abstract"] = abstract
                paper["_eval_abstract_enriched_from"] = source
                stats["filled"] += 1
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            else:
                stats["skipped" if source.startswith("skipped") else "miss"] += 1
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
    Path(args.output).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
