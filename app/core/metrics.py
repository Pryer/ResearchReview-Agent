"""轻量级进程内性能指标与 Token 消耗收集器。

记录各 Agent 节点的执行耗时、成功/失败次数，以及 LLM 调用的 Token 使用量（prompt/completion/total），
供监控 API 或链路分析使用。
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional


_MAX_SAMPLES_PER_NODE = 200


class MetricsCollector:
    """线程安全的进程内指标收集器。"""

    def __init__(self, max_samples_per_node: int = _MAX_SAMPLES_PER_NODE) -> None:
        self._lock = threading.Lock()
        self._max_samples = max_samples_per_node
        self._durations: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self._max_samples)
        )
        self._success_count: Dict[str, int] = defaultdict(int)
        self._failure_count: Dict[str, int] = defaultdict(int)
        self._other_status_count: Dict[str, int] = defaultdict(int)
        self._source_return_counts: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self._max_samples)
        )
        # LLM Token 监控
        self._llm_calls_total = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._llm_usage_by_model: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        )
        self._llm_usage_by_operation: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        )
        self._llm_durations: Deque[int] = deque(maxlen=self._max_samples)

    def record_step(
        self,
        step_name: str,
        status: str,
        duration_ms: int | None,
    ) -> None:
        """记录一次节点执行结果（由 ``append_step`` 自动调用）。"""
        with self._lock:
            if duration_ms is not None:
                self._durations[step_name].append(int(duration_ms))
            if status == "success":
                self._success_count[step_name] += 1
            elif status == "failed":
                self._failure_count[step_name] += 1
            else:
                self._other_status_count[step_name] += 1

    def record_source(self, source: str, success: bool, count: int) -> None:
        """记录一次数据源检索结果。"""
        with self._lock:
            key = f"source:{source}"
            if success:
                self._success_count[key] += 1
            else:
                self._failure_count[key] += 1
            self._source_return_counts[key].append(count)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int = 0,
        operation: str = "completion",
        provider: str = "",
    ) -> None:
        """记录一次 LLM 调用的 Token 使用情况与耗时。"""
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        total_tokens = prompt_tokens + completion_tokens

        with self._lock:
            self._llm_calls_total += 1
            self._prompt_tokens_total += prompt_tokens
            self._completion_tokens_total += completion_tokens
            if duration_ms > 0:
                self._llm_durations.append(int(duration_ms))

            model_key = model or "unknown_model"
            m_stat = self._llm_usage_by_model[model_key]
            m_stat["prompt_tokens"] += prompt_tokens
            m_stat["completion_tokens"] += completion_tokens
            m_stat["total_tokens"] += total_tokens
            m_stat["calls"] += 1

            op_key = operation or "completion"
            o_stat = self._llm_usage_by_operation[op_key]
            o_stat["prompt_tokens"] += prompt_tokens
            o_stat["completion_tokens"] += completion_tokens
            o_stat["total_tokens"] += total_tokens
            o_stat["calls"] += 1

    def get_token_report(self) -> Dict[str, Any]:
        """获取 Token 消耗明细快照。"""
        with self._lock:
            durations = list(self._llm_durations)
            return {
                "total_calls": self._llm_calls_total,
                "total_prompt_tokens": self._prompt_tokens_total,
                "total_completion_tokens": self._completion_tokens_total,
                "total_tokens": self._prompt_tokens_total + self._completion_tokens_total,
                "avg_llm_duration_ms": round(statistics.mean(durations), 1) if durations else None,
                "by_model": {k: dict(v) for k, v in self._llm_usage_by_model.items()},
                "by_operation": {k: dict(v) for k, v in self._llm_usage_by_operation.items()},
            }

    def get_report(self) -> Dict[str, Any]:
        """生成当前汇总报告（包含节点耗时、数据源及 Token 消耗）。"""
        with self._lock:
            all_keys = (
                set(self._durations)
                | set(self._success_count)
                | set(self._failure_count)
                | set(self._other_status_count)
            )
            node_names = sorted(k for k in all_keys if not k.startswith("source:"))
            source_names = sorted(
                k[len("source:"):] for k in all_keys if k.startswith("source:")
            )

            nodes: Dict[str, Any] = {}
            for name in node_names:
                durations = list(self._durations.get(name, []))
                success = self._success_count.get(name, 0)
                failure = self._failure_count.get(name, 0)
                other = self._other_status_count.get(name, 0)
                total = success + failure
                nodes[name] = {
                    "success_count": success,
                    "failure_count": failure,
                    "other_status_count": other,
                    "success_rate": round(success / total, 4) if total else None,
                    "sample_count": len(durations),
                    "avg_duration_ms": (
                        round(statistics.mean(durations), 1) if durations else None
                    ),
                    "p95_duration_ms": (
                        round(_percentile(durations, 0.95), 1) if durations else None
                    ),
                    "max_duration_ms": max(durations) if durations else None,
                }

            sources: Dict[str, Any] = {}
            for name in source_names:
                key = f"source:{name}"
                success = self._success_count.get(key, 0)
                failure = self._failure_count.get(key, 0)
                total = success + failure
                counts = list(self._source_return_counts.get(key, []))
                sources[name] = {
                    "success_count": success,
                    "failure_count": failure,
                    "success_rate": round(success / total, 4) if total else None,
                    "avg_returned_count": (
                        round(statistics.mean(counts), 1) if counts else None
                    ),
                }

        # 在锁外生成 token 报告以避免重入
        tokens = self.get_token_report()
        return {"nodes": nodes, "sources": sources, "tokens": tokens}

    def reset(self) -> None:
        """清空所有统计（主要用于测试）。"""
        with self._lock:
            self._durations.clear()
            self._success_count.clear()
            self._failure_count.clear()
            self._other_status_count.clear()
            self._source_return_counts.clear()
            self._llm_calls_total = 0
            self._prompt_tokens_total = 0
            self._completion_tokens_total = 0
            self._llm_usage_by_model.clear()
            self._llm_usage_by_operation.clear()
            self._llm_durations.clear()


def _percentile(values: List[int], pct: float) -> float:
    """计算简单的百分位数（不依赖 numpy）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return float(ordered[index])


_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """获取全局单例收集器（进程内共享）。"""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector
