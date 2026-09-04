"""打印当前进程内 MetricsCollector 收集到的节点/数据源性能报告。

用法：
    python scripts/show_metrics.py

注意：MetricsCollector 是进程内单例，只在当前 Python 进程运行期间累积数据。
若要观察一次完整 Agent 运行的指标，需要在同一进程内先触发运行（例如通过
API 服务），再调用此脚本对应的 `get_report()`。作为独立脚本运行时，通常
用于快速检查 collector 的可用性和输出格式。
"""

from __future__ import annotations

import json

from app.core.metrics import get_metrics_collector


def main() -> None:
    collector = get_metrics_collector()
    report = collector.get_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
