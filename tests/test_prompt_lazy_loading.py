"""Prompt 模块边界与惰性加载回归测试。"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_clean_interpreter(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_prompt_package_and_legacy_catalog_are_lazy() -> None:
    _run_in_clean_interpreter(
        """
        import sys
        import app.prompt

        assert not [
            name for name in sys.modules
            if name.startswith("app.prompt.")
        ]

        import app.prompt_catalog as catalog
        assert "app.prompt.intent" not in sys.modules
        assert "意图识别" in catalog.INTENT_RECOGNITION_PROMPT
        assert "app.prompt.intent" in sys.modules
        assert "app.prompt.writing.introduction" not in sys.modules
        """
    )


def test_writer_prompt_loads_only_when_writer_task_requests_it() -> None:
    _run_in_clean_interpreter(
        """
        import sys
        import app.tools.write_deliverable as writer

        assert "app.prompt.writing.deliverable" not in sys.modules
        assert "WritingPlan" in writer.WRITER_PROMPT
        assert "app.prompt.writing.deliverable" in sys.modules
        assert "app.prompt.writing.introduction" not in sys.modules
        assert "app.prompt.writing.related_work" not in sys.modules
        assert "app.prompt.writing.literature_review" not in sys.modules
        """
    )
