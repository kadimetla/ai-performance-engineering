"""Baseline V1 engine polling loop (stops on an idle step and leaks KV cache)."""

from __future__ import annotations

import sys
from typing import Any, Iterator, List

from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.v1_engine_loop_common import MockRequestOutput, build_demo_stack


def baseline_engine_loop(
    engine_core: Any, core_client: Any
) -> Iterator[MockRequestOutput]:
    """
    Naive polling loop that assumes an idle step means the engine is done.

    This is a pre-V1 style loop: it stops when EngineCore reports executed=False
    and yields no outputs, which can strand queued work and leave KV pages alive.
    """
    while True:
        outputs, executed = engine_core.step()
        for ro in outputs:
            yield ro

        finished_ids: List[str] = [ro.request_id for ro in outputs if getattr(ro, "finished", False)]
        if finished_ids:
            core_client.report_finished_ids(finished_ids)

        if not executed and not outputs:
            break


def _demo() -> None:
    engine_core, core_client = build_demo_stack()
    outputs = list(baseline_engine_loop(engine_core, core_client))
    summary = {
        "steps": engine_core.calls,
        "tokens": "".join(ro.delta_text for ro in outputs),
        "reported_finished": list(core_client.finished_reported),
        "all_done": core_client.is_all_done(),
    }
    print("Baseline loop demo:", summary)


if __name__ == "__main__":
    _demo()
