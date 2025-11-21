"""Optimized V1 EngineCore/CoreClient polling loop with KV reclamation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Set

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:  # Optional, only for real vLLM runs.
    from vllm.v1.outputs import RequestOutput  # type: ignore
except Exception:  # pragma: no cover - vLLM may be absent locally
    RequestOutput = None  # type: ignore

from ch18.v1_engine_loop_common import MockRequestOutput, build_demo_stack

OutputType = RequestOutput if RequestOutput is not None else MockRequestOutput


def run_engine_loop(engine_core: Any, core_client: Any) -> Iterator[OutputType]:
    """
    Poll the engine, emit RequestOutputs as they are ready, and free KV cache pages.

    Mirrors the V1 guidance: keep polling even if executed_flag is False, dedupe
    finished IDs, and report them to the core client so paged KV cache can be
    reclaimed.
    """
    finished: Set[str] = set()

    while True:
        outputs, _executed = engine_core.step()

        for ro in outputs:
            yield ro

        newly_finished: Dict[str, OutputType] = {
            ro.request_id: ro for ro in outputs if getattr(ro, "finished", False) and ro.request_id not in finished
        }
        if newly_finished:
            core_client.report_finished_ids(list(newly_finished.keys()))
            finished.update(newly_finished.keys())

        if core_client.is_all_done():
            break


def _demo() -> None:
    engine_core, core_client = build_demo_stack()
    outputs = list(run_engine_loop(engine_core, core_client))
    summary = {
        "steps": engine_core.calls,
        "tokens": "".join(ro.delta_text for ro in outputs),
        "reported_finished": list(core_client.finished_reported),
        "all_done": core_client.is_all_done(),
    }
    print("Optimized loop demo:", summary)


if __name__ == "__main__":
    _demo()
