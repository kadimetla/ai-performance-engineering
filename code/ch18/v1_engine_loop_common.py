"""Shared scaffolding for V1 EngineCore/CoreClient polling loop demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class MockRequestOutput:
    """Minimal stand-in for vLLM's RequestOutput."""

    request_id: str
    delta_text: str = ""
    finished: bool = False


class MockEngineCore:
    """Scriptable EngineCore stub that surfaces (outputs, executed_flag) tuples."""

    def __init__(self, steps: Sequence[Tuple[List[MockRequestOutput], bool]]) -> None:
        self._steps = list(steps)
        self.calls = 0

    def step(self) -> Tuple[List[MockRequestOutput], bool]:
        if self.calls >= len(self._steps):
            return [], False
        outputs, executed = self._steps[self.calls]
        self.calls += 1
        return outputs, bool(executed)


class MockCoreClient:
    """Tracks finished IDs and produces simple all-done signals."""

    def __init__(self, expected_ids: Iterable[str]) -> None:
        self.expected_ids = set(expected_ids)
        self.finished_reported: List[str] = []

    def report_finished_ids(self, request_ids: List[str]) -> None:
        self.finished_reported.extend(request_ids)

    def is_all_done(self) -> bool:
        return self.expected_ids.issubset(set(self.finished_reported))


def build_demo_stack() -> Tuple[MockEngineCore, MockCoreClient]:
    """
    Build a deterministic mock stack:
      - Step 0: scheduler defers execution (executed_flag=False, no outputs)
      - Step 1: first tokens for two requests
      - Step 2: final token for req-0 (finished=True)
      - Step 3: final token for req-1 (finished=True)
    """
    schedule: List[Tuple[List[MockRequestOutput], bool]] = [
        ([], False),
        (
            [
                MockRequestOutput("req-0", delta_text="hello "),
                MockRequestOutput("req-1", delta_text="hola "),
            ],
            True,
        ),
        ([MockRequestOutput("req-0", delta_text="world", finished=True)], True),
        ([MockRequestOutput("req-1", delta_text="mundo", finished=True)], True),
    ]
    engine_core = MockEngineCore(schedule)
    core_client = MockCoreClient(expected_ids=["req-0", "req-1"])
    return engine_core, core_client
