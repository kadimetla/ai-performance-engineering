"""Structured progress reporting for long-running runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

PROGRESS_SCHEMA_VERSION = "1.0"


@dataclass
class ProgressEvent:
    """Single progress event for a run."""

    run_id: str = ""
    timestamp: str = ""
    phase: str = ""
    phase_index: int = 0
    total_phases: int = 0
    step: str = ""
    step_detail: Optional[str] = None
    percent_complete: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = PROGRESS_SCHEMA_VERSION

    def ensure_defaults(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ProgressRecorder:
    """Write structured progress events to a JSON file."""

    def __init__(self, run_id: str, progress_path: Path, history_limit: int = 50) -> None:
        self.run_id = run_id
        self.progress_path = Path(progress_path)
        self.history_limit = max(1, history_limit)
        self._start_time = time.time()
        self._history: List[Dict[str, Any]] = []
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        if self.progress_path.exists():
            try:
                payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
                history = payload.get("history")
                if isinstance(history, list):
                    self._history = history
                current = payload.get("current", {})
                if isinstance(current, dict):
                    elapsed = current.get("elapsed_seconds")
                    if isinstance(elapsed, (int, float)) and elapsed >= 0:
                        self._start_time = time.time() - float(elapsed)
            except Exception:
                self._history = []

    def emit(self, event: ProgressEvent) -> None:
        if not event.run_id:
            event.run_id = self.run_id
        event.ensure_defaults()
        if event.elapsed_seconds is None:
            event.elapsed_seconds = time.time() - self._start_time
        if event.percent_complete is not None:
            event.percent_complete = max(0.0, min(100.0, float(event.percent_complete)))
        if event.eta_seconds is None and event.percent_complete:
            if event.percent_complete > 0:
                event.eta_seconds = (
                    event.elapsed_seconds * (100.0 - event.percent_complete) / event.percent_complete
                )
        event_dict = asdict(event)
        self._history.append(event_dict)
        payload = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "current": event_dict,
            "history": self._history[-self.history_limit:],
        }
        self._write_payload(payload)

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        tmp_path = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.progress_path)
