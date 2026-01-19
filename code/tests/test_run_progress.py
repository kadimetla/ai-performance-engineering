from __future__ import annotations

import json
from pathlib import Path

import mcp.mcp_server as mcp_server
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.harness import run_benchmarks


def test_progress_recorder_writes_payload(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_001", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="baseline_timing",
            phase_index=1,
            total_phases=2,
            step="ch01:demo",
        )
    )
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_001"
    assert data["current"]["phase"] == "baseline_timing"
    assert data["current"]["step"] == "ch01:demo"
    assert data["current"]["run_id"] == "run_001"


def test_job_status_includes_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "20250101_000000"
    progress_path = run_dir / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_002", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="optimized_timing",
            phase_index=2,
            total_phases=2,
            step="ch01:demo",
        )
    )

    job_id = "job-test-001"
    record = {
        "job_id": job_id,
        "tool": "aisp_run_benchmarks",
        "status": "running",
        "submitted_at": 0.0,
        "arguments": {},
        "run_id": "run_002",
        "run_dir": str(run_dir),
        "progress_path": str(progress_path),
    }
    with mcp_server._JOB_LOCK:
        mcp_server._JOB_STORE[job_id] = record
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "running"
        assert payload["progress"]["phase"] == "optimized_timing"
        assert payload["progress"]["run_id"] == "run_002"
    finally:
        with mcp_server._JOB_LOCK:
            mcp_server._JOB_STORE.pop(job_id, None)


def test_progress_phases_include_llm() -> None:
    phases = run_benchmarks.PROGRESS_PHASES
    for key in (
        "llm_analysis",
        "llm_patch_apply",
        "llm_patch_rebenchmark",
        "llm_patch_verify",
        "llm_explain",
    ):
        assert key in phases
