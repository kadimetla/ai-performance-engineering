from __future__ import annotations

import mcp.mcp_server as mcp_server


def test_apply_patches_requires_llm_analysis_or_force() -> None:
    result = mcp_server.tool_run_benchmarks(
        {
            "targets": ["ch10:atomic_reduction"],
            "apply_patches": True,
        }
    )
    assert result.get("success") is False
    assert "apply_patches" in (result.get("error") or "")


def test_rebenchmark_requires_apply_patches() -> None:
    result = mcp_server.tool_run_benchmarks(
        {
            "targets": ["ch10:atomic_reduction"],
            "rebenchmark_llm_patches": True,
            "llm_analysis": True,
        }
    )
    assert result.get("success") is False
    assert "rebenchmark_llm_patches" in (result.get("error") or "")


def test_llm_explain_requires_rebenchmark() -> None:
    result = mcp_server.tool_run_benchmarks(
        {
            "targets": ["ch10:atomic_reduction"],
            "apply_patches": True,
            "llm_analysis": True,
            "llm_explain": True,
        }
    )
    assert result.get("success") is False
    assert "llm_explain" in (result.get("error") or "")
