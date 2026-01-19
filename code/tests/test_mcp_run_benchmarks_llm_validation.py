from __future__ import annotations

import os

import mcp.mcp_server as mcp_server
from core.llm import reset_config


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


def test_llm_analysis_requires_configured_backend() -> None:
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "VLLM_API_BASE",
        "OLLAMA_HOST",
        "LLM_PROVIDER",
        "PERF_LLM_PROVIDER",
    ]
    original = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = ""
        reset_config()
        result = mcp_server.tool_run_benchmarks(
            {
                "targets": ["ch10:atomic_reduction"],
                "llm_analysis": True,
            }
        )
        assert result.get("success") is False
        assert "LLM" in (result.get("error") or "")
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_config()
