from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

import mcp.server as mcp_server


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "mcp-profiles"
MICROBENCH_DIR = REPO_ROOT / "artifacts" / "mcp-microbench"
REPORT_OUTPUT = REPO_ROOT / "artifacts" / "mcp-report.pdf"
EXPORT_OUTPUT = REPO_ROOT / "artifacts" / "mcp-export.csv"
BENCH_FILE = REPO_ROOT / "benchmark_test_results.json"
PROFILE_FIXTURE_DIR = REPO_ROOT / "benchmark_profiles" / "ch4"
NSYS_SAMPLE = PROFILE_FIXTURE_DIR / "baseline_nccl_baseline.nsys-rep"
NCU_SAMPLE = PROFILE_FIXTURE_DIR / "baseline_nvlink_baseline.ncu-rep"


@dataclass(frozen=True)
class ToolCase:
    name: str
    params: Dict[str, Any]
    category: str
    slow: bool = False
    timeout: int = 15


CATEGORY_TOOLS: Dict[str, List[str]] = {
    "gpu": [
        "aisp_gpu_info",
        "aisp_gpu_bandwidth",
        "aisp_gpu_topology",
        "aisp_gpu_topology_matrix",
        "aisp_gpu_power",
    ],
    "system": [
        "aisp_system_software",
        "aisp_system_dependencies",
        "aisp_system_context",
        "aisp_system_capabilities",
        "aisp_cpu_memory_analysis",
        "aisp_system_parameters",
        "aisp_container_limits",
        "aisp_full_system_analysis",
    ],
    "benchmarking": [
        "aisp_available_benchmarks",
        "aisp_benchmark_targets",
        "aisp_list_chapters",
        "aisp_run_benchmarks",
        "aisp_verify_benchmarks",
        "aisp_benchmark_report",
        "aisp_benchmark_export",
        "aisp_benchmark_compare_runs",
    ],
    "analysis": [
        "aisp_analyze_bottlenecks",
        "aisp_analyze_pareto",
        "aisp_analyze_scaling",
        "aisp_analyze_stacking",
        "aisp_analyze_whatif",
    ],
    "optimization": [
        "aisp_recommend",
        "aisp_optimize_roi",
        "aisp_optimize_techniques",
    ],
    "distributed": [
        "aisp_distributed_plan",
        "aisp_distributed_nccl",
        "aisp_launch_plan",
    ],
    "inference": [
        "aisp_inference_vllm",
        "aisp_inference_quantization",
    ],
    "ai_llm": [
        "aisp_ask",
        "aisp_explain",
        "aisp_ai_status",
    ],
    "profiling": [
        "aisp_profile_flame",
        "aisp_profile_memory",
        "aisp_profile_kernels",
        "aisp_profile_roofline",
        "aisp_profile_nsys",
        "aisp_profile_ncu",
        "aisp_nsys_summary",
        "aisp_nsys_ncu_available",
        "aisp_compare_nsys",
        "aisp_compare_ncu",
    ],
    "exports": [
        "aisp_export_csv",
        "aisp_export_pdf",
        "aisp_export_html",
    ],
    "microbench": [
        "aisp_test_speed",
        "aisp_test_roofline",
        "aisp_test_disk",
        "aisp_test_pcie",
        "aisp_test_mem_hierarchy",
        "aisp_test_tensor_core",
        "aisp_test_sfu",
        "aisp_test_network_loopback",
        "aisp_test_network",
    ],
    "code_analysis": [
        "aisp_warp_divergence",
        "aisp_bank_conflicts",
        "aisp_memory_access",
        "aisp_comm_overlap",
        "aisp_data_loading",
        "aisp_energy_analysis",
        "aisp_predict_scaling",
    ],
    "huggingface": [
        "aisp_hf_search",
        "aisp_hf_trending",
    ],
    "cluster_cost": [
        "aisp_cluster_slurm",
        "aisp_cost_estimate",
    ],
    "utility": [
        "aisp_status",
        "aisp_context_summary",
        "aisp_context_full",
        "aisp_triage",
        "aisp_help",
        "aisp_suggest_tools",
    ],
}

SLOW_TOOLS = {
    "aisp_gpu_bandwidth",
    "aisp_run_benchmarks",
    "aisp_verify_benchmarks",
    "aisp_profile_nsys",
    "aisp_profile_ncu",
    "aisp_profile_flame",
    "aisp_profile_memory",
    "aisp_profile_kernels",
    "aisp_profile_roofline",
    "aisp_compare_nsys",
    "aisp_compare_ncu",
    "aisp_test_speed",
    "aisp_test_roofline",
    "aisp_test_disk",
    "aisp_test_pcie",
    "aisp_test_mem_hierarchy",
    "aisp_test_tensor_core",
}

BENCHMARK_SLOW_TOOLS = {"aisp_run_benchmarks", "aisp_verify_benchmarks"}

TOOL_PARAMS: Dict[str, Dict[str, Any]] = {
    "aisp_run_benchmarks": {
        "targets": ["ch10:atomic"],
        "profile": "minimal",
        "llm_analysis": False,
    },
    "aisp_verify_benchmarks": {"targets": ["ch10:atomic"]},
    "aisp_benchmark_report": {
        "data_file": str(BENCH_FILE),
        "output": str(REPORT_OUTPUT),
        "format": "pdf",
        "title": "MCP Report",
        "author": "MCP Tests",
    },
    "aisp_benchmark_export": {
        "data_file": str(BENCH_FILE),
        "format": "csv",
        "output": str(EXPORT_OUTPUT),
    },
    "aisp_benchmark_compare_runs": {
        "baseline": str(BENCH_FILE),
        "candidate": str(BENCH_FILE),
        "top": 3,
    },
    "aisp_analyze_whatif": {"max_vram_gb": 24, "max_latency_ms": 50, "include_context": False},
    "aisp_recommend": {"model_size": 7, "gpus": 1, "goal": "throughput", "include_context": False},
    "aisp_distributed_plan": {"model_size": 7, "gpus": 4, "nodes": 1, "include_context": False},
    "aisp_distributed_nccl": {"nodes": 1, "gpus": 4, "include_context": False},
    "aisp_launch_plan": {"model_params": 7, "nodes": 1, "gpus": 2, "batch_size": 1},
    "aisp_inference_vllm": {"model": "7b", "target": "throughput", "include_context": False},
    "aisp_inference_quantization": {"model_size": 7, "include_context": False},
    "aisp_ask": {"question": "What is tensor parallelism?", "include_context": False},
    "aisp_explain": {"concept": "warp divergence", "include_context": False},
    "aisp_profile_nsys": {
        "command": ["python", "-c", "print('nsys')"],
        "output_name": "mcp_nsys_test",
        "output_dir": str(ARTIFACT_DIR),
        "preset": "light",
        "full_timeline": False,
        "trace_forks": False,
        "include_context": False,
    },
    "aisp_profile_ncu": {
        "command": ["python", "-c", "print('ncu')"],
        "output_name": "mcp_ncu_test",
        "output_dir": str(ARTIFACT_DIR),
        "workload_type": "memory_bound",
        "include_context": False,
    },
    "aisp_compare_nsys": {"profiles_dir": str(PROFILE_FIXTURE_DIR), "include_context": False},
    "aisp_compare_ncu": {"profiles_dir": str(PROFILE_FIXTURE_DIR), "include_context": False},
    "aisp_nsys_summary": {"report_path": str(NSYS_SAMPLE), "include_context": False},
    "aisp_export_csv": {"detailed": False, "include_context": False},
    "aisp_export_pdf": {"include_context": False},
    "aisp_export_html": {"include_context": False},
    "aisp_test_speed": {"gemm_size": 256, "mem_size_mb": 8, "mem_stride": 64, "include_context": False},
    "aisp_test_roofline": {"size_mb": 8, "strides": [64, 128]},
    "aisp_test_disk": {"file_size_mb": 8, "block_size_kb": 128, "tmp_dir": str(MICROBENCH_DIR)},
    "aisp_test_pcie": {"size_mb": 8, "iters": 1},
    "aisp_test_mem_hierarchy": {"size_mb": 8, "stride": 64},
    "aisp_test_tensor_core": {"size": 512, "precision": "fp16"},
    "aisp_test_sfu": {"elements": 1024},
    "aisp_test_network_loopback": {"size_mb": 4, "port": 50007},
    "aisp_hf_search": {"query": "llama", "limit": 3},
    "aisp_cluster_slurm": {"model": "7b", "nodes": 1, "gpus": 2},
    "aisp_cost_estimate": {"model_size": 7, "training_tokens": 100, "provider": "aws"},
    "aisp_help": {"query": "benchmark"},
    "aisp_suggest_tools": {"query": "profile this model"},
}


def _build_cases() -> List[ToolCase]:
    cases: List[ToolCase] = []
    for category, tools in CATEGORY_TOOLS.items():
        for name in tools:
            params = TOOL_PARAMS.get(name, {})
            cases.append(
                ToolCase(
                    name=name,
                    params=params,
                    category=category,
                    slow=name in SLOW_TOOLS,
                    timeout=900 if name in BENCHMARK_SLOW_TOOLS else 600 if name in SLOW_TOOLS else 60,
                )
            )
    return cases


ALL_TOOL_CASES = _build_cases()
SLOW_TOOL_CASES = [case for case in ALL_TOOL_CASES if case.slow]


@pytest.fixture(scope="module", autouse=True)
def prepare_artifacts() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MICROBENCH_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mcp_server,
        "_context_snapshot",
        lambda: {"summary": {}, "full": {}, "summary_age_seconds": 0, "full_age_seconds": 0},
    )
    return mcp_server.MCPServer()


def _payload_from_result(result: mcp_server.ToolResult) -> Dict[str, Any]:
    assert result.content, "Tool response must include content"
    entry = result.content[0]
    assert entry.get("type") == "application/json", f"Unexpected content type: {entry.get('type')}"
    payload = entry.get("json")
    assert isinstance(payload, dict), "Payload must be a JSON object"
    return payload


def _call_with_timeout(server: mcp_server.MCPServer, case: ToolCase) -> mcp_server.ToolResult:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(server.call_tool, case.name, case.params)
        return fut.result(timeout=case.timeout)


def _case_ids(cases: Iterable[ToolCase]) -> List[str]:
    return [case.name for case in cases]


def test_expected_tool_registration_matches_catalog():
    expected = {case.name for case in ALL_TOOL_CASES}
    registered = set(mcp_server.TOOLS.keys())
    assert expected == registered, "Tool catalog must mirror MCP server registry"
    assert len(expected) == 76


def test_tool_list_protocol_matches_registration(server: mcp_server.MCPServer):
    tool_list = server.get_tool_list()
    names = {tool["name"] for tool in tool_list}
    expected = {case.name for case in ALL_TOOL_CASES}
    assert names == expected


@pytest.mark.parametrize("case", ALL_TOOL_CASES, ids=_case_ids(ALL_TOOL_CASES))
def test_tool_call_returns_json_envelope(server: mcp_server.MCPServer, case: ToolCase):
    result = server.call_tool(case.name, case.params)
    payload = _payload_from_result(result)
    assert payload["tool"] == case.name
    assert payload["status"] == "ok"
    assert "result" in payload
    assert "context_summary" in payload


def test_mcp_protocol_round_trip(server: mcp_server.MCPServer):
    async def _exercise():
        init = await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init and init["result"]["protocolVersion"] == "2024-11-05"

        tool_list = await server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert tool_list and "tools" in tool_list["result"]

        sample_tool = "aisp_status"
        call = await server.handle_message(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": sample_tool, "arguments": {}}}
        )
        assert call and call["result"]["content"][0]["type"] == "application/json"
        payload = call["result"]["content"][0]["json"]
        assert payload["tool"] == sample_tool

    asyncio.run(_exercise())


@pytest.mark.parametrize("case", SLOW_TOOL_CASES, ids=_case_ids(SLOW_TOOL_CASES))
def test_slow_tools_opt_in_execution(server: mcp_server.MCPServer, case: ToolCase, monkeypatch: pytest.MonkeyPatch):
    # Remove the pytest stub guard so slow tools execute their real logic.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    result = _call_with_timeout(server, case)
    payload = _payload_from_result(result)
    assert payload["tool"] == case.name
    assert payload["status"] in {"ok", "error"}


def test_benchmark_export_runs_inprocess(server: mcp_server.MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    output_path = tmp_path / "export.json"
    params = {"data_file": str(BENCH_FILE), "format": "json", "output": str(output_path)}
    result = server.call_tool("aisp_benchmark_export", params)
    payload = _payload_from_result(result)
    assert payload["tool"] == "aisp_benchmark_export"
    assert payload["result"].get("output") == str(output_path)
    assert output_path.exists()
