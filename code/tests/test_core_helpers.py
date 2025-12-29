import json
import subprocess
import sys
from pathlib import Path

from core import (
    profile_artifacts,
    compile_analysis,
    costs,
    optimization_reports,
    optimization_stack,
    whatif,
    ncu_analysis,
    profile_insights,
)


def test_profile_artifacts_empty(tmp_path: Path):
    # With no traces, loaders return default structures and messages
    empty_root = tmp_path
    assert profile_artifacts.load_flame_graph_data(empty_root).get("message")
    assert profile_artifacts.load_memory_timeline(empty_root).get("message")
    assert profile_artifacts.load_cpu_gpu_timeline(empty_root) is not None
    assert profile_artifacts.load_kernel_breakdown({"children": []})["kernels"] == []


def test_compile_analysis_empty():
    result = compile_analysis.load_compile_analysis(Path.cwd(), [])
    assert "compile_benchmarks" in result
    assert isinstance(result.get("recommendations"), list)


def test_costs_and_roi_empty():
    cost = costs.calculate_costs([], {"name": "H100"})
    assert cost["current_rate"] > 0
    roi_result = optimization_reports.compute_roi([], cost)
    assert "techniques" in roi_result


def test_optimization_stack_fallbacks():
    # These should not raise even if advanced_analysis is missing
    assert optimization_stack.get_all_optimizations()
    assert optimization_stack.get_optimization_playbooks()
    assert optimization_stack.calculate_compound_optimization([], {}) is not None
    assert optimization_stack.get_optimal_optimization_stack(2.0, "medium", {}) is not None


def test_whatif_and_ncu_empty(tmp_path: Path):
    scenarios = whatif.get_scenarios()
    assert scenarios.get("scenarios")
    ncu = ncu_analysis.load_ncu_deepdive(tmp_path)
    assert "available" in ncu


def test_profile_insights_bottlenecks_and_score():
    flame_data = {
        "value": 100.0,
        "children": [
            {"name": "gpu_memcpy", "value": 30},
            {"name": "python_function", "value": 20},
            {"name": "overhead", "value": 6},
        ],
    }
    kernel_data = {
        "summary": {"total_time_us": 80},
        "kernels": [
            {"name": "gemm_kernel", "time_us": 20},
            {"name": "copy_kernel", "time_us": 15},
        ],
    }
    hw_caps = {
        "features": [
            {"name": "TMA Copy", "supported": True, "optimization": "Use async copies"},
            {"name": "FP8 Tensor Cores", "supported": True, "optimization": "Enable FP8"},
        ],
        "architecture": "blackwell",
        "gpu": {"name": "B200"},
    }

    result = profile_insights.detect_bottlenecks(flame_data, kernel_data, hw_caps)
    assert result["bottlenecks"], "Expected bottlenecks from synthetic data"

    score = profile_insights.calculate_optimization_score(hw_caps, result, kernel_data)
    assert 0 <= score["score"] <= 100
    assert score["quick_wins"], "Feature-based quick wins should be suggested"


def test_profile_insights_ncu_comparison_and_recommendations(tmp_path: Path):
    baseline_csv = tmp_path / "demo_baseline_ncu.csv"
    optimized_csv = tmp_path / "demo_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\noccupancy,40\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\noccupancy,45\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics"), "CSV-based NCU comparison should return metrics"

    recs = profile_insights.generate_recommendations_from_profiles(
        {
            "ncu_comparison": comparison,
            "nsys_comparison": {"metrics": [{"name": "dram_util", "delta": -20}]},
        }
    )
    assert recs, "Recommendations should be produced from comparison data"


def test_profile_insights_nsys_comparison(tmp_path: Path):
    script = tmp_path / "nvtx_script.py"
    script.write_text(
        (
            "import torch\n"
            "assert torch.cuda.is_available(), 'CUDA required for nsys test'\n"
            "x = torch.randn(1024, device='cuda')\n"
            "with torch.cuda.nvtx.range('nsys_test_range'):\n"
            "    y = x * 2\n"
            "torch.cuda.synchronize()\n"
            "print(float(y[0].item()))\n"
        ),
        encoding="utf-8",
    )

    baseline_prefix = tmp_path / "baseline_test"
    subprocess.run(
        [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "-t",
            "cuda,nvtx,osrt",
            "-o",
            str(baseline_prefix),
            sys.executable,
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    baseline_rep = baseline_prefix.with_suffix(".nsys-rep")
    assert baseline_rep.exists()
    optimized_rep = tmp_path / "optimized_test.nsys-rep"
    optimized_rep.write_bytes(baseline_rep.read_bytes())

    comparison = profile_insights.compare_nsys_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics"), "nsys comparison should yield metrics when NVTX ranges exist"


def test_profile_insights_nsys_requires_pair_key(tmp_path: Path):
    (tmp_path / "baseline_one.nsys-rep").write_text("stub", encoding="utf-8")
    (tmp_path / "optimized_one.nsys-rep").write_text("stub", encoding="utf-8")
    (tmp_path / "baseline_two.nsys-rep").write_text("stub", encoding="utf-8")
    (tmp_path / "optimized_two.nsys-rep").write_text("stub", encoding="utf-8")

    comparison = profile_insights.compare_nsys_files(tmp_path)
    assert comparison is not None
    assert comparison.get("error")
    assert comparison.get("candidates")


def test_profile_insights_ncu_comparison_from_rep(tmp_path: Path):
    script = tmp_path / "ncu_script.py"
    script.write_text(
        (
            "import torch\n"
            "assert torch.cuda.is_available(), 'CUDA required for ncu test'\n"
            "x = torch.randn(512, 512, device='cuda')\n"
            "y = torch.randn(512, 512, device='cuda')\n"
            "z = x @ y\n"
            "torch.cuda.synchronize()\n"
            "print(float(z[0, 0]))\n"
        ),
        encoding="utf-8",
    )

    metrics = ",".join(
        [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    )

    out_prefix = tmp_path / "ncu_test"
    subprocess.run(
        [
            "ncu",
            "--metrics",
            metrics,
            "--force-overwrite",
            "-o",
            str(out_prefix),
            sys.executable,
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    rep = out_prefix.with_suffix(".ncu-rep")
    assert rep.exists()
    (tmp_path / "baseline_test.ncu-rep").write_bytes(rep.read_bytes())
    (tmp_path / "optimized_test.ncu-rep").write_bytes(rep.read_bytes())

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("kernel_comparison"), "ncu comparison should yield kernel comparisons"


def test_profile_insights_ncu_requires_pair_key(tmp_path: Path):
    baseline_csv = tmp_path / "pair_one_baseline_ncu.csv"
    optimized_csv = tmp_path / "pair_one_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\n")

    baseline_csv_two = tmp_path / "pair_two_baseline_ncu.csv"
    optimized_csv_two = tmp_path / "pair_two_optimized_ncu.csv"
    baseline_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,40\n")
    optimized_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,60\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("error")
    assert comparison.get("candidates")


def test_profile_insights_ncu_pair_key_selects_csv(tmp_path: Path):
    baseline_csv = tmp_path / "pair_one_baseline_ncu.csv"
    optimized_csv = tmp_path / "pair_one_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\n")

    baseline_csv_two = tmp_path / "pair_two_baseline_ncu.csv"
    optimized_csv_two = tmp_path / "pair_two_optimized_ncu.csv"
    baseline_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,40\n")
    optimized_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,60\n")

    comparison = profile_insights.compare_ncu_files(
        tmp_path,
        pair_key="pair_one",
        include_ncu_details=True,
    )
    assert comparison is not None
    assert comparison.get("metrics")
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "50"
    assert metrics["sm__throughput"]["optimized"] == "70"
    assert "baseline_sources" not in comparison


def test_profile_insights_normalizes_repeated_names():
    name = (
        "optimized_precisionfp8_pad_inner_matmul_optimized_"
        "optimized_precisionfp8_pad_inner_matmul.nsys-rep"
    )
    normalized = profile_insights._normalize_profile_name(name)
    assert normalized == "precisionfp8_pad_inner_matmul"
