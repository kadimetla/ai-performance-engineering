#!/usr/bin/env python3
"""
GPU Performance Lab Dashboard Server

A sleek web dashboard for viewing benchmark results, LLM analysis,
optimization insights, deep profiling comparisons, and live optimization streaming.

Features:
- Benchmark results visualization
- LLM-generated insights
- nsys/ncu profile comparison with recommendations
- Live optimization console with SSE streaming

Usage:
    python -m dashboard.api.server [--port 6970] [--data results.json]
"""

import http.server
import json
import os
import queue
import re
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import threading
import time
import uuid
import typer
from urllib.parse import urlparse, parse_qs


# Find the code root (repository root)
CODE_ROOT = Path(__file__).resolve().parents[2]

# Allow rapid restarts without hitting EADDRINUSE on 6970
socketserver.TCPServer.allow_reuse_address = True

# Add tools to path for imports
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    load_benchmark_data as load_benchmark_results,
)

# Shared non-HTTP core
try:
    from core.perf_core_base import PerformanceCoreBase
except Exception:
    PerformanceCoreBase = object  # fallback if core import fails during docs build
from core import profile_artifacts
from core.compile_analysis import load_compile_analysis
from core.costs import calculate_costs
from core.optimization_reports import compute_roi
from core.code_diff import find_code_pair, summarize_diff
from core import profile_insights
from core import optimization_stack
from core import whatif as whatif_core
from core import advanced_wrappers
from core.ncu_analysis import load_ncu_deepdive
from core.kernel_efficiency import score_kernels
from core.warmup_audit import run_warmup_audit
from core.report_export import generate_html_report

# Global optimization job store for SSE streaming
_optimization_jobs: Dict[str, Dict[str, Any]] = {}
_job_events: Dict[str, queue.Queue] = {}

# =============================================================================
# LLM IMPORTS - required for AI-powered analysis
# =============================================================================

# Import LLM engine for real AI-powered analysis
try:
    from core.llm_engine import PerformanceAnalysisEngine, LLMConfig
    LLM_ENGINE_AVAILABLE = True
except ImportError:
    LLM_ENGINE_AVAILABLE = False

# Import LLM advisor for comprehensive optimization recommendations
try:
    from core.optimization.parallelism_planner.llm_advisor import (
        LLMOptimizationAdvisor, SystemContext, OptimizationRequest, OptimizationGoal
    )
    LLM_ADVISOR_AVAILABLE = True
except ImportError:
    LLM_ADVISOR_AVAILABLE = False

# Import distributed training tools
try:
    from core.optimization.parallelism_planner.distributed_training import DistributedTrainingAnalyzer
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Import RL optimization tools
try:
    from core.optimization.parallelism_planner.rl_optimization import RLOptimizationAnalyzer
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Import vLLM optimization tools
try:
    from core.optimization.parallelism_planner.vllm_optimization import VLLMOptimizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Inference optimizer functionality is built into PerformanceCore
INFERENCE_OPTIMIZER_AVAILABLE = True


class PerformanceCore(PerformanceCoreBase, http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the dashboard and API endpoints."""
    
    LLM_SETUP_ERROR = {
        "available": False,
        "error": "LLM engine unavailable. Set OPENAI_API_KEY or ANTHROPIC_API_KEY (or configure VLLM_API_BASE/OLLAMA_HOST).",
        "setup_instructions": [
            "export OPENAI_API_KEY=sk-...",
            "or export ANTHROPIC_API_KEY=...",
            "or configure a local vLLM endpoint via VLLM_API_BASE",
        ],
    }
    _llm_engine: Optional[Any] = None
    _llm_advisor: Optional[Any] = None
    
    def __init__(self, *args, data_file: Optional[Path] = None, **kwargs):
        # Initialize shared core state without invoking HTTP handler
        try:
            PerformanceCoreBase.__init__(self, data_file=data_file)  # type: ignore[arg-type]
        except Exception:
            self.data_file = data_file
            self._analyzer = PerformanceAnalyzer(lambda: load_benchmark_results(self.data_file))
        # Now initialize HTTP handler
        http.server.SimpleHTTPRequestHandler.__init__(self, *args, **kwargs)

    @property
    def analyzer(self) -> PerformanceAnalyzer:
        if not hasattr(self, "_analyzer") or self._analyzer is None:
            data_path = getattr(self, "data_file", None)
            self._analyzer = PerformanceAnalyzer(lambda: load_benchmark_results(data_path))
        return self._analyzer

    def _parse_query(self) -> Dict[str, List[str]]:
        """Parse query parameters from the current request path."""
        parsed = urlparse(self.path)
        return parse_qs(parsed.query or "")
    
    def _get_llm_engine(self):
        """Instantiate and cache the LLM analysis engine."""
        if getattr(self, "_llm_engine", None):
            return self._llm_engine
        if not LLM_ENGINE_AVAILABLE:
            self._llm_init_error = "LLM engine import failed"
            return None
        try:
            self._llm_engine = PerformanceAnalysisEngine()
            return self._llm_engine
        except Exception as exc:  # pragma: no cover - defensive
            self._llm_init_error = str(exc)
            return None
    
    def _get_llm_advisor(self):
        """Instantiate and cache the LLM optimization advisor."""
        if getattr(self, "_llm_advisor", None):
            return self._llm_advisor
        if not LLM_ADVISOR_AVAILABLE:
            self._llm_advisor_init_error = "LLM advisor import failed"
            return None
        try:
            self._llm_advisor = LLMOptimizationAdvisor()
            return self._llm_advisor
        except Exception as exc:  # pragma: no cover - defensive
            self._llm_advisor_init_error = str(exc)
            return None
    
    def load_benchmark_data(self) -> dict:
        return load_benchmark_results(self.data_file)
    
    def do_GET(self):
        if self.path == '/api/data':
            self.send_json_response(self.load_benchmark_data())
        elif self.path == '/api/gpu':
            self.send_json_response(self.get_gpu_info())
        elif self.path == '/api/software':
            self.send_json_response(self.get_software_info())
        elif self.path == '/api/deps':
            self.send_json_response(self.get_dependency_health())
        elif self.path == '/api/deps/check-updates':
            self.send_json_response(self.check_dependency_updates())
        elif self.path == '/api/speedtest':
            self.send_json_response(self.run_speed_tests())
        elif self.path == '/api/gpu-bandwidth':
            self.send_json_response(self.run_gpu_bandwidth_test())
        elif self.path == '/api/network-test':
            self.send_json_response(self.run_network_tests())
        elif self.path == '/api/system-context':
            self.send_json_response(self.get_full_system_context())
        elif self.path == '/api/llm-analysis':
            self.send_json_response(self.load_llm_analysis())
        elif self.path == '/api/profiles':
            self.send_json_response(self.load_profile_data())
        elif self.path == '/api/available':
            self.send_json_response(self.get_available_benchmarks())
        elif self.path == '/api/scan-all':
            self.send_json_response(self.scan_all_chapters_and_labs())
        elif self.path == '/api/targets':
            self.send_json_response(self.list_benchmark_targets())
        # CSV Export endpoints
        elif self.path == '/api/export/csv':
            self.send_csv_response(self.export_benchmarks_csv())
        elif self.path == '/api/export/csv/detailed':
            self.send_csv_response(self.export_detailed_csv())
        # PDF Export endpoints
        elif self.path == '/api/export/pdf':
            self.export_pdf_report()
        elif self.path == '/api/export/html':
            self.export_html_report()
        # Profiler visualization endpoints
        elif self.path == '/api/profiler/flame':
            self.send_json_response(self.get_flame_graph_data())
        elif self.path == '/api/profiler/memory':
            self.send_json_response(self.get_memory_timeline())
        elif self.path == '/api/profiler/timeline':
            self.send_json_response(self.get_cpu_gpu_timeline())
        elif self.path == '/api/profiler/kernels':
            self.send_json_response(self.get_kernel_breakdown())
        elif self.path == '/api/profiler/hta':
            self.send_json_response(self.get_hta_analysis())
        elif self.path == '/api/profiler/compile':
            self.send_json_response(self.get_compile_analysis())
        elif self.path == '/api/profiler/roofline':
            self.send_json_response(self.get_roofline_data())
        # NEW: Deep profile comparison endpoints
        elif self.path == '/api/deep-profile/list':
            self.send_json_response(self.list_deep_profile_pairs())
        elif self.path.startswith('/api/deep-profile/compare/'):
            chapter = self.path.split('/api/deep-profile/compare/')[1]
            self.send_json_response(self.compare_profiles(chapter))
        elif self.path == '/api/deep-profile/recommendations':
            self.send_json_response(self.get_profile_recommendations())
        # NEW: Live optimization SSE streaming
        elif self.path.startswith('/api/optimize/stream/'):
            job_id = self.path.split('/api/optimize/stream/')[1]
            self.stream_optimization_events(job_id)
        elif self.path == '/api/optimize/jobs':
            self.send_json_response(self.list_optimization_jobs())
        # NEW: Multi-metric analysis endpoints
        elif self.path == '/api/analysis/pareto':
            self.send_json_response(self.get_pareto_frontier())
        elif self.path == '/api/analysis/tradeoffs':
            self.send_json_response(self.get_tradeoff_analysis())
        elif self.path == '/api/analysis/recommendations':
            self.send_json_response(self.get_constraint_recommendations())
        elif self.path == '/api/analysis/leaderboards':
            self.send_json_response(self.get_categorized_leaderboards())
        elif self.path.startswith('/api/analysis/whatif'):
            # Parse query params: ?vram=24&latency=50&throughput=1000
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_whatif_recommendations(params))
        elif self.path == '/api/analysis/stacking':
            self.send_json_response(self.get_optimization_stacking())
        elif self.path == '/api/stacking':
            # Legacy alias for static dashboard
            self.send_json_response(self.get_optimization_stacking())
        elif self.path == '/api/analysis/power':
            self.send_json_response(self.get_power_efficiency())
        elif self.path == '/api/analysis/scaling':
            self.send_json_response(self.get_scaling_analysis())
        elif self.path.startswith('/api/report'):
            params = self._parse_query()
            self.send_json_response(self.generate_report_from_query(params))
        elif self.path.startswith('/api/compare-runs'):
            params = self._parse_query()
            self.send_json_response(self.compare_runs_from_query(params))
        elif self.path.startswith('/api/export/generic'):
            params = self._parse_query()
            self.send_json_response(self.export_results_from_query(params))
        elif self.path.startswith('/api/launch-plan'):
            params = self._parse_query()
            self.send_json_response(self.generate_launch_plan_from_query(params))
        elif self.path.startswith('/api/roofline'):
            params = self._parse_query()
            size_mb = int((params.get("size_mb") or ["32"])[0])
            strides = [int(s) for s in (params.get("stride") or [])] if params.get("stride") else None
            self.send_json_response(self.roofline_sweep(size_mb=size_mb, strides=strides))
        # Microbench endpoints
        elif self.path.startswith('/api/export/csv'):
            detailed = False
            try:
                detailed = bool(int(self._parse_query().get('detailed', [0])[0]))
            except Exception:
                pass
            if detailed:
                self.send_json_response({"csv": self.export_detailed_csv(), "detailed": True})
            else:
                self.send_json_response({"csv": self.export_benchmarks_csv(), "detailed": False})
        elif self.path == '/api/export/html':
            self.send_json_response({"html": self.export_html_report()})
        elif self.path == '/api/export/pdf':
            self.export_pdf_report()
        elif self.path.startswith('/api/microbench/disk'):
            from core.diagnostics import microbench
            params = self._parse_query()
            res = microbench.disk_io_test(
                file_size_mb=int(params.get('file_size_mb', [256])[0]),
                block_size_kb=int(params.get('block_size_kb', [1024])[0]),
                tmp_dir=params.get('tmp_dir', [None])[0],
            )
            self.send_json_response(res)
        elif self.path.startswith('/api/microbench/pcie'):
            from core.diagnostics import microbench
            params = self._parse_query()
            res = microbench.pcie_bandwidth_test(
                size_mb=int(params.get('size_mb', [256])[0]),
                iters=int(params.get('iters', [10])[0]),
            )
            self.send_json_response(res)
        elif self.path.startswith('/api/microbench/mem'):
            from core.diagnostics import microbench
            params = self._parse_query()
            res = microbench.mem_hierarchy_test(
                size_mb=int(params.get('size_mb', [256])[0]),
                stride=int(params.get('stride', [128])[0]),
            )
            self.send_json_response(res)
        elif self.path.startswith('/api/microbench/roofline'):
            params = self._parse_query()
            size_mb = int(params.get('size_mb', [32])[0])
            strides = [int(s) for s in params.get('stride', [])] if params.get('stride') else None
            self.send_json_response(self.roofline_sweep(size_mb=size_mb, strides=strides))
        elif self.path.startswith('/api/microbench/tensor'):
            from core.diagnostics import microbench
            params = self._parse_query()
            res = microbench.tensor_core_bench(
                size=int(params.get('size', [4096])[0]),
                precision=params.get('precision', ['fp16'])[0],
            )
            self.send_json_response(res)
        elif self.path.startswith('/api/microbench/sfu'):
            from core.diagnostics import microbench
            params = self._parse_query()
            res = microbench.sfu_bench(
                size=int(params.get('elements', [64 * 1024 * 1024])[0]),
            )
            self.send_json_response(res)
        elif self.path.startswith('/api/microbench/loopback'):
            from core.diagnostics import microbench
            params = self._parse_query()
            res = microbench.network_loopback_test(
                size_mb=int(params.get('size_mb', [64])[0]),
                port=int(params.get('port', [50007])[0]),
            )
            self.send_json_response(res)
        elif self.path == '/api/nsight/availability':
            from core.profiling.nsight_automation import NsightAutomation
            automation = NsightAutomation(Path("artifacts/mcp-profiles"))
            self.send_json_response({
                "nsys_available": automation.nsys_available,
                "ncu_available": automation.ncu_available,
                "output_dir": str(automation.output_dir),
            })
        elif self.path == '/api/nsight/compare/nsys':
            profiles_dir = self._parse_query().get('dir', [''])[0]
            from core import profile_insights
            result = profile_insights.compare_nsys_files(Path(profiles_dir))
            self.send_json_response(result or {"error": "No comparable nsys files found"})
        elif self.path == '/api/nsight/compare/ncu':
            profiles_dir = self._parse_query().get('dir', [''])[0]
            from core import profile_insights
            result = profile_insights.compare_ncu_files(Path(profiles_dir))
            self.send_json_response(result or {"error": "No comparable ncu files found"})
        # =====================================================================
        # ADVANCED SYSTEM ANALYSIS (NEW!)
        # =====================================================================
        elif self.path == '/api/analysis/cpu-memory':
            self.send_json_response(self.get_cpu_memory_analysis())
        elif self.path == '/api/analysis/system-params':
            self.send_json_response(self.get_system_parameters())
        elif self.path == '/api/analysis/container-limits':
            self.send_json_response(self.get_container_limits())
        elif self.path.startswith('/api/analysis/warp-divergence'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            code = params.get('code', [''])[0]
            self.send_json_response(self.analyze_warp_divergence(code))
        elif self.path.startswith('/api/analysis/bank-conflicts'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            stride = int(params.get('stride', ['1'])[0])
            element_size = int(params.get('element_size', ['4'])[0])
            self.send_json_response(self.analyze_bank_conflicts(stride, element_size))
        elif self.path.startswith('/api/analysis/memory-access'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            stride = int(params.get('stride', ['1'])[0])
            element_size = int(params.get('element_size', ['4'])[0])
            self.send_json_response(self.analyze_memory_access(stride, element_size))
        elif self.path.startswith('/api/analysis/auto-tune'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            kernel = params.get('kernel', ['matmul'])[0]
            max_configs = int(params.get('max_configs', ['50'])[0])
            self.send_json_response(self.run_auto_tuning(kernel, max_configs))
        elif self.path == '/api/analysis/full-system':
            self.send_json_response(self.get_full_system_analysis())
        # Hardware scaling prediction
        elif self.path.startswith('/api/analysis/predict-scaling'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            from_gpu = params.get('from', ['H100'])[0]
            to_gpu = params.get('to', ['B200'])[0]
            workload = params.get('workload', ['inference'])[0]
            self.send_json_response(self.predict_hardware_scaling(from_gpu, to_gpu, workload))
        # Energy efficiency
        elif self.path.startswith('/api/analysis/energy'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            gpu = params.get('gpu', ['H100'])[0]
            power_limit = params.get('power_limit', [None])[0]
            power_limit = int(power_limit) if power_limit else None
            self.send_json_response(self.analyze_energy_efficiency(gpu, power_limit))
        # Multi-GPU scaling
        elif self.path.startswith('/api/analysis/multi-gpu-scaling'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            gpus = int(params.get('gpus', ['8'])[0])
            nvlink = params.get('nvlink', ['true'])[0].lower() == 'true'
            workload = params.get('workload', ['training'])[0]
            self.send_json_response(self.estimate_multi_gpu_scaling(gpus, nvlink, workload))
        # Advanced optimization analysis
        elif self.path == '/api/analysis/optimizations':
            self.send_json_response(self.get_all_optimizations())
        elif self.path == '/api/analysis/playbooks':
            self.send_json_response(self.get_optimization_playbooks())
        elif self.path.startswith('/api/analysis/compound'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            opts = params.get('opts', [''])[0].split(',')
            self.send_json_response(self.calculate_compound_optimization(opts))
        elif self.path.startswith('/api/analysis/optimal-stack'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            target = float(params.get('target', ['10'])[0])
            difficulty = params.get('difficulty', ['medium'])[0]
            self.send_json_response(self.get_optimal_optimization_stack(target, difficulty))
        elif self.path.startswith('/api/analysis/occupancy'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            threads = int(params.get('threads', ['256'])[0])
            shared = int(params.get('shared', ['0'])[0])
            registers = int(params.get('registers', ['32'])[0])
            self.send_json_response(self.calculate_occupancy(threads, shared, registers))
        # Warmup Audit endpoint
        elif self.path.startswith('/api/audit/warmup'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            check_recommended = params.get('check_recommended', ['false'])[0].lower() == 'true'
            self.send_json_response(self.run_warmup_audit(check_recommended))
        # =====================================================================
        # LLM-POWERED DYNAMIC ANALYSIS (NOT HARD-CODED!)
        # =====================================================================
        elif self.path.startswith('/api/llm/status'):
            probe_flag = False
            try:
                params = self._parse_query()
                probe_flag = params.get("probe", ["0"])[0].lower() in {"1", "true", "yes"}
            except Exception:
                probe_flag = False
            self.send_json_response(self.get_llm_status(probe=probe_flag))
        elif self.path == '/api/llm/analyze-bottlenecks':
            self.send_json_response(self.llm_analyze_bottlenecks())
        elif self.path.startswith('/api/llm/distributed'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.llm_distributed_recommendations({
                "num_nodes": int(params.get("nodes", ["1"])[0]),
                "gpus_per_node": int(params.get("gpus", ["8"])[0]),
                "model_params_b": float(params.get("params", ["70"])[0]),
                "interconnect": params.get("interconnect", ["infiniband"])[0],
            }))
        elif self.path.startswith('/api/llm/inference'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.llm_inference_recommendations({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "target_latency_ms": float(params.get("latency", ["0"])[0]) or None,
                "target_throughput": float(params.get("throughput", ["0"])[0]) or None,
                "max_batch_size": int(params.get("batch", ["32"])[0]),
                "max_sequence_length": int(params.get("seq", ["4096"])[0]),
            }))
        elif self.path.startswith('/api/llm/rlhf'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.llm_rlhf_recommendations({
                "policy_size_b": float(params.get("policy", ["7"])[0]),
                "reward_size_b": float(params.get("reward", ["7"])[0]),
                "num_gpus": int(params.get("gpus", ["8"])[0]),
            }))
        elif self.path.startswith('/api/llm/custom-query'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            query = unquote(params.get("q", [""])[0])
            self.send_json_response(self.llm_custom_query(query))
        elif self.path.startswith('/api/analysis/cost'):
            # Parse query params: ?gpu=H100&rate=4.00
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            gpu = params.get('gpu', [None])[0]
            rate = params.get('rate', [None])[0]
            custom_rate = float(rate) if rate else None
            self.send_json_response(self.get_cost_analysis(gpu=gpu, custom_rate=custom_rate))
        # NEW: Hardware capabilities for optimization suggestions
        elif self.path == '/api/hardware-capabilities':
            self.send_json_response(self.get_hardware_capabilities())
        elif self.path == '/api/profiler/bottlenecks':
            self.send_json_response(self.detect_bottlenecks())
        elif self.path == '/api/analysis/bottlenecks':
            self.send_json_response(self.get_bottleneck_summary())
        elif self.path == '/api/profiler/optimization-score':
            self.send_json_response(self.calculate_optimization_score())
        # NEW: Book-based technique explanations
        elif self.path.startswith('/api/explain/'):
            from urllib.parse import unquote
            params = self.path.split('/api/explain/')[1]
            # Parse: technique/chapter (e.g., unroll8/ch8)
            parts = params.split('/')
            technique = unquote(parts[0]) if parts else ''
            chapter = unquote(parts[1]) if len(parts) > 1 else None
            self.send_json_response(self.get_technique_explanation(technique, chapter))
        # NEW: LLM-powered deep explanation with full context
        elif self.path.startswith('/api/explain-llm/'):
            from urllib.parse import unquote
            params = self.path.split('/api/explain-llm/')[1]
            parts = params.split('/')
            technique = unquote(parts[0]) if parts else ''
            chapter = unquote(parts[1]) if len(parts) > 1 else None
            benchmark = unquote(parts[2]) if len(parts) > 2 else None
            self.send_json_response(self.get_llm_explanation(technique, chapter, benchmark))
        # =====================================================================
        # NEW AWESOME FEATURES
        # =====================================================================
        # Interactive Roofline Model
        elif self.path == '/api/roofline/interactive':
            self.send_json_response(self.get_interactive_roofline())
        # Cost Calculator & TCO
        elif self.path == '/api/cost/calculator':
            self.send_json_response(self.get_cost_calculator())
        elif self.path == '/api/cost/roi':
            self.send_json_response(self.get_optimization_roi())
        # Code Diff Viewer
        elif self.path.startswith('/api/diff/'):
            chapter = self.path.split('/api/diff/')[1]
            self.send_json_response(self.get_code_diff(chapter))
        # Kernel Efficiency Dashboard
        elif self.path == '/api/efficiency/kernels':
            self.send_json_response(self.get_kernel_efficiency())
        # What-If Simulator
        elif self.path == '/api/whatif/simulate':
            self.send_json_response(self.get_whatif_scenarios())
        # NCU Deep Dive
        elif self.path == '/api/ncu/deepdive':
            self.send_json_response(self.get_ncu_deepdive())
        # =====================================================================
        # OPTIMIZATION INTELLIGENCE ENGINE (LLM-Powered)
        # =====================================================================
        elif self.path.startswith('/api/intelligence/recommend'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_intelligent_recommendation(params))
        elif self.path.startswith('/api/intelligence/distributed'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_distributed_training_plan(params))
        elif self.path.startswith('/api/intelligence/vllm'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_vllm_config(params))
        elif self.path.startswith('/api/intelligence/rl'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_rl_config(params))
        elif self.path == '/api/intelligence/techniques':
            self.send_json_response(self.get_optimization_techniques())
        # Shareable Report
        elif self.path == '/api/report/generate':
            self.send_html_report()
        # =====================================================================
        # NEW: Multi-GPU, History, Batch Optimizer, Themes
        # =====================================================================
        # Multi-GPU / NVLink Topology
        elif self.path == '/api/gpu/topology':
            self.send_json_response(self.get_gpu_topology())
        elif self.path == '/api/gpu/nvlink':
            self.send_json_response(self.get_nvlink_status())
        # Historical Performance Tracking
        elif self.path == '/api/history':
            self.send_json_response(self.get_history_summary())
        elif self.path == '/api/history/runs':
            self.send_json_response(self.get_history_runs())
        elif self.path == '/api/history/trends':
            self.send_json_response(self.get_performance_trends())
        # Batch Size Optimizer
        elif self.path == '/api/batch/optimize':
            self.send_json_response(self.get_batch_size_recommendations())
        # =====================================================================
        # PARALLELISM STRATEGY ADVISOR
        # =====================================================================
        elif self.path == '/api/parallelism/topology':
            self.send_json_response(self.get_parallelism_topology())
        elif self.path == '/api/parallelism/presets':
            self.send_json_response(self.get_parallelism_presets())
        elif self.path.startswith('/api/parallelism/recommend'):
            # Parse query params: ?model=llama-3.1-70b&batch=8&seq=4096&goal=throughput&training=false
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_parallelism_recommendations({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "batch_size": int(params.get("batch", ["1"])[0]),
                "seq_length": int(params.get("seq", ["2048"])[0]),
                "goal": params.get("goal", ["throughput"])[0],
                "is_training": params.get("training", ["false"])[0].lower() == "true",
            }))
        elif self.path.startswith('/api/parallelism/analyze-model'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model_id = unquote(params.get("model", [""])[0])
            self.send_json_response(self.analyze_parallelism_model(model_id))
        elif self.path == '/api/parallelism/clusters':
            self.send_json_response(self.get_cluster_presets())
        # NEW: Advanced parallelism features
        elif self.path.startswith('/api/parallelism/sharding'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_sharding_recommendations({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "dp_size": int(params.get("dp", ["8"])[0]),
                "gpu_memory_gb": float(params.get("memory", ["80"])[0]),
                "batch_size": int(params.get("batch", ["1"])[0]),
                "seq_length": int(params.get("seq", ["2048"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/pareto'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_pareto_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "gpu_cost": float(params.get("cost", ["4.0"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/launch'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_launch_commands({
                "num_nodes": int(params.get("nodes", ["1"])[0]),
                "gpus_per_node": int(params.get("gpus", ["8"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "sharding": params.get("sharding", ["none"])[0],
                "script": unquote(params.get("script", ["train.py"])[0]),
            }))
        elif self.path == '/api/parallelism/calibration':
            self.send_json_response(self.get_calibration_data())
        elif self.path.startswith('/api/parallelism/sharding'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_sharding_recommendations({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "world_size": int(params.get("world_size", ["8"])[0]),
                "gpu_memory_gb": float(params.get("memory", ["80"])[0]),
                "batch_size": int(params.get("batch", ["1"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/launch'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_launch_commands({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "framework": params.get("framework", ["torchrun"])[0],
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "num_nodes": int(params.get("nodes", ["1"])[0]),
            }))
        elif self.path == '/api/parallelism/pareto':
            self.send_json_response(self.get_pareto_analysis())
        elif self.path.startswith('/api/parallelism/estimate'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_training_estimate({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "tokens": int(params.get("tokens", ["1000000000000"])[0]),
                "throughput": float(params.get("throughput", ["100000"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "gpu_cost": float(params.get("cost", ["4.0"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/compare'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            models = params.get("models", ["llama-3.1-8b,llama-3.1-70b"])[0].split(",")
            self.send_json_response(self.get_model_comparison({
                "models": [unquote(m.strip()) for m in models],
            }))
        elif self.path.startswith('/api/parallelism/slurm'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.generate_slurm_script({
                "job_name": params.get("name", ["train"])[0],
                "nodes": int(params.get("nodes", ["1"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "time": int(params.get("time", ["24"])[0]),
                "script": unquote(params.get("script", ["train.py"])[0]),
            }))
        # NEW: Advanced parallelism validation, optimizations, and profiles
        elif self.path.startswith('/api/parallelism/validate'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.validate_parallelism_config({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "cp": int(params.get("cp", ["1"])[0]),
                "ep": int(params.get("ep", ["1"])[0]),
                "batch_size": int(params.get("batch", ["1"])[0]),
                "seq_length": int(params.get("seq", ["2048"])[0]),
                "training": params.get("training", ["false"])[0].lower() == "true",
            }))
        elif self.path.startswith('/api/parallelism/optimize'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_advanced_optimizations({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "goal": params.get("goal", ["balanced"])[0],
                "batch_size": int(params.get("batch", ["1"])[0]),
                "seq_length": int(params.get("seq", ["4096"])[0]),
            }))
        elif self.path == '/api/parallelism/profiles':
            self.send_json_response(self.list_performance_profiles())
        elif self.path.startswith('/api/parallelism/profile'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_performance_profile({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "workload": params.get("workload", ["pretraining"])[0],
                "batch_size": int(params.get("batch", ["32"])[0]),
                "seq_length": int(params.get("seq", ["4096"])[0]),
                "lora": params.get("lora", ["false"])[0].lower() == "true",
                "inference_mode": params.get("inference_mode", ["batch"])[0],
            }))
        # NEW: Advanced analysis endpoints
        elif self.path.startswith('/api/parallelism/bottleneck'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_bottleneck_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "batch_size": int(params.get("batch", ["8"])[0]),
                "seq_length": int(params.get("seq", ["4096"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/scaling'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_scaling_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "throughput": float(params.get("throughput", ["100000"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "max_gpus": int(params.get("max_gpus", ["512"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/whatif'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_whatif_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "batch_size": int(params.get("batch", ["8"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/batch-size'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_batch_size_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "seq_length": int(params.get("seq", ["4096"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "target_batch": int(params.get("target", ["1024"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/auto-tune'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_auto_tune({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "goal": params.get("goal", ["throughput"])[0],
                "target_batch": int(params.get("target", ["1024"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/inference-opt'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_inference_optimization({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "goal": params.get("goal", ["throughput"])[0],
            }))
        # NEW: Distributed Training & Advanced Features
        elif self.path.startswith('/api/distributed/nccl'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_nccl_tuning({
                "nodes": int(params.get("nodes", ["1"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "model_size": float(params.get("model_size", ["70"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "diagnose": params.get("diagnose", ["false"])[0] == "true",
            }))
        elif self.path.startswith('/api/distributed/rlhf'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_rlhf_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "algorithm": params.get("algorithm", ["ppo"])[0],
                "batch_size": int(params.get("batch", ["4"])[0]),
                "seq_length": int(params.get("seq", ["2048"])[0]),
                "memory": float(params.get("memory", ["80"])[0]),
                "compare": params.get("compare", ["false"])[0] == "true",
            }))
        elif self.path.startswith('/api/distributed/moe'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_moe_config({
                "model": unquote(params.get("model", ["mixtral-8x7b"])[0]),
                "num_experts": int(params.get("experts", ["8"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "memory": float(params.get("memory", ["80"])[0]),
                "batch_size": int(params.get("batch", ["8"])[0]),
            }))
        elif self.path.startswith('/api/distributed/long-context'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_long_context_config({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "seq_length": int(params.get("seq", ["128000"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "memory": float(params.get("memory", ["80"])[0]),
                "method": params.get("method", ["auto"])[0],
            }))
        elif self.path.startswith('/api/distributed/vllm'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_vllm_config({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "gpus": int(params.get("gpus", ["1"])[0]),
                "memory": float(params.get("memory", ["80"])[0]),
                "target": params.get("target", ["throughput"])[0],
                "max_seq_length": int(params.get("seq", ["8192"])[0]),
                "quantization": params.get("quant", [None])[0],
                "compare_engines": params.get("compare", ["false"])[0] == "true",
            }))
        elif self.path.startswith('/api/distributed/comm-overlap'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_comm_overlap_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "batch_size": int(params.get("batch", ["8"])[0]),
                "seq_length": int(params.get("seq", ["4096"])[0]),
            }))
        # NEW: LLM-Powered Optimization Advisor
        elif self.path.startswith('/api/llm/advisor'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_llm_optimization_advice({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "goal": params.get("goal", ["throughput"])[0],
                "gpus": int(params.get("gpus", ["8"])[0]),
                "is_training": params.get("training", ["true"])[0].lower() == "true",
                "provider": params.get("provider", ["anthropic"])[0],
            }))
        # NEW: Troubleshooting and diagnostics
        elif self.path == '/api/parallelism/troubleshoot/topics':
            self.send_json_response(self.get_troubleshooting_topics())
        elif self.path.startswith('/api/parallelism/troubleshoot/diagnose'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.diagnose_training_error({
                "error": unquote(params.get("error", [""])[0]),
            }))
        elif self.path.startswith('/api/parallelism/troubleshoot/nccl'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_nccl_recommendations({
                "interconnect": params.get("interconnect", ["nvlink"])[0],
            }))
        elif self.path.startswith('/api/parallelism/memory'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_memory_analysis({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "batch_size": int(params.get("batch", ["8"])[0]),
                "seq_length": int(params.get("seq", ["4096"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
            }))
        elif self.path.startswith('/api/parallelism/export'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.export_config({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "nodes": int(params.get("nodes", ["1"])[0]),
                "gpus": int(params.get("gpus", ["8"])[0]),
                "tp": int(params.get("tp", ["1"])[0]),
                "pp": int(params.get("pp", ["1"])[0]),
                "dp": int(params.get("dp", ["8"])[0]),
                "batch_size": int(params.get("batch", ["256"])[0]),
                "zero_stage": int(params.get("zero", ["2"])[0]),
            }))
        # NEW: RL/RLHF optimization
        elif self.path.startswith('/api/parallelism/rl'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_rl_optimization({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "algorithm": params.get("algorithm", ["ppo"])[0],
                "gpus": int(params.get("gpus", ["8"])[0]),
                "use_peft": params.get("peft", ["true"])[0].lower() == "true",
            }))
        # NEW: vLLM optimization
        elif self.path.startswith('/api/parallelism/vllm'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_vllm_optimization({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "goal": params.get("goal", ["throughput"])[0],
                "gpus": int(params.get("gpus", ["1"])[0]),
                "max_seq_len": int(params.get("seq", ["8192"])[0]),
            }))
        # NEW: Large-scale cluster optimization
        elif self.path.startswith('/api/parallelism/large-scale'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_large_scale_optimization({
                "model": unquote(params.get("model", ["llama-3.1-70b"])[0]),
                "nodes": int(params.get("nodes", ["8"])[0]),
                "gpus_per_node": int(params.get("gpus", ["8"])[0]),
                "network": params.get("network", ["infiniband"])[0],
                "batch_size": int(params.get("batch", ["1024"])[0]),
            }))
        # =====================================================================
        # CLUSTER RESILIENCE (Fault Tolerance, Spot Instances, Elastic Scaling)
        # =====================================================================
        elif self.path.startswith('/api/cluster/fault-tolerance'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_fault_tolerance_config({
                "model_params_b": float(params.get("params", ["70"])[0]),
                "num_nodes": int(params.get("nodes", ["1"])[0]),
                "gpus_per_node": int(params.get("gpus", ["8"])[0]),
                "training_hours": int(params.get("hours", ["24"])[0]),
                "use_spot": params.get("spot", ["false"])[0].lower() == "true",
                "cloud_provider": params.get("cloud", ["aws"])[0],
            }))
        elif self.path.startswith('/api/cluster/spot-config'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_spot_instance_config({
                "model_params_b": float(params.get("params", ["70"])[0]),
                "cloud_provider": params.get("cloud", ["aws"])[0],
                "budget_sensitive": params.get("budget", ["true"])[0].lower() == "true",
            }))
        elif self.path.startswith('/api/cluster/elastic-scaling'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.get_elastic_scaling_config({
                "model_params_b": float(params.get("params", ["70"])[0]),
                "initial_nodes": int(params.get("nodes", ["4"])[0]),
                "traffic_pattern": params.get("traffic", ["variable"])[0],
            }))
        elif self.path.startswith('/api/cluster/diagnose'):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_json_response(self.diagnose_cluster_error({
                "error": unquote(params.get("error", [""])[0]),
            }))
        # HuggingFace Model Integration
        elif self.path == '/api/hf/trending':
            self.send_json_response(self.get_hf_trending_models())
        elif self.path.startswith('/api/hf/search'):
            # Parse query params: ?q=llama
            query_string = self.path.split('?')[1] if '?' in self.path else ''
            params = dict(p.split('=') for p in query_string.split('&') if '=' in p)
            from urllib.parse import unquote
            search_query = unquote(params.get('q', ''))
            self.send_json_response(self.search_hf_models(search_query))
        elif self.path.startswith('/api/hf/model/'):
            # Get specific model info: /api/hf/model/meta-llama/Llama-2-7b
            model_id = self.path.split('/api/hf/model/')[1]
            from urllib.parse import unquote
            self.send_json_response(self.get_hf_model_info(unquote(model_id)))
        # Advanced batch optimizer features
        elif self.path == '/api/batch/models-that-fit':
            self.send_json_response(self.get_models_that_fit())
        elif self.path.startswith('/api/batch/throughput'):
            query_string = self.path.split('?')[1] if '?' in self.path else ''
            params = dict(p.split('=') for p in query_string.split('&') if '=' in p)
            self.send_json_response(self.get_throughput_estimate({
                "params": float(params.get("params", 7e9)),
                "precision": params.get("precision", "fp16"),
            }))
        # Theme preferences (stored in memory for session)
        elif self.path == '/api/themes':
            self.send_json_response(self.get_available_themes())
        # Webhook configuration (persistent)
        elif self.path == '/api/webhooks':
            self.send_json_response(self.get_webhooks())
        # Code diff for baseline vs optimized
        elif self.path.startswith('/api/code-diff/'):
            parts = self.path.split('/api/code-diff/')[1].split('/')
            if len(parts) >= 2:
                chapter, name = parts[0], '/'.join(parts[1:])
                self.send_json_response(self.get_code_diff(chapter, name))
            else:
                self.send_json_response({"error": "Invalid code-diff path"})
        # GPU control endpoints
        elif self.path == '/api/gpu/control':
            self.send_json_response(self.get_gpu_control_state())
        elif self.path == '/api/gpu/topology':
            self.send_json_response(self.get_gpu_topology())
        elif self.path == '/api/cuda/environment':
            self.send_json_response(self.get_cuda_environment())
        # =====================================================================
        # LLM-POWERED INTELLIGENT ANALYSIS (NEW!)
        # =====================================================================
        elif self.path.startswith('/api/ai/analyze'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            analysis_type = params.get('type', ['bottleneck'])[0]
            self.send_json_response(self.run_ai_analysis(analysis_type))
        elif self.path == '/api/ai/suggest':
            self.send_json_response(self.get_ai_suggestions())
        elif self.path == '/api/ai/context':
            self.send_json_response(self.get_ai_context())
        # =====================================================================
        # PARALLELISM PLANNER & DISTRIBUTED TRAINING (NEW!)
        # =====================================================================
        elif self.path.startswith('/api/parallelism/nccl'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            nodes = int(params.get('nodes', ['1'])[0])
            gpus = int(params.get('gpus', ['8'])[0])
            diagnose = params.get('diagnose', ['false'])[0].lower() == 'true'
            self.send_json_response(self.get_nccl_recommendations(nodes, gpus, diagnose))
        elif self.path.startswith('/api/parallelism/rlhf'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model = params.get('model', ['llama-3.1-70b'])[0]
            algorithm = params.get('algorithm', ['ppo'])[0]
            compare = params.get('compare', ['false'])[0].lower() == 'true'
            self.send_json_response(self.get_rlhf_optimization(model, algorithm, compare))
        elif self.path.startswith('/api/parallelism/moe'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model = params.get('model', ['mixtral-8x7b'])[0]
            self.send_json_response(self.get_moe_optimization(model))
        elif self.path.startswith('/api/parallelism/long-context'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model = params.get('model', ['llama-3.1-70b'])[0]
            seq_length = int(params.get('seq_length', ['128000'])[0])
            self.send_json_response(self.get_long_context_optimization(model, seq_length))
        elif self.path.startswith('/api/parallelism/vllm'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model = params.get('model', ['llama-3.1-70b'])[0]
            target = params.get('target', ['throughput'])[0]
            compare = params.get('compare', ['false'])[0].lower() == 'true'
            self.send_json_response(self.get_vllm_config(model, target, compare))
        elif self.path.startswith('/api/parallelism/comm-overlap'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model = params.get('model', ['llama-3.1-70b'])[0]
            self.send_json_response(self.get_comm_overlap_analysis(model))
        elif self.path.startswith('/api/parallelism/slurm'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            model = params.get('model', ['llama-3.1-70b'])[0]
            nodes = int(params.get('nodes', ['1'])[0])
            gpus = int(params.get('gpus', ['8'])[0])
            framework = params.get('framework', ['pytorch'])[0]
            self.send_json_response(self.generate_slurm_script(model, nodes, gpus, framework))
        elif self.path.startswith('/api/'):
            self.send_json_response({"error": "Unknown API endpoint"})
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests for starting optimizations."""
        if self.path == '/api/optimize/start':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.start_optimization_job(params)
            self.send_json_response(result)
        elif self.path == '/api/optimize/stop':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.stop_optimization_job(params.get('job_id'))
            self.send_json_response(result)
        # NEW: LLM-powered kernel analysis
        elif self.path == '/api/profiler/analyze-kernel':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.analyze_kernel_with_llm(params)
            self.send_json_response(result)
        # NEW: Generate optimization patch from analysis
        elif self.path == '/api/profiler/generate-patch':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.generate_optimization_patch(params)
            self.send_json_response(result)
        # NEW: AI Chat for profiling questions
        elif self.path == '/api/profiler/ask':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.ask_profiler_ai(params)
            self.send_json_response(result)
        # NEW: Parallelism strategy recommendation (POST for complex queries)
        elif self.path == '/api/parallelism/recommend':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_parallelism_recommendations(params)
            self.send_json_response(result)
        # NEW: Sharding recommendations (POST)
        elif self.path == '/api/parallelism/sharding':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_sharding_recommendations(params)
            self.send_json_response(result)
        # NEW: Pareto analysis (POST)
        elif self.path == '/api/parallelism/pareto':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_pareto_analysis(params)
            self.send_json_response(result)
        # NEW: Launch commands (POST)
        elif self.path == '/api/parallelism/launch':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_launch_commands(params)
            self.send_json_response(result)
        # NEW: AI Free-Form Query
        elif self.path == '/api/ai/query':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.run_ai_query(params.get('query', ''))
            self.send_json_response(result)
        # NEW: Webhook configuration
        elif self.path == '/api/webhook/test':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.test_webhook(params)
            self.send_json_response(result)
        elif self.path == '/api/webhook/send':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.send_webhook_notification(params)
            self.send_json_response(result)
        elif self.path == '/api/webhooks/save':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.save_webhooks(params)
            self.send_json_response(result)
        # Quick benchmark runner
        elif self.path == '/api/benchmark/run':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.run_benchmark(params)
            self.send_json_response(result)
        elif self.path == '/api/run-benchmark':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            # Legacy alias used by the static dashboard
            result = self.run_benchmark(params)
            self.send_json_response(result)
        # GPU control POST endpoints
        elif self.path == '/api/gpu/power-limit':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.set_gpu_power_limit(params)
            self.send_json_response(result)
        elif self.path == '/api/gpu/clock-pin':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.set_gpu_clock_pin(params)
            self.send_json_response(result)
        elif self.path == '/api/gpu/persistence':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.set_gpu_persistence(params)
            self.send_json_response(result)
        elif self.path == '/api/gpu/preset':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.apply_gpu_preset(params)
            self.send_json_response(result)
        # Custom batch size calculation
        elif self.path == '/api/batch/calculate':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.calculate_batch_for_model(params)
            self.send_json_response(result)
        # Quantization comparison
        elif self.path == '/api/batch/quantization':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_quantization_comparison(params)
            self.send_json_response(result)
        # Multi-GPU scaling
        elif self.path == '/api/batch/multi-gpu':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_multi_gpu_scaling(params)
            self.send_json_response(result)
        # Cloud cost estimation
        elif self.path == '/api/batch/cloud-cost':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_cloud_cost_estimate(params)
            self.send_json_response(result)
        # Deploy config generator
        elif self.path == '/api/batch/deploy-config':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.generate_deploy_config(params)
            self.send_json_response(result)
        # Fine-tuning estimation
        elif self.path == '/api/batch/finetune':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_finetuning_estimate(params)
            self.send_json_response(result)
        # LLM-powered advisor (dynamic recommendations, not hardcoded)
        elif self.path == '/api/batch/llm-advisor':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.get_llm_optimization_advice(params)
            self.send_json_response(result)
        # Compound optimization analysis
        elif self.path == '/api/batch/compound':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.calculate_compound_optimizations(params)
            self.send_json_response(result)
        # =====================================================================
        # UNIFIED API - Route to unified API handler for all optimization
        # =====================================================================
        elif self.path.startswith('/api/unified/') or self.path in [
            '/api/optimize/suggest',
            '/api/optimize/search', 
            '/api/optimize/distributed',
            '/api/optimize/rlhf',
            '/api/optimize/vllm',
            '/api/optimize/compound',
            '/api/ask',
            '/api/validate',
        ]:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(body) if body else {}
            except json.JSONDecodeError:
                params = {}
            result = self.handle_unified_api(self.path, params)
            self.send_json_response(result)
        else:
            self.send_json_response({"error": "Unknown POST endpoint"})
    
    def send_json_response(self, data: dict):
        """Send a JSON response."""
        response = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def send_csv_response(self, csv_data: str, filename: str = "benchmark_results.csv"):
        """Send a CSV response for download."""
        response = csv_data.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv')
        self.send_header('Content-Length', len(response))
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def export_benchmarks_csv(self) -> str:
        """Export benchmark results to CSV format."""
        return super().export_benchmarks_csv()
    
    def export_detailed_csv(self) -> str:
        """Export detailed benchmark results including all metrics."""
        return super().export_detailed_csv()
    
    def get_flame_graph_data(self) -> dict:
        """Get flame graph data from profile traces."""
        return super().get_flame_graph_data()
    
    def get_memory_timeline(self) -> dict:
        """Get memory usage timeline data."""
        return super().get_memory_timeline()
    
    def get_cpu_gpu_timeline(self) -> dict:
        """Get CPU/GPU parallel timeline data."""
        return super().get_cpu_gpu_timeline()
    
    def get_kernel_breakdown(self) -> dict:
        """Get detailed kernel timing breakdown."""
        return super().get_kernel_breakdown()
    
    def get_hta_analysis(self) -> dict:
        """Get HTA (Holistic Trace Analysis) results."""
        return super().get_hta_analysis()
    
    def get_compile_analysis(self) -> dict:
        """Get torch.compile analysis results from REAL benchmark data."""
        return super().get_compile_analysis()
    
    def get_roofline_data(self) -> dict:
        """Get REAL roofline data computed from benchmark throughput metrics."""
        return super().get_roofline_data()
    
    def get_available_benchmarks(self) -> dict:
        """Scan all chapters and labs for available benchmarks."""
        return super().get_available_benchmarks()
    
    def _scan_directory(self, directory: Path, dir_type: str) -> dict:
        """Scan a directory for baseline/optimized file pairs."""
        return super()._scan_directory(directory, dir_type)
    
    def scan_all_chapters_and_labs(self) -> dict:
        """Comprehensive scan of all chapters and labs with detailed info."""
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_results": [],
            "summary": {
                "total_directories": 0,
                "total_benchmarks": 0,
                "with_expectations": 0,
                "with_profiles": 0,
                "with_llm_analysis": 0,
                "python_benchmarks": 0,
                "cuda_benchmarks": 0,
            }
        }
        
        # Scan all chapters
        for ch_dir in sorted(CODE_ROOT.glob("ch[0-9]*")):
            if ch_dir.is_dir():
                scan = self._detailed_scan(ch_dir, "chapter")
                if scan:
                    result['scan_results'].append(scan)
                    result['summary']['total_directories'] += 1
                    result['summary']['total_benchmarks'] += scan['benchmark_count']
                    result['summary']['python_benchmarks'] += scan['python_count']
                    result['summary']['cuda_benchmarks'] += scan['cuda_count']
                    if scan['has_expectations']:
                        result['summary']['with_expectations'] += 1
                    if scan['has_profiles']:
                        result['summary']['with_profiles'] += 1
                    if scan['llm_analysis_count'] > 0:
                        result['summary']['with_llm_analysis'] += 1
        
        # Scan all labs
        labs_dir = CODE_ROOT / 'labs'
        if labs_dir.exists():
            for lab_dir in sorted(labs_dir.iterdir()):
                if lab_dir.is_dir() and not lab_dir.name.startswith('.'):
                    scan = self._detailed_scan(lab_dir, "lab")
                    if scan:
                        result['scan_results'].append(scan)
                        result['summary']['total_directories'] += 1
                        result['summary']['total_benchmarks'] += scan['benchmark_count']
                        result['summary']['python_benchmarks'] += scan['python_count']
                        result['summary']['cuda_benchmarks'] += scan['cuda_count']
                        if scan['has_expectations']:
                            result['summary']['with_expectations'] += 1
                        if scan['has_profiles']:
                            result['summary']['with_profiles'] += 1
                        if scan['llm_analysis_count'] > 0:
                            result['summary']['with_llm_analysis'] += 1
        
        return result
    
    def _detailed_scan(self, directory: Path, dir_type: str) -> Optional[dict]:
        """Detailed scan of a single directory."""
        baseline_py = list(directory.glob("baseline_*.py"))
        baseline_cu = list(directory.glob("baseline_*.cu"))
        
        if not baseline_py and not baseline_cu:
            return None
        
        # Count optimized files
        optimized_py = list(directory.glob("optimized_*.py"))
        optimized_cu = list(directory.glob("optimized_*.cu"))
        
        # Check for expectations
        has_expectations = (
            (directory / 'expectations_b200.json').exists() or
            (directory / 'expectations_gb10.json').exists()
        )
        
        # Check for profiles
        profile_dir = CODE_ROOT / 'benchmark_profiles' / directory.name
        has_profiles = profile_dir.exists() and any(profile_dir.iterdir()) if profile_dir.exists() else False
        
        # Count LLM analysis files
        llm_analysis_count = 0
        if profile_dir.exists():
            llm_analysis_count = len(list(profile_dir.glob("llm_analysis_*.md")))
        
        # Also check for LLM explanation files in the directory itself
        llm_analysis_count += len(list(directory.glob("*_llm_explanation.md")))
        
        # Check for results in benchmark_test_results.json
        has_results = self._check_has_results(directory.name)
        
        benchmarks = []
        for baseline in baseline_py + baseline_cu:
            name = baseline.stem.replace("baseline_", "")
            file_type = "python" if baseline.suffix == ".py" else "cuda"
            
            # Find optimized variants
            optimized = [f.name for f in directory.glob(f"optimized_{name}*.py")] + \
                        [f.name for f in directory.glob(f"optimized_{name}*.cu")]
            
            benchmarks.append({
                "name": name,
                "type": file_type,
                "baseline": baseline.name,
                "optimized": optimized,
                "optimization_count": len(optimized),
            })
        
        return {
            "name": directory.name,
            "path": str(directory.relative_to(CODE_ROOT)),
            "type": dir_type,
            "benchmark_count": len(baseline_py) + len(baseline_cu),
            "python_count": len(baseline_py),
            "cuda_count": len(baseline_cu),
            "optimized_count": len(optimized_py) + len(optimized_cu),
            "has_expectations": has_expectations,
            "has_profiles": has_profiles,
            "has_results": has_results,
            "llm_analysis_count": llm_analysis_count,
            "benchmarks": benchmarks,
        }
    
    def _check_has_results(self, directory_name: str) -> bool:
        """Check if directory has results in benchmark_test_results.json."""
        results_file = CODE_ROOT / 'benchmark_test_results.json'
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    for result in data.get('results', []):
                        if result.get('chapter') == directory_name:
                            return True
                        # Also check labs path
                        if f"labs/{directory_name}" in str(result.get('chapter', '')):
                            return True
            except:
                pass
        return False
    
    def list_benchmark_targets(self) -> dict:
        """List all available benchmark targets in chapter:example format."""
        targets = []
        
        # Scan chapters
        for ch_dir in sorted(CODE_ROOT.glob('ch*')):
            if ch_dir.is_dir():
                chapter = ch_dir.name
                # Find all baseline files
                for baseline in ch_dir.glob('baseline_*.py'):
                    name = baseline.stem.replace('baseline_', '')
                    targets.append(f"{chapter}:{name}")
                for baseline in ch_dir.glob('baseline_*.cu'):
                    name = baseline.stem.replace('baseline_', '')
                    targets.append(f"{chapter}:{name}")
        
        # Scan labs
        labs_dir = CODE_ROOT / 'labs'
        if labs_dir.exists():
            for lab_dir in sorted(labs_dir.glob('*')):
                if lab_dir.is_dir():
                    lab_name = f"labs/{lab_dir.name}"
                    for baseline in lab_dir.glob('baseline_*.py'):
                        name = baseline.stem.replace('baseline_', '')
                        targets.append(f"{lab_name}:{name}")
        
        return {"targets": sorted(set(targets)), "count": len(set(targets))}
    
    def _transform_benchmark_data(self, raw_data: dict) -> dict:
        """Transform raw benchmark data to dashboard format."""
        benchmarks = []
        all_speedups = []
        
        for chapter_result in raw_data.get('results', []):
            chapter = chapter_result.get('chapter', 'unknown')
            
            for bench in chapter_result.get('benchmarks', []):
                name = bench.get('example', 'unknown')
                baseline_time = bench.get('baseline_time_ms', 0)
                best_speedup = bench.get('best_speedup', 1.0)
                status = bench.get('status', 'unknown')
                bench_type = bench.get('type', 'python')
                
                # Get best optimized time
                optimized_time = baseline_time / best_speedup if best_speedup > 0 else baseline_time
                
                # Extract GPU metrics if available
                gpu_metrics = bench.get('baseline_gpu_metrics', {})
                
                # Extract optimization details
                optimizations = []
                for opt in bench.get('optimizations', []):
                    optimizations.append({
                        'technique': opt.get('technique', ''),
                        'speedup': opt.get('speedup', 1.0),
                        'time_ms': opt.get('time_ms', 0),
                        'file': opt.get('file', '')
                    })
                
                # Extract LLM patch info including verification
                llm_patch_info = None
                llm_patches = bench.get('llm_patches', [])
                best_patch = bench.get('best_llm_patch')
                
                # Count refinement attempts across all patches
                total_refinements = sum(1 for p in llm_patches if p.get('refined'))
                total_refinement_attempts = sum(p.get('refinement_attempts', 0) for p in llm_patches)
                
                if best_patch:
                    llm_patch_info = {
                        'variant_name': best_patch.get('variant_name'),
                        'speedup': best_patch.get('actual_speedup'),
                        'refined': best_patch.get('refined', False),
                        'refinement_attempts': best_patch.get('refinement_attempts', 0),
                    }
                
                # Aggregate LLM patch stats
                llm_stats = None
                if llm_patches:
                    successful = [p for p in llm_patches if p.get('success')]
                    failed = [p for p in llm_patches if not p.get('success')]
                    verified = [p for p in llm_patches if p.get('verification', {}).get('verified')]
                    llm_stats = {
                        'total': len(llm_patches),
                        'successful': len(successful),
                        'failed': len(failed),
                        'refined': total_refinements,
                        'refinement_attempts': total_refinement_attempts,
                        'verified': len(verified),
                    }
                
                # Extract verification status - check both LLM patches and direct verification
                verification_status = None
                # First check direct verification (baseline vs optimized)
                direct_verification = bench.get('verification')
                if direct_verification:
                    verification_status = {
                        'verified': direct_verification.get('verified', False),
                        'type': direct_verification.get('verification_type', 'output_comparison'),
                        'errors': direct_verification.get('errors', []),
                        'details': direct_verification.get('details', {}),
                    }
                # Then check LLM patch verification
                for patch in llm_patches:
                    if patch.get('verification'):
                        v = patch['verification']
                        verification_status = {
                            'verified': v.get('verified', False),
                            'type': v.get('verification_type', 'unknown'),
                            'errors': v.get('errors', []),
                            'details': v.get('details', {}),
                        }
                        break
                
                benchmarks.append({
                    'name': name,
                    'chapter': chapter,
                    'type': bench_type,
                    'baseline_time_ms': baseline_time,
                    'optimized_time_ms': optimized_time,
                    'speedup': best_speedup,
                    'status': status,
                    'gpu_temp': gpu_metrics.get('temperature_gpu_c'),
                    'gpu_power': gpu_metrics.get('power_draw_w'),
                    'gpu_util': gpu_metrics.get('utilization_gpu_pct'),
                    'optimizations': optimizations,
                    'error': bench.get('error'),
                    'p75_ms': bench.get('baseline_p75_ms'),
                    'llm_patch': llm_patch_info,
                    'llm_stats': llm_stats,
                    'verification': verification_status,
                })
                
                if best_speedup > 0:
                    all_speedups.append(best_speedup)
        
        # Sort by speedup descending
        benchmarks.sort(key=lambda x: x['speedup'], reverse=True)
        
        # Calculate summary
        summary = raw_data.get('results', [{}])[0].get('summary', {}) if raw_data.get('results') else {}
        
        return {
            "timestamp": raw_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
            "benchmarks": benchmarks,
            "summary": {
                "total_benchmarks": len(benchmarks),
                "avg_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else 0,
                "max_speedup": max(all_speedups) if all_speedups else 0,
                "min_speedup": min(all_speedups) if all_speedups else 0,
                "successful": summary.get('successful', 0),
                "failed": summary.get('failed', 0),
                "failed_regression": summary.get('failed_regression', 0),
            }
        }
    
    def load_llm_analysis(self) -> dict:
        """Load ALL LLM analysis files from entire codebase."""
        analysis = []
        
        # 1. Scan benchmark_profiles directory
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        if profiles_dir.exists():
            for md_file in profiles_dir.rglob('llm_analysis*.md'):
                try:
                    content = md_file.read_text()
                    parts = md_file.relative_to(profiles_dir).parts
                    chapter = parts[0] if parts else 'unknown'
                    name = md_file.stem.replace('llm_analysis_', '')
                    
                    analysis.append({
                        'chapter': chapter,
                        'name': name,
                        'content': content,
                        'path': str(md_file.relative_to(CODE_ROOT)),
                        'source': 'benchmark_profiles',
                    })
                except Exception as e:
                    print(f"Error loading {md_file}: {e}")
        
        # 2. Scan ALL chapters for LLM explanation files
        for ch_dir in CODE_ROOT.glob("ch[0-9]*"):
            if ch_dir.is_dir():
                for md_file in ch_dir.glob('*_llm_explanation.md'):
                    try:
                        content = md_file.read_text()
                        analysis.append({
                            'chapter': ch_dir.name,
                            'name': md_file.stem.replace('_llm_explanation', ''),
                            'content': content,
                            'path': str(md_file.relative_to(CODE_ROOT)),
                            'source': 'chapter',
                            'type': 'explanation'
                        })
                    except Exception as e:
                        print(f"Error loading {md_file}: {e}")
        
        # 3. Scan ALL labs for LLM analysis/explanation files
        labs_dir = CODE_ROOT / 'labs'
        if labs_dir.exists():
            for lab_dir in labs_dir.iterdir():
                if lab_dir.is_dir():
                    # Look for any LLM-related markdown
                    for md_file in lab_dir.glob('*llm*.md'):
                        try:
                            content = md_file.read_text()
                            analysis.append({
                                'chapter': f"labs/{lab_dir.name}",
                                'name': md_file.stem,
                                'content': content,
                                'path': str(md_file.relative_to(CODE_ROOT)),
                                'source': 'lab',
                            })
                        except Exception as e:
                            print(f"Error loading {md_file}: {e}")
                    
                    # Note: TECHNIQUE*.md files are reference docs, not LLM analysis - skip them
        
        # Note: Root *ANALYSIS*.md files are reports, not LLM-generated - skip them
        
        return {
            "analyses": analysis, 
            "count": len(analysis),
            "sources": {
                "benchmark_profiles": len([a for a in analysis if a.get('source') == 'benchmark_profiles']),
                "chapters": len([a for a in analysis if a.get('source') == 'chapter']),
                "labs": len([a for a in analysis if a.get('source') == 'lab']),
                "root": len([a for a in analysis if a.get('source') == 'root']),
            }
        }
    
    def load_profile_data(self) -> dict:
        """Load available profile data from ALL sources."""
        profiles = []
        
        # Scan benchmark_profiles directory
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        if profiles_dir.exists():
            for chapter_dir in profiles_dir.iterdir():
                if chapter_dir.is_dir():
                    chapter = chapter_dir.name
                    chapter_profiles = {
                        'chapter': chapter,
                        'nsys_reports': [],
                        'ncu_reports': [],
                        'torch_traces': [],
                        'sqlite_dbs': [],
                    }
                    
                    for f in chapter_dir.iterdir():
                        if f.suffix == '.nsys-rep':
                            chapter_profiles['nsys_reports'].append(f.name)
                        elif f.suffix == '.ncu-rep':
                            chapter_profiles['ncu_reports'].append(f.name)
                        elif f.suffix == '.json' and 'torch_trace' in f.name:
                            chapter_profiles['torch_traces'].append(f.name)
                        elif f.suffix == '.sqlite':
                            chapter_profiles['sqlite_dbs'].append(f.name)
                    
                    if any([chapter_profiles['nsys_reports'], 
                            chapter_profiles['ncu_reports'],
                            chapter_profiles['torch_traces'],
                            chapter_profiles['sqlite_dbs']]):
                        profiles.append(chapter_profiles)
        
        return {
            "profiles": profiles,
            "total_chapters_with_profiles": len(profiles),
            "total_nsys": sum(len(p['nsys_reports']) for p in profiles),
            "total_ncu": sum(len(p['ncu_reports']) for p in profiles),
            "total_torch_traces": sum(len(p['torch_traces']) for p in profiles),
        }
    
    def get_gpu_info(self) -> dict:
        """Get GPU information using nvidia-smi."""
        return super().get_gpu_info()
    
    def get_software_info(self) -> dict:
        """Get software version information for performance-impacting libraries."""
        return super().get_software_info()
    
    def get_dependency_health(self) -> dict:
        """Check health of critical dependencies (CUTLASS, TransformerEngine).

        This endpoint helps diagnose SM100a (Blackwell) build issues.
        """
        return super().get_dependency_health()
    
    def check_dependency_updates(self) -> dict:
        """Check for upstream updates to CUTLASS and TransformerEngine.
        
        This calls the check_upstream_versions script and returns the results.
        Note: This makes GitHub API calls which are rate-limited.
        """
        import subprocess
        
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "scripts" / "check_upstream_versions.py"
        
        result = {
            "checked": False,
            "error": None,
            "cutlass": None,
            "transformer_engine": None,
            "te_bundled_cutlass": None,
            "any_updates": False,
        }
        
        if not script.exists():
            result["error"] = "check_upstream_versions.py not found"
            return result
        
        try:
            proc = subprocess.run(
                ["python3", str(script), "--json", "--check-te-cutlass"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(project_root),
            )
            
            if proc.returncode in (0, 1):  # 0=up to date, 1=updates available
                import json
                data = json.loads(proc.stdout)
                result["checked"] = True
                result["cutlass"] = data.get("cutlass")
                result["transformer_engine"] = data.get("transformer_engine")
                result["te_bundled_cutlass"] = data.get("te_bundled_cutlass")
                result["any_updates"] = data.get("any_updates_available", False)
            else:
                result["error"] = proc.stderr or "Unknown error"
                
        except subprocess.TimeoutExpired:
            result["error"] = "Timeout checking updates (GitHub API may be slow)"
        except json.JSONDecodeError as e:
            result["error"] = f"Failed to parse response: {e}"
        except Exception as e:
            error_str = str(e)
            if "rate limit" in error_str.lower() or "403" in error_str:
                result["error"] = "GitHub API rate limit exceeded. Set GITHUB_TOKEN env var."
            else:
                result["error"] = str(e)
        
        return result
    
    # =========================================================================
    # PARALLELISM PLANNER API METHODS
    # =========================================================================
    
    def get_nccl_recommendations(self, nodes: int, gpus: int, diagnose: bool) -> dict:
        """Get NCCL tuning recommendations."""
        try:
            from core.optimization.parallelism_planner.distributed_training import NCCLTuningAdvisor, NCCLConfig
            
            advisor = NCCLTuningAdvisor()
            config = NCCLConfig(
                num_nodes=nodes,
                gpus_per_node=gpus,
            )
            
            if diagnose:
                result = advisor.diagnose_issues()
            else:
                result = advisor.get_recommendations(config)
            
            return {"success": True, "recommendations": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_rlhf_optimization(self, model: str, algorithm: str, compare: bool) -> dict:
        """Get RLHF memory and optimization recommendations."""
        try:
            from core.optimization.parallelism_planner.distributed_training import RLHFMemoryCalculator, RLHFAlgorithm
            
            alg_map = {
                "ppo": RLHFAlgorithm.PPO,
                "dpo": RLHFAlgorithm.DPO,
                "grpo": RLHFAlgorithm.GRPO,
                "reinforce": RLHFAlgorithm.REINFORCE,
            }
            
            calculator = RLHFMemoryCalculator()
            if compare:
                results = {}
                for alg_name, alg in alg_map.items():
                    results[alg_name] = calculator.calculate(model, alg).__dict__
                return {"success": True, "comparison": results}
            else:
                alg = alg_map.get(algorithm.lower(), RLHFAlgorithm.PPO)
                result = calculator.calculate(model, alg)
                return {"success": True, "memory_estimate": result.__dict__}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_moe_optimization(self, model: str) -> dict:
        """Get MoE parallelism optimization recommendations."""
        try:
            from core.optimization.parallelism_planner.distributed_training import MoEOptimizer
            
            optimizer = MoEOptimizer()
            result = optimizer.optimize(model)
            return {"success": True, "moe_config": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_long_context_optimization(self, model: str, seq_length: int) -> dict:
        """Get long-context optimization recommendations."""
        try:
            from core.optimization.parallelism_planner.distributed_training import LongContextOptimizer
            
            optimizer = LongContextOptimizer()
            result = optimizer.optimize(model, seq_length)
            return {"success": True, "long_context_config": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_vllm_config(self, model: str, target: str, compare: bool) -> dict:
        """Get vLLM configuration or compare inference engines."""
        try:
            from core.optimization.parallelism_planner.distributed_training import VLLMConfigGenerator
            
            generator = VLLMConfigGenerator()
            if compare:
                result = generator.compare_engines(model)
                return {"success": True, "engine_comparison": result}
            else:
                result = generator.generate(model, target=target)
                return {"success": True, "vllm_config": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_comm_overlap_analysis(self, model: str) -> dict:
        """Get communication-computation overlap analysis."""
        try:
            from core.optimization.parallelism_planner.distributed_training import CommunicationOverlapAnalyzer
            
            analyzer = CommunicationOverlapAnalyzer()
            result = analyzer.analyze(model)
            return {"success": True, "overlap_analysis": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_slurm_script(self, model: str, nodes: int, gpus: int, framework: str) -> dict:
        """Generate SLURM job script for distributed training."""
        try:
            from core.optimization.parallelism_planner.extras import JobScriptGenerator

            generator = JobScriptGenerator()
            launch_cmd = (
                f"torchrun --nnodes={nodes} --nproc_per_node={gpus} "
                f"train.py --model {model} --framework {framework}"
            )
            script = generator.generate_slurm(
                job_name=f"{model}-{framework}",
                num_nodes=nodes,
                gpus_per_node=gpus,
                time_hours=24,
                partition="gpu",
                script="train.py",
                launch_command=launch_cmd,
            )
            return {"success": True, "slurm_script": script}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_speed_tests(self) -> dict:
        """Run disk and network speed tests. Returns speed in MB/s or Gb/s."""
        import tempfile
        import time
        
        results = {
            "disk_read_speed": None,
            "disk_write_speed": None,
            "nfs_read_speed": None,
            "nfs_write_speed": None,
            "local_disk_path": None,
            "nfs_path": None,
            "test_size_mb": 256,
            "timestamp": None,
        }
        
        TEST_SIZE_MB = 256
        TEST_SIZE_BYTES = TEST_SIZE_MB * 1024 * 1024
        
        # Find local disk (prefer /tmp or first non-NFS mount)
        local_path = "/tmp"
        nfs_path = None
        
        # Detect NFS mounts
        try:
            result = subprocess.run(
                ['mount', '-t', 'nfs,nfs4'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if ' on ' in line:
                        parts = line.split(' on ')
                        if len(parts) >= 2:
                            mount_point = parts[1].split(' type ')[0].strip()
                            if os.path.isdir(mount_point) and os.access(mount_point, os.W_OK):
                                nfs_path = mount_point
                                break
        except Exception:
            pass
        
        results["local_disk_path"] = local_path
        results["nfs_path"] = nfs_path
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Helper to format speed
        def format_speed(mb_per_sec):
            if mb_per_sec >= 1000:
                return f"{mb_per_sec/1000:.1f} GB/s"
            return f"{mb_per_sec:.0f} MB/s"
        
        # Test local disk write speed
        try:
            test_file = os.path.join(local_path, f".speedtest_{os.getpid()}.tmp")
            data = os.urandom(TEST_SIZE_BYTES)
            
            # Write test
            start = time.perf_counter()
            with open(test_file, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            write_time = time.perf_counter() - start
            write_speed = TEST_SIZE_MB / write_time if write_time > 0 else 0
            results["disk_write_speed"] = format_speed(write_speed)
            
            # Read test (clear cache if possible)
            try:
                subprocess.run(['sync'], timeout=5)
                # Drop caches requires root, skip if not available
                try:
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('3')
                except (PermissionError, FileNotFoundError):
                    pass
            except Exception:
                pass
            
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                _ = f.read()
            read_time = time.perf_counter() - start
            read_speed = TEST_SIZE_MB / read_time if read_time > 0 else 0
            results["disk_read_speed"] = format_speed(read_speed)
            
            # Cleanup
            os.remove(test_file)
        except Exception as e:
            results["disk_error"] = str(e)
        
        # Test NFS speed if available
        if nfs_path:
            try:
                test_file = os.path.join(nfs_path, f".speedtest_{os.getpid()}.tmp")
                data = os.urandom(TEST_SIZE_BYTES)
                
                # NFS Write test
                start = time.perf_counter()
                with open(test_file, 'wb') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                write_time = time.perf_counter() - start
                write_speed = TEST_SIZE_MB / write_time if write_time > 0 else 0
                results["nfs_write_speed"] = format_speed(write_speed)
                
                # NFS Read test
                start = time.perf_counter()
                with open(test_file, 'rb') as f:
                    _ = f.read()
                read_time = time.perf_counter() - start
                read_speed = TEST_SIZE_MB / read_time if read_time > 0 else 0
                results["nfs_read_speed"] = format_speed(read_speed)
                
                # Cleanup
                os.remove(test_file)
            except Exception as e:
                results["nfs_error"] = str(e)
        
        # Network speed via iperf3 (if server available)
        try:
            # Check if iperf3 is available and try localhost test
            result = subprocess.run(
                ['which', 'iperf3'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                results["iperf3_available"] = True
                # Note: actual iperf3 test requires a server endpoint
                # We just note it's available for manual testing
        except Exception:
            pass
        
        return results
    
    def run_gpu_bandwidth_test(self) -> dict:
        """Run GPU memory bandwidth and P2P bandwidth tests using PyTorch."""
        import time
        
        results = {
            "hbm_bandwidth_gb_s": None,
            "h2d_bandwidth_gb_s": None,
            "d2h_bandwidth_gb_s": None,
            "p2p_bandwidth": [],
            "gpu_count": 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_size_mb": 256,
        }
        
        try:
            import torch
            if not torch.cuda.is_available():
                results["error"] = "CUDA not available"
                return results
            
            gpu_count = torch.cuda.device_count()
            results["gpu_count"] = gpu_count
            
            TEST_SIZE = 256 * 1024 * 1024  # 256 MB
            TEST_SIZE_GB = TEST_SIZE / (1024**3)
            ITERATIONS = 10
            
            # HBM (device-to-device) bandwidth test on GPU 0
            torch.cuda.set_device(0)
            torch.cuda.synchronize()
            
            # Allocate test tensors
            src = torch.randn(TEST_SIZE // 4, dtype=torch.float32, device='cuda')
            dst = torch.empty_like(src)
            
            # Warm up
            for _ in range(3):
                dst.copy_(src)
            torch.cuda.synchronize()
            
            # HBM bandwidth test
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                dst.copy_(src)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            # Bandwidth = 2 * size * iterations / time (read + write)
            hbm_bw = (2 * TEST_SIZE_GB * ITERATIONS) / elapsed
            results["hbm_bandwidth_gb_s"] = round(hbm_bw, 1)
            
            # Host-to-Device bandwidth
            host_tensor = torch.randn(TEST_SIZE // 4, dtype=torch.float32, pin_memory=True)
            device_tensor = torch.empty(TEST_SIZE // 4, dtype=torch.float32, device='cuda')
            
            # Warm up
            for _ in range(3):
                device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            h2d_bw = (TEST_SIZE_GB * ITERATIONS) / elapsed
            results["h2d_bandwidth_gb_s"] = round(h2d_bw, 1)
            
            # Device-to-Host bandwidth
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                host_tensor.copy_(device_tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            d2h_bw = (TEST_SIZE_GB * ITERATIONS) / elapsed
            results["d2h_bandwidth_gb_s"] = round(d2h_bw, 1)
            
            # P2P bandwidth between GPUs (if multiple GPUs)
            if gpu_count > 1:
                p2p_results = []
                for src_gpu in range(min(gpu_count, 4)):  # Test up to 4 GPUs
                    for dst_gpu in range(min(gpu_count, 4)):
                        if src_gpu != dst_gpu:
                            try:
                                # Check if P2P is possible
                                can_p2p = torch.cuda.can_device_access_peer(src_gpu, dst_gpu)
                                
                                if can_p2p:
                                    torch.cuda.set_device(src_gpu)
                                    src_tensor = torch.randn(TEST_SIZE // 4, dtype=torch.float32, device=f'cuda:{src_gpu}')
                                    dst_tensor = torch.empty(TEST_SIZE // 4, dtype=torch.float32, device=f'cuda:{dst_gpu}')
                                    
                                    # Warm up
                                    for _ in range(3):
                                        dst_tensor.copy_(src_tensor)
                                    torch.cuda.synchronize()
                                    
                                    start = time.perf_counter()
                                    for _ in range(ITERATIONS):
                                        dst_tensor.copy_(src_tensor)
                                    torch.cuda.synchronize()
                                    elapsed = time.perf_counter() - start
                                    
                                    p2p_bw = (TEST_SIZE_GB * ITERATIONS) / elapsed
                                    p2p_results.append({
                                        "src": src_gpu,
                                        "dst": dst_gpu,
                                        "bandwidth_gb_s": round(p2p_bw, 1),
                                        "nvlink": p2p_bw > 25,  # NVLink typically > 25 GB/s
                                    })
                                    
                                    del src_tensor, dst_tensor
                                else:
                                    p2p_results.append({
                                        "src": src_gpu,
                                        "dst": dst_gpu,
                                        "bandwidth_gb_s": None,
                                        "nvlink": False,
                                        "note": "P2P not enabled",
                                    })
                            except Exception as e:
                                p2p_results.append({
                                    "src": src_gpu,
                                    "dst": dst_gpu,
                                    "error": str(e)[:50],
                                })
                
                results["p2p_bandwidth"] = p2p_results
            
            # Cleanup
            del src, dst, host_tensor, device_tensor
            torch.cuda.empty_cache()
            
        except ImportError:
            results["error"] = "PyTorch not available"
        except Exception as e:
            results["error"] = str(e)[:100]
        
        return results

    def roofline_sweep(self, size_mb: int = 32, strides: Optional[List[int]] = None) -> dict:
        """Stride sweep for memory roofline."""
        from core.diagnostics import microbench
        rows = []
        strides = strides or [32, 64, 128, 256, 512, 1024, 2048, 4096]
        for stride in strides:
            res = microbench.mem_hierarchy_test(size_mb=size_mb, stride=stride)
            rows.append({"stride": stride, "bandwidth_gbps": res.get("bandwidth_gbps")})
        return {"size_mb": size_mb, "rows": rows}

    def generate_launch_plan_from_query(self, params: Dict[str, List[str]]) -> dict:
        """Generate launch plan JSON from query parameters."""
        try:
            from core.optimization.parallelism_planner.launch_plan import generate_launch_plan
            plan = generate_launch_plan(
                model_params=int((params.get("model_params") or ["70"])[0]),
                nodes=int((params.get("nodes") or ["1"])[0]),
                gpus_per_node=int((params.get("gpus") or ["8"])[0]),
                tp=int((params.get("tp") or ["1"])[0]),
                pp=int((params.get("pp") or ["1"])[0]),
                dp=int((params.get("dp") or ["1"])[0]),
                batch_size=int((params.get("batch_size") or ["1"])[0]),
                script=(params.get("script") or ["train.py"])[0],
                extra_args=(params.get("extra_args") or [None])[0],
            )
            return {"command": plan.command, "plan": json.loads(plan.to_json())}
        except Exception as exc:
            return {"error": str(exc)}

    def export_results_from_query(self, params: Dict[str, List[str]]) -> dict:
        """Export benchmark results to csv/markdown/json."""
        fmt = (params.get("format") or ["csv"])[0].lower()
        analyzer = self.analyzer
        data = analyzer._load_data()  # type: ignore[attr-defined]
        benchmarks = data.get("benchmarks", [])
        if fmt == "json":
            return {"format": "json", "payload": data}
        elif fmt == "markdown":
            lines = ["| Benchmark | Speedup | Baseline (ms) | Type |", "|---|---|---|---|"]
            for b in benchmarks:
                lines.append(
                    f"| {b.get('chapter')}:{b.get('name')} | {b.get('speedup', 0):.2f}x | {b.get('baseline_time_ms', 0):.3f} | {b.get('type', 'python')} |"
                )
            return {"format": "markdown", "payload": "\n".join(lines)}
        elif fmt == "csv":
            import csv
            from io import StringIO
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["benchmark", "speedup", "baseline_ms", "type"])
            for b in benchmarks:
                writer.writerow(
                    [
                        f"{b.get('chapter')}:{b.get('name')}",
                        f"{b.get('speedup', 0):.2f}",
                        f"{b.get('baseline_time_ms', 0):.3f}",
                        b.get("type", "python"),
                    ]
                )
            return {"format": "csv", "payload": buf.getvalue()}
        else:
            return {"error": "format must be csv|markdown|json"}

    def compare_runs_from_query(self, params: Dict[str, List[str]]) -> dict:
        """Compare two benchmark JSON payloads on the server side."""
        base = (params.get("baseline") or [None])[0]
        cand = (params.get("candidate") or [None])[0]
        top = int((params.get("top") or ["10"])[0])
        if not base or not cand:
            return {"error": "baseline and candidate required"}

        def _load(path: str) -> dict:
            with open(path) as f:
                return json.load(f)

        try:
            b = _load(base)
            c = _load(cand)
        except Exception as exc:
            return {"error": str(exc)}

        def _flatten(blob: dict) -> dict:
            flat = {}
            for chapter in blob.get("results", []):
                chap = chapter.get("chapter", "unknown")
                for bench in chapter.get("benchmarks", []):
                    key = f"{chap}:{bench.get('example', bench.get('name', 'unknown'))}"
                    flat[key] = bench
            return flat

        bflat = _flatten(b)
        cflat = _flatten(c)
        deltas = []
        for key, cand_bench in cflat.items():
            base_bench = bflat.get(key)
            if not base_bench:
                continue
            delta = (cand_bench.get("best_speedup", 0) or 0) - (base_bench.get("best_speedup", 0) or 0)
            deltas.append((key, delta, base_bench.get("best_speedup", 0) or 0, cand_bench.get("best_speedup", 0) or 0))
        deltas.sort(key=lambda x: x[1])
        regressions = [d for d in deltas if d[1] < 0][:top]
        improvements = sorted([d for d in deltas if d[1] > 0], key=lambda x: -x[1])[:top]
        return {
            "regressions": [{"name": n, "delta": d, "baseline": b, "candidate": c} for n, d, b, c in regressions],
            "improvements": [{"name": n, "delta": d, "baseline": b, "candidate": c} for n, d, b, c in improvements],
        }

    def generate_report_from_query(self, params: Dict[str, List[str]]) -> dict:
        """Generate PDF/HTML report and return the output path."""
        data_file = (params.get("data_file") or [None])[0]
        output = Path((params.get("output") or ["report.pdf"])[0])
        fmt = (params.get("format") or ["pdf"])[0]
        title = (params.get("title") or ["GPU Performance Report"])[0]
        author = (params.get("author") or ["AI Performance Engineering"])[0]
        try:
            from core.analysis.reporting.generator import generate_report, ReportConfig
            cfg = ReportConfig(title=title, author=author)
            path = generate_report(data_file or "benchmark_test_results.json", str(output), format=fmt, config=cfg)
            return {"output": str(path)}
        except Exception as exc:
            return {"error": str(exc)}
    
    def get_full_system_context(self) -> dict:
        """Get complete system context optimized for LLM-based analysis."""
        return super().get_full_system_context()
    
    def llm_analyze(self, params: dict) -> dict:
        """
        Analyze performance data using LLM.
        
        The LLM receives rich context from our rule-based analysis
        to provide informed, specific recommendations.
        """
        engine = self._get_llm_engine()
        
        if not engine:
            error = dict(self.LLM_SETUP_ERROR)
            if hasattr(self, '_llm_init_error'):
                error["init_error"] = self._llm_init_error
            return error
        
        # Build rich context from our rules/analysis
        rule_context = self._build_rule_context(params)
        
        analysis_type = params.get("type", "profile")
        
        # Enhance the prompt with our rule-based context
        enhanced_context = {
            "rule_based_analysis": rule_context,
            "user_context": params.get("context", {}),
            "analysis_type": analysis_type
        }
        
        if analysis_type == "profile":
            profile_data = {
                "kernels": rule_context.get("profiling", {}),
                "detected_bottlenecks": rule_context.get("detected_patterns", []),
                "hardware": rule_context.get("hardware", {}),
                "optimization_opportunities": rule_context.get("optimization_opportunities", {})
            }
            response = engine.analyze_profile(
                profile_data=profile_data,
                constraints=params.get("constraints", {}),
                workload_info=params.get("workload", {})
            )
        elif analysis_type == "distributed":
            response = engine.analyze_distributed(
                cluster_info=params.get("cluster", {}),
                performance_data=params.get("performance", {}),
                training_config=params.get("training_config", {}),
                comm_patterns=params.get("comm_patterns", {})
            )
        elif analysis_type == "inference":
            response = engine.analyze_inference(
                model_info=params.get("model", {}),
                serving_config=params.get("serving_config", {}),
                metrics=params.get("metrics", {}),
                traffic_pattern=params.get("traffic", {})
            )
        elif analysis_type == "rlhf":
            response = engine.analyze_rlhf(
                model_config=params.get("model_config", {}),
                algorithm=params.get("algorithm", "ppo"),
                actor_info=params.get("actor", {}),
                critic_info=params.get("critic", {}),
                reference_info=params.get("reference", {}),
                reward_info=params.get("reward", {}),
                performance_data=params.get("performance", {}),
                memory_usage=params.get("memory", {})
            )
        else:
            response = engine.ask(
                params.get("question", "Analyze this system for optimization opportunities"),
                context=enhanced_context
            )
        
        return {
            "success": True,
            "llm_powered": True,
            "analysis": response,
            "context_provided": {
                "gpu": rule_context.get("hardware", {}).get("gpu", {}).get("name", "Unknown"),
                "architecture": rule_context.get("hardware", {}).get("architecture", "Unknown"),
                "kernel_count": len(rule_context.get("profiling", {}).get("top_kernels", [])),
                "detected_patterns": len(rule_context.get("detected_patterns", [])),
                "quick_wins_identified": len(rule_context.get("optimization_opportunities", {}).get("quick_wins", []))
            }
        }
    
    def llm_recommend(self, params: dict) -> dict:
        """
        Get LLM-powered optimization recommendations.
        
        Uses the full system context to generate tailored recommendations.
        """
        advisor = self._get_llm_advisor()
        
        if not advisor:
            error = dict(self.LLM_SETUP_ERROR)
            if hasattr(self, '_llm_advisor_init_error'):
                error["init_error"] = self._llm_advisor_init_error
            return error
        
        try:
            # Build system context from real data
            hw_caps = self.get_hardware_capabilities()
            gpu_info = hw_caps.get("gpu", {})
            
            context = SystemContext(
                gpu_name=gpu_info.get("name", "Unknown"),
                gpu_architecture=hw_caps.get("architecture", "Unknown"),
                gpu_memory_gb=gpu_info.get("memory_gb", 80),
                gpu_count=params.get("gpu_count", 8),
                model_name=params.get("model", ""),
                model_params_b=params.get("model_params_b", 70),
                batch_size=params.get("batch_size", 8),
                sequence_length=params.get("sequence_length", 4096),
                is_training=params.get("is_training", True),
                precision=params.get("precision", "bf16"),
                tensor_parallel=params.get("tensor_parallel", 1),
                pipeline_parallel=params.get("pipeline_parallel", 1),
                data_parallel=params.get("data_parallel", 8),
            )
            
            goal_map = {
                "throughput": OptimizationGoal.THROUGHPUT,
                "latency": OptimizationGoal.LATENCY,
                "memory": OptimizationGoal.MEMORY,
                "efficiency": OptimizationGoal.EFFICIENCY,
                "cost": OptimizationGoal.COST,
            }
            
            request = OptimizationRequest(
                context=context,
                goal=goal_map.get(params.get("goal", "throughput"), OptimizationGoal.THROUGHPUT),
                constraints=params.get("constraints", []),
                specific_questions=params.get("questions", []),
            )
            
            advice = advisor.get_advice(request)
            
            return {
                "success": True,
                "llm_powered": True,
                "summary": advice.summary,
                "recommendations": advice.priority_recommendations,
                "parallelism": advice.parallelism_changes,
                "memory_optimizations": advice.memory_optimizations,
                "kernel_optimizations": advice.kernel_optimizations,
                "communication_optimizations": advice.communication_optimizations,
                "compound_strategies": advice.compound_strategies,
                "launch_command": advice.launch_command,
                "environment_variables": advice.environment_variables,
                "expected_improvements": advice.expected_improvements,
                "warnings": advice.warnings,
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "fallback": self._fallback_recommendations(params)
            }
    
    def llm_chat(self, params: dict) -> dict:
        """
        Chat with the LLM about performance optimization.
        
        This provides an interactive Q&A interface powered by real LLM.
        """
        engine = self._get_llm_engine()
        
        if not engine:
            return {
                "error": "LLM engine not available",
                "hint": "Start Ollama (ollama serve) or set OPENAI_API_KEY"
            }
        
        question = params.get("question", "")
        if not question:
            return {"error": "No question provided"}
        
        try:
            # Collect context
            hw_caps = self.get_hardware_capabilities()
            kernel_data = self.get_kernel_data()
            
            context = {
                "hardware": hw_caps,
                "kernel_summary": kernel_data.get("summary", {}),
                "top_kernels": kernel_data.get("kernels", [])[:10],
                "user_context": params.get("context", {})
            }
            
            response = engine.ask(question, context=context)
            
            return {
                "success": True,
                "llm_powered": True,
                "question": question,
                "answer": response,
                "context_used": {
                    "gpu": hw_caps.get("gpu", {}).get("name", "Unknown"),
                    "kernel_count": len(kernel_data.get("kernels", []))
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_llm_status(self, probe: bool = False) -> dict:
        """
        Get status of LLM backends using unified client.
        
        LLM is REQUIRED for this platform - no fallbacks!
        """
        try:
            from core.llm import get_llm_status as unified_status, get_config
            status = unified_status(probe=probe)
            
            # Add backend details for UI
            config = get_config()
            status["backends"] = [{
                "name": config.provider.title(),
                "available": status["available"],
                "model": config.model,
                "type": "api" if config.provider in ('openai', 'anthropic') else "local"
            }]
            status["probed"] = probe
            
            return status
            
        except Exception as e:
            status = {
                "available": False,
                "provider": None,
                "model": None,
                "backends": [],
                "error": str(e),
                "probed": probe,
            }
        
        # If no LLM available, provide setup instructions
        if not status.get("available"):
            status["setup_required"] = True
            status["setup_instructions"] = self.LLM_SETUP_ERROR.get("setup_instructions", [])
        
        return status
    
    def get_distributed_status(self) -> dict:
        """Get distributed training capabilities status."""
        return {
            "distributed_analyzer_available": DISTRIBUTED_AVAILABLE,
            "capabilities": {
                "cluster_topology_analysis": True,
                "parallelism_recommendation": True,
                "scaling_efficiency_analysis": True,
                "communication_bottleneck_analysis": True,
                "nccl_optimization": True
            },
            "supported_frameworks": [
                "PyTorch FSDP",
                "DeepSpeed ZeRO",
                "Megatron-LM",
                "PyTorch DDP"
            ],
            "supported_parallelism": [
                "Tensor Parallel (TP)",
                "Pipeline Parallel (PP)",
                "Data Parallel (DP)",
                "Context Parallel (CP)",
                "Expert Parallel (EP)"
            ]
        }
    
    def get_inference_status(self) -> dict:
        """Get inference optimization capabilities status."""
        return {
            "vllm_optimizer_available": VLLM_AVAILABLE,
            "inference_optimizer_available": INFERENCE_OPTIMIZER_AVAILABLE,
            "capabilities": {
                "vllm_configuration": True,
                "quantization_analysis": True,
                "speculative_decoding": True,
                "continuous_batching": True,
                "prefix_caching": True
            },
            "supported_frameworks": [
                "vLLM",
                "TensorRT-LLM",
                "Text Generation Inference (TGI)",
                "llama.cpp"
            ],
            "supported_optimizations": [
                "FP8 Quantization",
                "INT8/INT4 Quantization",
                "Flash Attention",
                "CUDA Graphs",
                "Speculative Decoding",
                "Continuous Batching",
                "Prefix Caching"
            ]
        }
    
    def get_optimization_presets(self) -> dict:
        """Get pre-configured optimization presets."""
        return {
            "presets": [
                {
                    "id": "training_memory",
                    "name": "Memory-Efficient Training",
                    "description": "Maximize model size in available memory",
                    "techniques": ["FSDP", "Activation Checkpointing", "BF16", "Flash Attention"],
                    "use_case": "Training large models on limited GPU memory"
                },
                {
                    "id": "training_throughput",
                    "name": "Maximum Training Throughput",
                    "description": "Maximize tokens/second during training",
                    "techniques": ["torch.compile", "CUDA Graphs", "TP/PP", "Large Batches"],
                    "use_case": "Fast training when memory is not a constraint"
                },
                {
                    "id": "inference_latency",
                    "name": "Low-Latency Inference",
                    "description": "Minimize time-to-first-token and inter-token latency",
                    "techniques": ["FP8", "Flash Attention", "Speculative Decoding", "CUDA Graphs"],
                    "use_case": "Real-time applications, chatbots"
                },
                {
                    "id": "inference_throughput",
                    "name": "High-Throughput Serving",
                    "description": "Maximize requests/second for batch inference",
                    "techniques": ["Continuous Batching", "Prefix Caching", "INT8", "TP"],
                    "use_case": "Batch processing, API serving"
                },
                {
                    "id": "rlhf_efficient",
                    "name": "Efficient RLHF Training",
                    "description": "Optimize RLHF training pipeline",
                    "techniques": ["Frozen Reference", "vLLM Generation", "Reward Batching"],
                    "use_case": "RLHF/PPO/DPO training"
                },
                {
                    "id": "distributed_large",
                    "name": "Large-Scale Distributed",
                    "description": "Optimize for 100+ GPU training",
                    "techniques": ["3D Parallelism", "Gradient Compression", "Async AllReduce"],
                    "use_case": "Training on large GPU clusters"
                }
            ]
        }
    
    # =========================================================================
    # DISTRIBUTED TRAINING CLUSTER ANALYSIS
    # =========================================================================
    
    def analyze_cluster_topology(self, params: dict) -> dict:
        """
        Analyze cluster topology for distributed training using LLM.
        
        Rules provide context, LLM provides the analysis.
        """
        engine = self._get_llm_engine()
        if not engine:
            return dict(self.LLM_SETUP_ERROR)
        
        # Gather context (rules as guidance)
        gpu_topology = self.get_gpu_topology()
        nvlink_status = self.get_nvlink_status()
        
        num_nodes = params.get("num_nodes", 1)
        gpus_per_node = params.get("gpus_per_node", 8)
        network_type = params.get("network_type", "infiniband")
        network_bandwidth_gbps = params.get("network_bandwidth_gbps", 400)
        
        total_gpus = num_nodes * gpus_per_node
        intra_node_bw = nvlink_status.get("total_bandwidth_gb_s", 900)
        inter_node_bw = network_bandwidth_gbps / 8  # Gbps to GB/s
        bw_ratio = intra_node_bw / inter_node_bw if inter_node_bw > 0 else float('inf')
        
        # Rule-based heuristics as CONTEXT for LLM
        heuristic_guidance = {
            "high_bw_ratio": bw_ratio > 10,
            "suggested_tp": min(gpus_per_node, 8) if bw_ratio > 10 else min(gpus_per_node // 2, 4),
            "suggested_pp": 1 if bw_ratio > 10 else 2,
            "suggested_dp": num_nodes if bw_ratio > 10 else total_gpus // 8,
            "guidance": "High NVLink/network ratio favors TP within node" if bw_ratio > 10 else "Consider hybrid parallelism"
        }
        
        # Build rich context for LLM
        cluster_context = {
            "cluster_config": {
                "num_nodes": num_nodes,
                "gpus_per_node": gpus_per_node,
                "total_gpus": total_gpus,
                "network_type": network_type,
                "network_bandwidth_gbps": network_bandwidth_gbps
            },
            "bandwidth_analysis": {
                "intra_node_gb_s": intra_node_bw,
                "inter_node_gb_s": inter_node_bw,
                "ratio": round(bw_ratio, 2)
            },
            "gpu_topology": gpu_topology,
            "nvlink_status": nvlink_status,
            "heuristic_guidance": heuristic_guidance
        }
        
        # LLM generates the actual analysis
        llm_analysis = engine.ask(
            f"""Analyze this {total_gpus}-GPU distributed training cluster and provide:
1. Optimal parallelism strategy (TP/PP/DP/CP)
2. Communication optimization recommendations
3. Potential bottlenecks and mitigations
4. NCCL environment variable recommendations

Context: {json.dumps(cluster_context, indent=2)}""",
            context=cluster_context
        )
        
        return {
            "llm_powered": True,
            "topology": cluster_context["cluster_config"],
            "bandwidth": cluster_context["bandwidth_analysis"],
            "gpu_topology": gpu_topology,
            "nvlink_status": nvlink_status,
            "heuristic_guidance": heuristic_guidance,
            "analysis": llm_analysis
        }
    
    def recommend_distributed_strategy(self, params: dict) -> dict:
        """
        Recommend distributed training strategy based on model and cluster.
        """
        model_params_b = params.get("model_params_b", 70)
        num_gpus = params.get("num_gpus", 8)
        gpu_memory_gb = params.get("gpu_memory_gb", 80)
        sequence_length = params.get("sequence_length", 4096)
        batch_size = params.get("batch_size", 8)
        
        # Calculate memory requirements
        params_memory_gb = model_params_b * 2  # BF16
        optimizer_memory_gb = model_params_b * 8  # Adam states
        activation_memory_gb = model_params_b * batch_size * sequence_length / 1e9 * 0.1
        
        total_memory_gb = params_memory_gb + optimizer_memory_gb + activation_memory_gb
        memory_per_gpu = total_memory_gb / num_gpus
        
        # Determine strategy
        strategies = []
        
        if memory_per_gpu > gpu_memory_gb:
            strategies.append({
                "name": "FSDP with CPU Offload",
                "reason": "Model too large for GPU memory even with sharding",
                "config": {
                    "sharding_strategy": "FULL_SHARD",
                    "cpu_offload": True,
                    "activation_checkpointing": True
                }
            })
        elif memory_per_gpu > gpu_memory_gb * 0.8:
            strategies.append({
                "name": "FSDP Full Shard",
                "reason": "Tight memory fit requires full sharding",
                "config": {
                    "sharding_strategy": "FULL_SHARD",
                    "activation_checkpointing": True
                }
            })
        elif memory_per_gpu > gpu_memory_gb * 0.5:
            strategies.append({
                "name": "FSDP Shard Grad Op",
                "reason": "Moderate memory pressure allows partial sharding",
                "config": {
                    "sharding_strategy": "SHARD_GRAD_OP",
                    "activation_checkpointing": False
                }
            })
        else:
            strategies.append({
                "name": "DDP",
                "reason": "Sufficient memory for standard data parallelism",
                "config": {
                    "strategy": "ddp",
                    "find_unused_parameters": False
                }
            })
        
        # Add TP/PP recommendations for large models
        if model_params_b > 30:
            strategies.append({
                "name": "Tensor + Pipeline Parallelism",
                "reason": "Large model benefits from model parallelism",
                "config": {
                    "tensor_parallel": min(8, num_gpus),
                    "pipeline_parallel": max(1, num_gpus // 8),
                    "micro_batches": 4
                }
            })
        
        # Use LLM for detailed strategy
        llm_strategy = None
        advisor = self._get_llm_advisor()
        if advisor:
            try:
                context = SystemContext(
                    model_params_b=model_params_b,
                    gpu_count=num_gpus,
                    gpu_memory_gb=gpu_memory_gb,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    is_training=True
                )
                request = OptimizationRequest(
                    context=context,
                    goal=OptimizationGoal.THROUGHPUT,
                    specific_questions=["What is the optimal distributed training strategy?"]
                )
                advice = advisor.get_advice(request)
                llm_strategy = {
                    "parallelism": advice.parallelism_changes,
                    "memory_optimizations": advice.memory_optimizations,
                    "launch_command": advice.launch_command
                }
            except Exception:
                pass
        
        return {
            "model_params_b": model_params_b,
            "num_gpus": num_gpus,
            "memory_analysis": {
                "params_memory_gb": round(params_memory_gb, 1),
                "optimizer_memory_gb": round(optimizer_memory_gb, 1),
                "activation_memory_gb": round(activation_memory_gb, 1),
                "total_memory_gb": round(total_memory_gb, 1),
                "memory_per_gpu_gb": round(memory_per_gpu, 1),
                "gpu_memory_gb": gpu_memory_gb
            },
            "recommended_strategies": strategies,
            "llm_strategy": llm_strategy
        }
    
    def analyze_scaling_efficiency(self, params: dict) -> dict:
        """Analyze scaling efficiency for distributed training."""
        # This would analyze actual training metrics
        baseline_throughput = params.get("baseline_throughput", 1000)  # tokens/s
        scaled_throughput = params.get("scaled_throughput", 7500)  # tokens/s
        num_gpus = params.get("num_gpus", 8)
        
        ideal_throughput = baseline_throughput * num_gpus
        efficiency = scaled_throughput / ideal_throughput if ideal_throughput > 0 else 0
        
        # Identify bottlenecks
        bottlenecks = []
        if efficiency < 0.7:
            bottlenecks.append({
                "type": "communication",
                "severity": "high",
                "description": "Significant communication overhead detected"
            })
        if efficiency < 0.85 and efficiency >= 0.7:
            bottlenecks.append({
                "type": "load_imbalance",
                "severity": "medium",
                "description": "Possible load imbalance across GPUs"
            })
        
        return {
            "baseline_throughput": baseline_throughput,
            "scaled_throughput": scaled_throughput,
            "ideal_throughput": ideal_throughput,
            "num_gpus": num_gpus,
            "efficiency": round(efficiency, 3),
            "efficiency_percent": round(efficiency * 100, 1),
            "grade": "A" if efficiency >= 0.9 else "B" if efficiency >= 0.8 else "C" if efficiency >= 0.7 else "D",
            "bottlenecks": bottlenecks,
            "recommendations": [
                "Enable gradient compression for AllReduce" if efficiency < 0.8 else None,
                "Increase micro-batch count for pipeline parallelism" if efficiency < 0.85 else None,
                "Consider async gradient updates" if efficiency < 0.75 else None
            ]
        }
    
    def analyze_communication_bottlenecks(self, params: dict) -> dict:
        """Analyze communication patterns for bottlenecks."""
        # In a real implementation, this would analyze NCCL traces
        return {
            "collectives": [
                {
                    "name": "AllReduce",
                    "time_ms": 45.2,
                    "percentage": 35.5,
                    "optimization": "Consider gradient compression or local SGD"
                },
                {
                    "name": "AllGather",
                    "time_ms": 28.1,
                    "percentage": 22.1,
                    "optimization": "Overlap with compute using async operations"
                },
                {
                    "name": "ReduceScatter",
                    "time_ms": 15.3,
                    "percentage": 12.0,
                    "optimization": "Already efficient, no action needed"
                }
            ],
            "recommendations": [
                {
                    "priority": "high",
                    "action": "Enable NCCL_ALGO=Ring for better bandwidth utilization",
                    "expected_improvement": "10-15% communication speedup"
                },
                {
                    "priority": "medium",
                    "action": "Set NCCL_BUFFSIZE=8388608 for larger transfers",
                    "expected_improvement": "5-10% for large AllReduce"
                }
            ],
            "environment_variables": {
                "NCCL_ALGO": "Ring",
                "NCCL_BUFFSIZE": "8388608",
                "NCCL_NTHREADS": "512",
                "NCCL_NSOCKS_PERTHREAD": "4"
            }
        }
    
    # =========================================================================
    # RL/RLHF OPTIMIZATION
    # =========================================================================
    
    def optimize_rlhf(self, params: dict) -> dict:
        """
        Optimize RLHF training setup using LLM.
        
        Rules provide context, LLM provides the recommendations.
        """
        engine = self._get_llm_engine()
        if not engine:
            return dict(self.LLM_SETUP_ERROR)
        
        algorithm = params.get("algorithm", "ppo")
        model_params_b = params.get("model_params_b", 70)
        num_gpus = params.get("num_gpus", 8)
        
        # Rule-based context for LLM
        memory_estimates = {
            "actor": f"{model_params_b * 2:.1f} GB (BF16 + gradients)",
            "critic": f"{model_params_b * 0.2:.1f} GB",
            "reference": f"{model_params_b:.1f} GB (frozen FP16)",
            "reward": f"{model_params_b * 0.1:.1f} GB",
            "total_estimated": f"{model_params_b * 3.3:.1f} GB"
        }
        
        # Known optimization patterns as context
        known_patterns = {
            "ppo": [
                "Frozen reference model saves ~25% memory",
                "vLLM for generation gives 3-5x speedup",
                "Reward batching for 2x reward computation speedup",
                "Async KL computation hides latency"
            ],
            "dpo": [
                "Reference-free DPO saves ~50% memory",
                "Chunked loss computation reduces activation memory by ~30%"
            ],
            "orpo": [
                "No reference model needed (implicit)",
                "Odds ratio computation is memory efficient"
            ]
        }
        
        # LLM generates the actual recommendations
        rlhf_context = {
            "algorithm": algorithm,
            "model_params_b": model_params_b,
            "num_gpus": num_gpus,
            "memory_estimates": memory_estimates,
            "known_optimization_patterns": known_patterns.get(algorithm, []),
            "hardware": self.get_hardware_capabilities()
        }
        
        analysis = engine.analyze_rlhf(
            model_config={"params_b": model_params_b, "algorithm": algorithm},
            algorithm=algorithm,
            actor_info={"params_b": model_params_b},
            critic_info={"params_b": model_params_b * 0.1},
            reference_info={"params_b": model_params_b, "frozen": True},
            reward_info={"params_b": model_params_b * 0.1},
            performance_data={"num_gpus": num_gpus},
            memory_usage=memory_estimates
        )
        
        return {
            "llm_powered": True,
            "algorithm": algorithm,
            "model_params_b": model_params_b,
            "num_gpus": num_gpus,
            "memory_estimates": memory_estimates,
            "known_patterns": known_patterns.get(algorithm, []),
            "analysis": analysis
        }
    
    def optimize_rl_algorithm(self, params: dict) -> dict:
        """Optimize specific RL algorithm (PPO, SAC, etc.)."""
        algorithm = params.get("algorithm", "ppo")
        
        configs = {
            "ppo": {
                "name": "Proximal Policy Optimization",
                "optimizations": [
                    "Vectorized environments (8-16 per GPU)",
                    "GAE with λ=0.95",
                    "Mini-batch size 2048-8192",
                    "Mixed precision training",
                    "Gradient accumulation for large batches"
                ],
                "config": {
                    "learning_rate": 3e-4,
                    "clip_range": 0.2,
                    "n_epochs": 10,
                    "batch_size": 64,
                    "n_steps": 2048,
                    "gae_lambda": 0.95,
                    "gamma": 0.99
                }
            },
            "sac": {
                "name": "Soft Actor-Critic",
                "optimizations": [
                    "Efficient replay buffer (prioritized)",
                    "Async environment sampling",
                    "Polyak averaging τ=0.005",
                    "GPU-batched sampling"
                ],
                "config": {
                    "learning_rate": 3e-4,
                    "buffer_size": 1000000,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "train_freq": 1
                }
            }
        }
        
        return configs.get(algorithm, {"error": f"Unknown algorithm: {algorithm}"})
    
    # =========================================================================
    # VLLM / INFERENCE OPTIMIZATION
    # =========================================================================
    
    def optimize_vllm(self, params: dict) -> dict:
        """
        Optimize vLLM configuration for inference serving.
        """
        model = params.get("model", "llama-70b")
        num_gpus = params.get("num_gpus", 8)
        max_tokens = params.get("max_tokens", 4096)
        target_throughput = params.get("target_throughput", None)
        target_latency_ms = params.get("target_latency_ms", None)
        
        # Determine optimal configuration
        config = {
            "tensor_parallel_size": min(num_gpus, 8),
            "max_num_seqs": 256,
            "max_num_batched_tokens": 32768,
            "gpu_memory_utilization": 0.95,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }
        
        # Adjust based on targets
        if target_latency_ms and target_latency_ms < 100:
            config["max_num_seqs"] = 64
            config["max_num_batched_tokens"] = 8192
        elif target_throughput and target_throughput > 10000:
            config["max_num_seqs"] = 512
            config["max_num_batched_tokens"] = 65536
        
        # Generate launch command
        launch_cmd = f"""python -m vllm.entrypoints.openai.api_server \\
    --model {model} \\
    --tensor-parallel-size {config['tensor_parallel_size']} \\
    --max-num-seqs {config['max_num_seqs']} \\
    --max-num-batched-tokens {config['max_num_batched_tokens']} \\
    --gpu-memory-utilization {config['gpu_memory_utilization']} \\
    {'--enable-prefix-caching' if config['enable_prefix_caching'] else ''} \\
    {'--enable-chunked-prefill' if config['enable_chunked_prefill'] else ''}"""
        
        return {
            "model": model,
            "num_gpus": num_gpus,
            "config": config,
            "launch_command": launch_cmd,
            "optimizations": [
                {
                    "name": "Prefix Caching",
                    "enabled": config["enable_prefix_caching"],
                    "benefit": "Reuse KV cache for common prefixes, 2-5x speedup for shared prompts"
                },
                {
                    "name": "Chunked Prefill",
                    "enabled": config["enable_chunked_prefill"],
                    "benefit": "Better latency for long prompts, reduces TTFT"
                },
                {
                    "name": "Continuous Batching",
                    "enabled": True,
                    "benefit": "Dynamic batching for optimal GPU utilization"
                }
            ],
            "metrics_to_monitor": [
                "tokens_per_second",
                "time_to_first_token_ms",
                "inter_token_latency_ms",
                "gpu_utilization_percent",
                "kv_cache_utilization_percent"
            ]
        }
    
    def optimize_inference(self, params: dict) -> dict:
        """
        General inference optimization recommendations.
        """
        model_params_b = params.get("model_params_b", 70)
        batch_size = params.get("batch_size", 1)
        sequence_length = params.get("sequence_length", 4096)
        latency_target_ms = params.get("latency_target_ms", 100)
        
        optimizations = []
        
        # Quantization recommendations
        if model_params_b > 30:
            optimizations.append({
                "name": "FP8 Quantization",
                "priority": "high",
                "speedup": "1.5-2x",
                "accuracy_impact": "Minimal (<0.1% degradation)",
                "how_to_enable": "Use Transformer Engine or vLLM with --quantization fp8"
            })
        
        if model_params_b > 7:
            optimizations.append({
                "name": "INT8 Weight-Only Quantization",
                "priority": "medium",
                "speedup": "1.3-1.5x",
                "memory_savings": "50%",
                "how_to_enable": "Use AWQ or GPTQ quantization"
            })
        
        # Attention optimizations
        optimizations.append({
            "name": "Flash Attention 2/3",
            "priority": "high",
            "speedup": "2-4x for attention",
            "memory_savings": "O(N) instead of O(N²)",
            "how_to_enable": "pip install flash-attn && model.use_flash_attention_2()"
        })
        
        # Speculative decoding
        if batch_size <= 4 and latency_target_ms < 50:
            optimizations.append({
                "name": "Speculative Decoding",
                "priority": "high",
                "speedup": "2-3x for autoregressive generation",
                "how_to_enable": "Use draft model with 10% size of main model"
            })
        
        # CUDA Graphs
        optimizations.append({
            "name": "CUDA Graphs",
            "priority": "medium",
            "speedup": "1.2-1.5x by reducing kernel launch overhead",
            "how_to_enable": "torch.cuda.make_graphed_callables() or vLLM --enforce-eager=False"
        })
        
        return {
            "model_params_b": model_params_b,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "latency_target_ms": latency_target_ms,
            "optimizations": optimizations,
            "recommended_stack": [
                "vLLM or TensorRT-LLM for serving",
                "FP8 quantization for compute",
                "Flash Attention for memory efficiency",
                "Continuous batching for throughput"
            ]
        }
    
    def analyze_speculative_decoding(self, params: dict) -> dict:
        """Analyze speculative decoding setup."""
        main_model = params.get("main_model", "llama-70b")
        draft_model = params.get("draft_model", "llama-7b")
        acceptance_rate = params.get("acceptance_rate", 0.7)
        k = params.get("speculation_length", 5)
        
        # Calculate expected speedup
        # Speedup = k / (1 + (1-α)k) where α is acceptance rate
        expected_speedup = k / (1 + (1 - acceptance_rate) * k)
        
        return {
            "main_model": main_model,
            "draft_model": draft_model,
            "speculation_length": k,
            "acceptance_rate": acceptance_rate,
            "expected_speedup": round(expected_speedup, 2),
            "recommendations": [
                f"Current k={k} with α={acceptance_rate} gives {expected_speedup:.2f}x speedup",
                "Try k=4-8 for optimal balance" if k < 4 or k > 8 else "k value is optimal",
                "Consider training draft model on target distribution to improve α" if acceptance_rate < 0.8 else "Acceptance rate is good"
            ],
            "optimal_k": min(8, max(4, int(1 / (1 - acceptance_rate)))) if acceptance_rate < 1 else 8
        }
    
    # =========================================================================
    # COMPOUND OPTIMIZATION STRATEGIES
    # =========================================================================
    
    def analyze_compound_optimizations(self, params: dict) -> dict:
        """
        Analyze compound optimization strategies that work well together.
        """
        model_params_b = params.get("model_params_b", 70)
        is_training = params.get("is_training", True)
        num_gpus = params.get("num_gpus", 8)
        
        strategies = []
        
        if is_training:
            strategies.append({
                "name": "Memory-Efficient Training Stack",
                "techniques": [
                    "FSDP with SHARD_GRAD_OP",
                    "Activation Checkpointing",
                    "BF16 Mixed Precision",
                    "Flash Attention 2"
                ],
                "combined_effect": "Train 2-3x larger models in same memory",
                "implementation_order": [
                    "1. Enable BF16 mixed precision",
                    "2. Add Flash Attention",
                    "3. Enable FSDP",
                    "4. Add activation checkpointing if needed"
                ],
                "compatibility": "All techniques are compatible and additive"
            })
            
            strategies.append({
                "name": "Maximum Throughput Stack",
                "techniques": [
                    "torch.compile with max-autotune",
                    "CUDA Graphs",
                    "Tensor Parallelism",
                    "Gradient Accumulation"
                ],
                "combined_effect": "2-4x throughput improvement",
                "implementation_order": [
                    "1. Enable torch.compile",
                    "2. Add CUDA Graphs for static shapes",
                    "3. Configure TP for large models",
                    "4. Tune gradient accumulation steps"
                ],
                "compatibility": "torch.compile may conflict with some dynamic operations"
            })
        else:
            strategies.append({
                "name": "Low-Latency Inference Stack",
                "techniques": [
                    "FP8 Quantization",
                    "Flash Attention",
                    "Speculative Decoding",
                    "CUDA Graphs"
                ],
                "combined_effect": "3-5x latency reduction",
                "implementation_order": [
                    "1. Enable Flash Attention",
                    "2. Apply FP8 quantization",
                    "3. Add CUDA Graphs",
                    "4. Implement speculative decoding"
                ],
                "compatibility": "All compatible for batch_size=1"
            })
            
            strategies.append({
                "name": "High-Throughput Serving Stack",
                "techniques": [
                    "Continuous Batching",
                    "Prefix Caching",
                    "INT8 Quantization",
                    "Tensor Parallelism"
                ],
                "combined_effect": "10x+ throughput vs naive serving",
                "implementation_order": [
                    "1. Deploy with vLLM/TGI",
                    "2. Enable prefix caching",
                    "3. Apply quantization",
                    "4. Scale with TP"
                ],
                "compatibility": "All compatible, vLLM handles automatically"
            })
        
        # Use LLM for custom compound strategy
        llm_strategy = None
        advisor = self._get_llm_advisor()
        if advisor:
            try:
                context = SystemContext(
                    model_params_b=model_params_b,
                    gpu_count=num_gpus,
                    is_training=is_training
                )
                request = OptimizationRequest(
                    context=context,
                    goal=OptimizationGoal.THROUGHPUT,
                    specific_questions=["What compound optimization strategies work best together?"]
                )
                advice = advisor.get_advice(request)
                llm_strategy = advice.compound_strategies
            except Exception:
                pass
        
        return {
            "model_params_b": model_params_b,
            "is_training": is_training,
            "num_gpus": num_gpus,
            "strategies": strategies,
            "llm_strategy": llm_strategy,
            "warning": "Always benchmark compound strategies - interactions can vary by workload"
        }
    
    def recommend_optimization_stack(self, params: dict) -> dict:
        """
        Recommend a complete optimization stack for the given workload.
        """
        # Get comprehensive LLM recommendation
        advisor = self._get_llm_advisor()
        
        if advisor:
            try:
                hw_caps = self.get_hardware_capabilities()
                gpu_info = hw_caps.get("gpu", {})
                
                context = SystemContext(
                    gpu_name=gpu_info.get("name", "H100"),
                    gpu_memory_gb=gpu_info.get("memory_gb", 80),
                    gpu_count=params.get("num_gpus", 8),
                    model_name=params.get("model", ""),
                    model_params_b=params.get("model_params_b", 70),
                    batch_size=params.get("batch_size", 8),
                    sequence_length=params.get("sequence_length", 4096),
                    is_training=params.get("is_training", True),
                    precision=params.get("precision", "bf16"),
                )
                
                request = OptimizationRequest(
                    context=context,
                    goal=OptimizationGoal.THROUGHPUT,
                    specific_questions=[
                        "What is the optimal stack of optimizations for this workload?",
                        "What order should I apply these optimizations?",
                        "What are the expected compound effects?"
                    ]
                )
                
                advice = advisor.get_advice(request)
                
                return {
                    "llm_powered": True,
                    "summary": advice.summary,
                    "recommended_stack": advice.priority_recommendations,
                    "compound_strategies": advice.compound_strategies,
                    "parallelism": advice.parallelism_changes,
                    "memory_optimizations": advice.memory_optimizations,
                    "launch_command": advice.launch_command,
                    "environment_variables": advice.environment_variables,
                    "expected_improvements": advice.expected_improvements,
                    "warnings": advice.warnings
                }
            except Exception as e:
                pass
        
        # Fallback to rule-based
        return self.analyze_compound_optimizations(params)
    
    # =========================================================================
    # BATCH SIZE OPTIMIZER + HUGGINGFACE INTEGRATION
    # =========================================================================
    
    # Cache for HuggingFace API responses (to avoid rate limiting)
    _hf_cache: Dict[str, Any] = {}
    _hf_cache_time: Dict[str, float] = {}
    _HF_CACHE_TTL = 300  # 5 minutes cache
    
    def _hf_api_request(self, endpoint: str, cache_key: str = None) -> dict:
        """Make a request to the HuggingFace API with caching."""
        import urllib.request
        import urllib.error
        
        cache_key = cache_key or endpoint
        now = time.time()
        
        # Check cache
        if cache_key in self._hf_cache:
            if now - self._hf_cache_time.get(cache_key, 0) < self._HF_CACHE_TTL:
                return self._hf_cache[cache_key]
        
        try:
            url = f"https://huggingface.co/api/{endpoint}"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'GPU-Performance-Dashboard/1.0'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                self._hf_cache[cache_key] = data
                self._hf_cache_time[cache_key] = now
                return data
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_param_count(self, model_info: dict) -> int:
        """Extract parameter count from HuggingFace model info."""
        # Try safetensors metadata first (most accurate)
        safetensors = model_info.get("safetensors", {})
        if safetensors and "total" in safetensors:
            return safetensors["total"]
        
        # Try config (transformer models)
        config = model_info.get("config", {})
        if config:
            # Common patterns for different model types
            if "num_parameters" in config:
                return config["num_parameters"]
            # Try to calculate from model dimensions
            hidden_size = config.get("hidden_size", config.get("d_model", 0))
            num_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", 0)))
            vocab_size = config.get("vocab_size", 0)
            if hidden_size and num_layers:
                # Rough transformer param estimate: 12 * L * H^2 + V * H
                return int(12 * num_layers * hidden_size * hidden_size + vocab_size * hidden_size)
        
        # Try model card or tags for hints
        tags = model_info.get("tags", [])
        model_id = model_info.get("id", model_info.get("modelId", "")).lower()
        
        # Parse size from model name/tags
        size_patterns = [
            (r'(\d+(?:\.\d+)?)[bB]', 1e9),   # 7B, 70B, 1.5B
            (r'(\d+(?:\.\d+)?)[mM]', 1e6),   # 124M, 350M
            (r'(\d+)[xX](\d+)[bB]', lambda m: int(m.group(1)) * int(m.group(2)) * 1e9),  # 8x7B
        ]
        
        for pattern, multiplier in size_patterns:
            for text in [model_id] + tags:
                match = re.search(pattern, text)
                if match:
                    if callable(multiplier):
                        return int(multiplier(match))
                    return int(float(match.group(1)) * multiplier)
        
        return 0  # Unknown
    
    def get_hf_trending_models(self) -> dict:
        """Get top 10 trending models from HuggingFace."""
        try:
            # Fetch most downloaded models - focus on LLMs (text-generation pipeline)
            models = self._hf_api_request(
                "models?sort=downloads&direction=-1&limit=50&pipeline_tag=text-generation",
                "trending_models"
            )
            
            if "error" in models:
                # Return fallback models if API fails
                return self._get_fallback_trending_models()
            
            # Process and filter to top 10 interesting models
            processed = []
            seen_families = set()
            
            for m in models:
                model_id = m.get("id", m.get("modelId", ""))
                if not model_id:
                    continue
                
                # Skip duplicates from same family (e.g., multiple Llama versions)
                family = model_id.split("/")[-1].split("-")[0].lower()
                if family in seen_families and len(processed) >= 5:
                    continue
                
                # Get detailed info for param count
                detail = self._hf_api_request(f"models/{model_id}", f"model_{model_id}")
                param_count = self._extract_param_count(detail) if detail else 0
                
                # If we couldn't get params from API, try to parse from name
                if param_count == 0:
                    param_count = self._extract_param_count({"id": model_id, "tags": m.get("tags", [])})
                
                if param_count == 0:
                    continue  # Skip if we can't determine params
                
                downloads = m.get("downloads", 0)
                likes = m.get("likes", 0)
                
                processed.append({
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "author": model_id.split("/")[0] if "/" in model_id else "unknown",
                    "params": param_count,
                    "params_display": self._format_params(param_count),
                    "downloads": downloads,
                    "downloads_display": self._format_number(downloads),
                    "likes": likes,
                    "pipeline_tag": m.get("pipeline_tag", "unknown"),
                    "tags": m.get("tags", [])[:5],
                    "hf_url": f"https://huggingface.co/{model_id}",
                })
                seen_families.add(family)
                
                if len(processed) >= 10:
                    break
            
            return {
                "success": True,
                "models": processed,
                "source": "huggingface_api",
                "cached": "trending_models" in self._hf_cache,
            }
            
        except Exception as e:
            return self._get_fallback_trending_models()
    
    def _get_fallback_trending_models(self) -> dict:
        """Return fallback models when HuggingFace API is unavailable."""
        # Current popular models as of late 2024 (updated regularly)
        fallback = [
            {"id": "meta-llama/Llama-3.2-3B", "params": 3_000_000_000},
            {"id": "meta-llama/Llama-3.1-8B", "params": 8_000_000_000},
            {"id": "meta-llama/Llama-3.1-70B", "params": 70_000_000_000},
            {"id": "mistralai/Mistral-7B-v0.3", "params": 7_000_000_000},
            {"id": "mistralai/Mixtral-8x7B-v0.1", "params": 46_700_000_000},
            {"id": "google/gemma-2-9b", "params": 9_000_000_000},
            {"id": "google/gemma-2-27b", "params": 27_000_000_000},
            {"id": "Qwen/Qwen2.5-7B", "params": 7_000_000_000},
            {"id": "Qwen/Qwen2.5-72B", "params": 72_000_000_000},
            {"id": "microsoft/phi-3-medium-4k-instruct", "params": 14_000_000_000},
        ]
        
        return {
            "success": True,
            "models": [{
                "id": m["id"],
                "name": m["id"].split("/")[-1],
                "author": m["id"].split("/")[0],
                "params": m["params"],
                "params_display": self._format_params(m["params"]),
                "downloads": 0,
                "downloads_display": "N/A",
                "likes": 0,
                "pipeline_tag": "text-generation",
                "tags": [],
                "hf_url": f"https://huggingface.co/{m['id']}",
            } for m in fallback],
            "source": "fallback",
            "cached": False,
        }
    
    def search_hf_models(self, query: str) -> dict:
        """Search HuggingFace models by name."""
        if not query or len(query) < 2:
            return {"success": False, "error": "Query too short", "models": []}
        
        try:
            models = self._hf_api_request(
                f"models?search={query}&sort=downloads&direction=-1&limit=20",
                f"search_{query}"
            )
            
            if "error" in models:
                return {"success": False, "error": models["error"], "models": []}
            
            processed = []
            for m in models[:15]:
                model_id = m.get("id", m.get("modelId", ""))
                if not model_id:
                    continue
                
                # Quick param extraction without detailed API call for speed
                param_count = self._extract_param_count({"id": model_id, "tags": m.get("tags", [])})
                
                processed.append({
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "author": model_id.split("/")[0] if "/" in model_id else "unknown",
                    "params": param_count,
                    "params_display": self._format_params(param_count) if param_count else "Unknown",
                    "downloads": m.get("downloads", 0),
                    "downloads_display": self._format_number(m.get("downloads", 0)),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag", "unknown"),
                    "hf_url": f"https://huggingface.co/{model_id}",
                })
            
            return {"success": True, "models": processed, "query": query}
            
        except Exception as e:
            return {"success": False, "error": str(e), "models": []}
    
    def get_hf_model_info(self, model_id: str) -> dict:
        """Get detailed info for a specific HuggingFace model."""
        try:
            detail = self._hf_api_request(f"models/{model_id}", f"model_{model_id}")
            
            if "error" in detail:
                return {"success": False, "error": detail["error"]}
            
            param_count = self._extract_param_count(detail)
            if param_count == 0:
                param_count = self._extract_param_count({"id": model_id, "tags": detail.get("tags", [])})
            
            return {
                "success": True,
                "model": {
                    "id": model_id,
                    "name": model_id.split("/")[-1],
                    "author": detail.get("author", model_id.split("/")[0] if "/" in model_id else "unknown"),
                    "params": param_count,
                    "params_display": self._format_params(param_count) if param_count else "Unknown",
                    "downloads": detail.get("downloads", 0),
                    "likes": detail.get("likes", 0),
                    "pipeline_tag": detail.get("pipeline_tag", "unknown"),
                    "tags": detail.get("tags", [])[:10],
                    "library_name": detail.get("library_name", "unknown"),
                    "hf_url": f"https://huggingface.co/{model_id}",
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _format_params(self, params: int) -> str:
        """Format parameter count for display."""
        if params >= 1e12:
            return f"{params/1e12:.1f}T"
        elif params >= 1e9:
            return f"{params/1e9:.1f}B"
        elif params >= 1e6:
            return f"{params/1e6:.0f}M"
        elif params >= 1e3:
            return f"{params/1e3:.0f}K"
        return str(params)
    
    def _format_number(self, num: int) -> str:
        """Format large numbers for display."""
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        return str(num)
    
    def _calculate_batch_for_params(self, params: int, vram_free_gb: float, precision: str = "fp16") -> dict:
        """Calculate batch size recommendations for a model with given param count."""
        # Memory multipliers by precision
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }
        
        bytes_per_param = precision_bytes.get(precision, 2)
        
        # Model weight memory
        weight_mem_gb = (params * bytes_per_param) / (1024 ** 3)
        
        # For training, need ~3x weights (weights + gradients + optimizer states)
        # For inference, just weights + KV cache overhead
        inference_mem_gb = weight_mem_gb * 1.2  # 20% overhead for KV cache, buffers
        training_mem_gb = weight_mem_gb * 3.5   # weights + grads + adam states + activations
        
        # Available for batch processing
        available_inference = max(0, vram_free_gb - inference_mem_gb - 1)  # 1GB buffer
        available_training = max(0, vram_free_gb - training_mem_gb - 2)    # 2GB buffer
        
        # Estimate memory per sample (rough heuristic based on model size)
        # Larger models need more activation memory per sample
        if params > 50e9:
            mem_per_sample_mb = 2000  # 70B+ models
        elif params > 10e9:
            mem_per_sample_mb = 800   # 10-50B models
        elif params > 3e9:
            mem_per_sample_mb = 400   # 3-10B models
        elif params > 1e9:
            mem_per_sample_mb = 200   # 1-3B models
        else:
            mem_per_sample_mb = 100   # <1B models
        
        # Calculate max batch sizes
        max_batch_inference = int(available_inference * 1024 / mem_per_sample_mb) if available_inference > 0 else 0
        max_batch_training = int(available_training * 1024 / (mem_per_sample_mb * 2)) if available_training > 0 else 0
        
        # Round to power of 2 for recommended
        def round_to_power_of_2(n):
            if n <= 0:
                return 0
            result = 1
            for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                if bs <= n:
                    result = bs
            return result
        
        recommended_inference = round_to_power_of_2(max_batch_inference)
        recommended_training = round_to_power_of_2(max_batch_training)
        
        # Calculate utilization
        util_inference = (recommended_inference * mem_per_sample_mb / 1024) / vram_free_gb * 100 if vram_free_gb > 0 else 0
        util_training = (recommended_training * mem_per_sample_mb * 2 / 1024) / vram_free_gb * 100 if vram_free_gb > 0 else 0
        
        can_run = weight_mem_gb < vram_free_gb
        
        return {
            "can_run": can_run,
            "weight_memory_gb": round(weight_mem_gb, 2),
            "precision": precision,
            "inference": {
                "recommended_batch_size": recommended_inference,
                "max_batch_size": max(1, max_batch_inference) if can_run else 0,
                "utilization_pct": round(min(util_inference, 100), 1),
            },
            "training": {
                "recommended_batch_size": recommended_training,
                "max_batch_size": max(0, max_batch_training),
                "utilization_pct": round(min(util_training, 100), 1),
            },
            "memory_per_sample_mb": mem_per_sample_mb,
            "suggestions": self._get_optimization_suggestions(params, vram_free_gb, precision, can_run),
        }
    
    def _get_optimization_suggestions(self, params: int, vram_gb: float, precision: str, can_run: bool) -> list:
        """Get optimization suggestions for running the model."""
        suggestions = []
        
        if not can_run:
            suggestions.append({
                "type": "critical",
                "text": "Model too large for available VRAM",
                "solutions": [
                    "Use quantization (INT8, INT4)",
                    "Use model parallelism across multiple GPUs",
                    "Try a smaller model variant",
                ]
            })
        
        if precision == "fp32":
            suggestions.append({
                "type": "optimization",
                "text": "Switch to FP16/BF16 for 2x memory savings",
                "benefit": "Double your batch size or fit larger models"
            })
        
        if params > 7e9 and precision in ["fp32", "fp16", "bf16"]:
            suggestions.append({
                "type": "optimization", 
                "text": "Consider INT8 quantization for 4x memory savings",
                "benefit": "Minimal accuracy loss, major memory reduction"
            })
        
        if params > 20e9:
            suggestions.append({
                "type": "advanced",
                "text": "Use Flash Attention 2 for efficient attention",
                "benefit": "Reduce memory usage and improve speed"
            })
            suggestions.append({
                "type": "advanced",
                "text": "Enable gradient checkpointing for training",
                "benefit": "Trade compute for memory to train larger batches"
            })
        
        return suggestions
    
    def calculate_batch_for_model(self, params: dict) -> dict:
        """Calculate batch size for a user-specified model."""
        gpu_info = self.get_gpu_info()
        vram_total_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        vram_used_gb = (gpu_info.get("memory_used", 0) or 0) / 1024
        vram_free_gb = vram_total_gb - vram_used_gb
        
        model_id = params.get("model_id", "")
        custom_params = params.get("params", 0)  # Allow custom param count
        precision = params.get("precision", "fp16")
        
        # If model_id provided, fetch from HuggingFace
        if model_id and not custom_params:
            model_info = self.get_hf_model_info(model_id)
            if model_info.get("success"):
                custom_params = model_info["model"]["params"]
                model_name = model_info["model"]["name"]
            else:
                return {"success": False, "error": f"Could not fetch model info: {model_info.get('error')}"}
        else:
            model_name = params.get("name", "Custom Model")
        
        if not custom_params or custom_params <= 0:
            return {"success": False, "error": "Could not determine parameter count. Please specify manually."}
        
        batch_info = self._calculate_batch_for_params(custom_params, vram_free_gb, precision)
        
        return {
            "success": True,
            "gpu": gpu_info.get("name", "Unknown"),
            "vram_total_gb": round(vram_total_gb, 1),
            "vram_free_gb": round(vram_free_gb, 1),
            "model": {
                "id": model_id,
                "name": model_name,
                "params": custom_params,
                "params_display": self._format_params(custom_params),
            },
            **batch_info
        }
    
    def get_batch_size_recommendations(self) -> dict:
        """Get batch size optimization recommendations using HuggingFace trending models."""
        gpu_info = self.get_gpu_info()
        
        # Get available VRAM
        vram_total_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        vram_used_gb = (gpu_info.get("memory_used", 0) or 0) / 1024
        vram_free_gb = vram_total_gb - vram_used_gb
        
        # Get trending models from HuggingFace
        trending = self.get_hf_trending_models()
        models = trending.get("models", [])[:10]
        
        recommendations = {
            "gpu": gpu_info.get("name", "Unknown"),
            "vram_total_gb": round(vram_total_gb, 1),
            "vram_free_gb": round(vram_free_gb, 1),
            "scenarios": [],
            "batch_size_curve": [],
            "source": trending.get("source", "unknown"),
        }
        
        for model in models:
            param_count = model.get("params", 0)
            if param_count <= 0:
                continue
            
            batch_info = self._calculate_batch_for_params(param_count, vram_free_gb, "fp16")
            
            recommendations["scenarios"].append({
                "model": model["name"],
                "model_id": model["id"],
                "params_millions": round(param_count / 1e6),
                "params_display": model.get("params_display", self._format_params(param_count)),
                "hf_url": model.get("hf_url", ""),
                "downloads": model.get("downloads_display", ""),
                "max_batch_size": batch_info["inference"]["max_batch_size"],
                "recommended_batch_size": batch_info["inference"]["recommended_batch_size"],
                "memory_per_sample_mb": batch_info["memory_per_sample_mb"],
                "utilization_pct": batch_info["inference"]["utilization_pct"],
                "can_run": batch_info["can_run"],
                "weight_memory_gb": batch_info["weight_memory_gb"],
            })
        
        # Generate batch size vs throughput curve (theoretical)
        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            # Throughput increases with batch size but with diminishing returns
            relative_throughput = min(bs, 64) / 64 * 100  # Normalize to 100%
            memory_usage = bs * 100  # MB per sample (rough)
            
            recommendations["batch_size_curve"].append({
                "batch_size": bs,
                "relative_throughput": round(relative_throughput, 1),
                "memory_mb": memory_usage,
            })
        
        return recommendations
    
    def get_models_that_fit(self) -> dict:
        """Find the best/largest models that fit on this GPU."""
        gpu_info = self.get_gpu_info()
        vram_total_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        vram_free_gb = vram_total_gb * 0.9  # Use 90% as available
        
        # Curated list of popular models by category with known param counts
        model_catalog = [
            # Flagship LLMs
            {"id": "meta-llama/Llama-3.1-405B", "params": 405e9, "category": "flagship", "quality": 100},
            {"id": "meta-llama/Llama-3.1-70B-Instruct", "params": 70e9, "category": "flagship", "quality": 95},
            {"id": "Qwen/Qwen2.5-72B-Instruct", "params": 72e9, "category": "flagship", "quality": 94},
            {"id": "mistralai/Mixtral-8x22B-v0.1", "params": 141e9, "category": "flagship", "quality": 93},
            {"id": "deepseek-ai/DeepSeek-V2.5", "params": 236e9, "category": "flagship", "quality": 96},
            # Large models (30-70B)
            {"id": "meta-llama/Llama-3.1-70B", "params": 70e9, "category": "large", "quality": 92},
            {"id": "mistralai/Mixtral-8x7B-v0.1", "params": 46.7e9, "category": "large", "quality": 88},
            {"id": "Qwen/Qwen2.5-32B-Instruct", "params": 32e9, "category": "large", "quality": 87},
            {"id": "microsoft/phi-3-medium-128k-instruct", "params": 14e9, "category": "large", "quality": 85},
            # Medium models (7-30B)
            {"id": "meta-llama/Llama-3.1-8B-Instruct", "params": 8e9, "category": "medium", "quality": 82},
            {"id": "mistralai/Mistral-7B-Instruct-v0.3", "params": 7e9, "category": "medium", "quality": 80},
            {"id": "Qwen/Qwen2.5-7B-Instruct", "params": 7.6e9, "category": "medium", "quality": 81},
            {"id": "google/gemma-2-9b-it", "params": 9e9, "category": "medium", "quality": 83},
            {"id": "microsoft/phi-3-small-128k-instruct", "params": 7e9, "category": "medium", "quality": 79},
            # Small models (<7B)  
            {"id": "meta-llama/Llama-3.2-3B-Instruct", "params": 3e9, "category": "small", "quality": 72},
            {"id": "microsoft/phi-3-mini-4k-instruct", "params": 3.8e9, "category": "small", "quality": 75},
            {"id": "Qwen/Qwen2.5-3B-Instruct", "params": 3e9, "category": "small", "quality": 73},
            {"id": "google/gemma-2-2b-it", "params": 2e9, "category": "small", "quality": 68},
            {"id": "meta-llama/Llama-3.2-1B-Instruct", "params": 1e9, "category": "small", "quality": 60},
            # Code models
            {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "params": 32e9, "category": "code", "quality": 90},
            {"id": "deepseek-ai/deepseek-coder-33b-instruct", "params": 33e9, "category": "code", "quality": 88},
            {"id": "codellama/CodeLlama-34b-Instruct-hf", "params": 34e9, "category": "code", "quality": 85},
            {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "params": 7e9, "category": "code", "quality": 78},
        ]
        
        results = {"fits": [], "almost_fits": [], "gpu": gpu_info.get("name", "Unknown"), "vram_gb": round(vram_free_gb, 1)}
        
        for model in model_catalog:
            for precision, multiplier in [("fp16", 2), ("int8", 1), ("int4", 0.5)]:
                mem_gb = (model["params"] * multiplier) / (1024**3)
                
                if mem_gb <= vram_free_gb * 0.85:  # Fits with 15% headroom
                    batch_info = self._calculate_batch_for_params(int(model["params"]), vram_free_gb, precision)
                    results["fits"].append({
                        "model_id": model["id"],
                        "name": model["id"].split("/")[-1],
                        "params": model["params"],
                        "params_display": self._format_params(int(model["params"])),
                        "category": model["category"],
                        "quality_score": model["quality"],
                        "precision": precision,
                        "memory_gb": round(mem_gb, 1),
                        "headroom_gb": round(vram_free_gb - mem_gb, 1),
                        "max_batch_size": batch_info["inference"]["max_batch_size"],
                        "recommended_batch_size": batch_info["inference"]["recommended_batch_size"],
                        "hf_url": f"https://huggingface.co/{model['id']}",
                    })
                    break  # Only show best precision that fits
                elif mem_gb <= vram_free_gb * 1.2:  # Almost fits (within 20%)
                    results["almost_fits"].append({
                        "model_id": model["id"],
                        "name": model["id"].split("/")[-1],
                        "params_display": self._format_params(int(model["params"])),
                        "category": model["category"],
                        "precision": precision,
                        "memory_gb": round(mem_gb, 1),
                        "over_by_gb": round(mem_gb - vram_free_gb, 1),
                    })
                    break
        
        # Sort by quality score
        results["fits"].sort(key=lambda x: -x["quality_score"])
        return results
    
    def get_quantization_comparison(self, params: dict) -> dict:
        """Compare batch sizes across different quantization levels."""
        gpu_info = self.get_gpu_info()
        vram_total_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        vram_free_gb = vram_total_gb - ((gpu_info.get("memory_used", 0) or 0) / 1024)
        
        model_params = params.get("params", 7e9)  # Default 7B
        model_name = params.get("name", "Model")
        
        precisions = ["fp32", "fp16", "bf16", "int8", "int4"]
        comparison = []
        
        for precision in precisions:
            batch_info = self._calculate_batch_for_params(int(model_params), vram_free_gb, precision)
            comparison.append({
                "precision": precision.upper(),
                "weight_memory_gb": batch_info["weight_memory_gb"],
                "can_run": batch_info["can_run"],
                "inference_batch": batch_info["inference"]["recommended_batch_size"],
                "inference_max": batch_info["inference"]["max_batch_size"],
                "training_batch": batch_info["training"]["recommended_batch_size"],
                "training_max": batch_info["training"]["max_batch_size"],
            })
        
        return {
            "model_name": model_name,
            "params": model_params,
            "params_display": self._format_params(int(model_params)),
            "gpu": gpu_info.get("name", "Unknown"),
            "vram_free_gb": round(vram_free_gb, 1),
            "comparison": comparison,
        }
    
    def get_multi_gpu_scaling(self, params: dict) -> dict:
        """Calculate scaling across multiple GPUs with tensor parallelism."""
        gpu_info = self.get_gpu_info()
        single_gpu_vram = (gpu_info.get("memory_total", 0) or 0) / 1024
        
        model_params = params.get("params", 70e9)  # Default 70B
        model_name = params.get("name", "Model")
        precision = params.get("precision", "fp16")
        
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}.get(precision, 2)
        model_mem_gb = (model_params * bytes_per_param) / (1024**3)
        
        gpu_configs = [1, 2, 4, 8]
        scaling = []
        
        for num_gpus in gpu_configs:
            total_vram = single_gpu_vram * num_gpus
            # With tensor parallelism, model is split across GPUs
            per_gpu_model_mem = model_mem_gb / num_gpus
            per_gpu_available = single_gpu_vram - per_gpu_model_mem - 2  # 2GB buffer per GPU
            
            can_run = per_gpu_model_mem < (single_gpu_vram * 0.85)
            
            # Batch size scales with available memory (roughly linear with TP)
            if can_run and per_gpu_available > 0:
                # Memory per sample increases slightly with TP due to communication buffers
                mem_per_sample = 400 * (1 + 0.1 * (num_gpus - 1))  # 10% overhead per additional GPU
                max_batch = int(per_gpu_available * 1024 / mem_per_sample)
                recommended = 1
                for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    if bs <= max_batch:
                        recommended = bs
            else:
                max_batch = 0
                recommended = 0
            
            # Throughput scaling (not perfectly linear due to communication overhead)
            # Typically 0.85-0.95x linear scaling with NVLink, 0.7-0.85x with PCIe
            throughput_efficiency = 0.9 ** (num_gpus - 1) if num_gpus > 1 else 1.0
            relative_throughput = num_gpus * throughput_efficiency
            
            scaling.append({
                "num_gpus": num_gpus,
                "total_vram_gb": round(total_vram, 1),
                "per_gpu_model_mem_gb": round(per_gpu_model_mem, 1),
                "can_run": can_run,
                "recommended_batch_size": recommended,
                "max_batch_size": max_batch,
                "relative_throughput": round(relative_throughput, 2),
                "throughput_efficiency": f"{throughput_efficiency * 100:.0f}%",
            })
        
        return {
            "model_name": model_name,
            "params_display": self._format_params(int(model_params)),
            "precision": precision.upper(),
            "single_gpu": gpu_info.get("name", "Unknown"),
            "model_memory_gb": round(model_mem_gb, 1),
            "scaling": scaling,
        }
    
    def get_cloud_cost_estimate(self, params: dict) -> dict:
        """Estimate cloud GPU costs for running models."""
        model_params = params.get("params", 7e9)
        batch_size = params.get("batch_size", 32)
        tokens_per_request = params.get("tokens", 512)
        requests_per_day = params.get("requests_per_day", 10000)
        precision = params.get("precision", "fp16")
        
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}.get(precision, 2)
        model_mem_gb = (model_params * bytes_per_param) / (1024**3)
        
        # GPU pricing (approximate $/hour as of late 2024)
        gpu_pricing = [
            {"name": "NVIDIA T4", "vram": 16, "price": 0.35, "tflops": 65},
            {"name": "NVIDIA A10G", "vram": 24, "price": 1.00, "tflops": 125},
            {"name": "NVIDIA L4", "vram": 24, "price": 0.80, "tflops": 121},
            {"name": "NVIDIA A100 40GB", "vram": 40, "price": 3.50, "tflops": 312},
            {"name": "NVIDIA A100 80GB", "vram": 80, "price": 5.00, "tflops": 312},
            {"name": "NVIDIA H100 80GB", "vram": 80, "price": 8.50, "tflops": 990},
            {"name": "NVIDIA H100 SXM", "vram": 80, "price": 12.00, "tflops": 1980},
        ]
        
        estimates = []
        for gpu in gpu_pricing:
            if model_mem_gb > gpu["vram"] * 0.85:
                # Need multiple GPUs
                num_gpus = int(model_mem_gb / (gpu["vram"] * 0.7)) + 1
                if num_gpus > 8:
                    continue  # Skip if needs more than 8 GPUs
            else:
                num_gpus = 1
            
            # Estimate tokens/second based on GPU performance
            base_tokens_per_sec = (gpu["tflops"] / model_params * 1e9) * 50  # Rough heuristic
            tokens_per_sec = base_tokens_per_sec * batch_size * 0.7  # Batch efficiency
            
            # Time to process daily requests
            total_tokens = requests_per_day * tokens_per_request
            hours_needed = (total_tokens / tokens_per_sec) / 3600
            
            # Cost calculation
            hourly_cost = gpu["price"] * num_gpus
            daily_cost = hourly_cost * max(hours_needed, 1)  # Minimum 1 hour
            monthly_cost = daily_cost * 30
            
            # If running 24/7
            monthly_24_7 = hourly_cost * 24 * 30
            
            estimates.append({
                "gpu": gpu["name"],
                "num_gpus": num_gpus,
                "vram_total": gpu["vram"] * num_gpus,
                "hourly_cost": round(hourly_cost, 2),
                "estimated_tokens_per_sec": round(tokens_per_sec, 0),
                "hours_for_daily_load": round(hours_needed, 2),
                "daily_cost": round(daily_cost, 2),
                "monthly_cost": round(monthly_cost, 2),
                "monthly_24_7_cost": round(monthly_24_7, 2),
            })
        
        return {
            "model_params": self._format_params(int(model_params)),
            "precision": precision.upper(),
            "batch_size": batch_size,
            "tokens_per_request": tokens_per_request,
            "requests_per_day": requests_per_day,
            "estimates": estimates,
        }
    
    def get_throughput_estimate(self, params: dict) -> dict:
        """Estimate token throughput for different configurations."""
        gpu_info = self.get_gpu_info()
        model_params = params.get("params", 7e9)
        precision = params.get("precision", "fp16")
        
        # GPU TFLOPS estimates by architecture
        gpu_name = gpu_info.get("name", "").lower()
        if "h100" in gpu_name:
            tflops = 990 if "sxm" in gpu_name else 700
        elif "h200" in gpu_name:
            tflops = 990
        elif "b100" in gpu_name or "b200" in gpu_name:
            tflops = 1800  # Blackwell
        elif "a100" in gpu_name:
            tflops = 312
        elif "l40" in gpu_name:
            tflops = 181
        elif "4090" in gpu_name:
            tflops = 165
        else:
            tflops = 100  # Conservative default
        
        # Precision multipliers
        precision_mult = {"fp32": 1.0, "fp16": 2.0, "bf16": 2.0, "int8": 4.0, "int4": 8.0}.get(precision, 2.0)
        effective_tflops = tflops * precision_mult
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        throughput_data = []
        
        for bs in batch_sizes:
            # Tokens per second estimation (simplified model)
            # Higher batch sizes = better throughput, but with diminishing returns
            efficiency = min(0.95, 0.3 + 0.1 * min(bs, 8))  # 30% base + 10% per batch up to 8
            tokens_per_sec = (effective_tflops * 1e12 / model_params) * efficiency * bs
            
            # Latency estimation (first token)
            first_token_latency_ms = (model_params / (effective_tflops * 1e12)) * 1000 * (1 + 0.1 * bs)
            
            throughput_data.append({
                "batch_size": bs,
                "tokens_per_sec": round(tokens_per_sec, 0),
                "first_token_latency_ms": round(first_token_latency_ms, 1),
                "requests_per_minute": round(tokens_per_sec / 100 * 60, 0),  # Assume 100 tokens avg
            })
        
        return {
            "gpu": gpu_info.get("name", "Unknown"),
            "model_params": self._format_params(int(model_params)),
            "precision": precision.upper(),
            "gpu_tflops": tflops,
            "effective_tflops": round(effective_tflops, 0),
            "throughput_data": throughput_data,
        }
    
    def generate_deploy_config(self, params: dict) -> dict:
        """Generate deployment configuration for popular inference servers."""
        model_id = params.get("model_id", "meta-llama/Llama-3.1-8B-Instruct")
        model_params = params.get("params", 8e9)
        precision = params.get("precision", "fp16")
        num_gpus = params.get("num_gpus", 1)
        max_batch_size = params.get("max_batch_size", 32)
        
        model_name = model_id.split("/")[-1]
        
        # vLLM config
        vllm_config = f"""# vLLM Deployment for {model_name}
# Run with: python -m vllm.entrypoints.openai.api_server --config vllm_config.yaml

model: "{model_id}"
tensor-parallel-size: {num_gpus}
dtype: "{precision}"
max-model-len: 4096
gpu-memory-utilization: 0.9
max-num-batched-tokens: {max_batch_size * 512}
max-num-seqs: {max_batch_size}
trust-remote-code: true

# Optional optimizations
enable-prefix-caching: true
enable-chunked-prefill: true
"""

        # Text Generation Inference (TGI) config  
        tgi_command = f"""# Text Generation Inference (TGI) Deployment
# Docker command for {model_name}

docker run --gpus all -p 8080:80 \\
    -v ~/.cache/huggingface:/data \\
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id {model_id} \\
    --num-shard {num_gpus} \\
    --dtype {precision} \\
    --max-batch-prefill-tokens {max_batch_size * 512} \\
    --max-batch-total-tokens {max_batch_size * 2048} \\
    --max-concurrent-requests {max_batch_size * 4}
"""

        # Ollama Modelfile (for smaller models)
        ollama_modelfile = f"""# Ollama Modelfile for {model_name}
# Create with: ollama create {model_name.lower()} -f Modelfile

FROM {model_id}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_batch {min(max_batch_size, 512)}

SYSTEM You are a helpful AI assistant.
"""

        # Python script for transformers
        transformers_script = f'''# Python script for {model_name} inference
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{model_id}"
device = "cuda"

# Load model with optimizations
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.{"float16" if precision == "fp16" else "bfloat16" if precision == "bf16" else "float32"},
    device_map="auto",
    trust_remote_code=True,
)

# Generate
def generate(prompt: str, max_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
response = generate("Explain quantum computing in simple terms:")
print(response)
'''

        return {
            "model_id": model_id,
            "model_name": model_name,
            "params_display": self._format_params(int(model_params)),
            "precision": precision,
            "num_gpus": num_gpus,
            "configs": {
                "vllm": vllm_config,
                "tgi": tgi_command,
                "ollama": ollama_modelfile,
                "transformers": transformers_script,
            }
        }
    
    def get_finetuning_estimate(self, params: dict) -> dict:
        """Estimate memory and compute for fine-tuning."""
        model_params = params.get("params", 7e9)
        model_name = params.get("name", "Model")
        dataset_size = params.get("dataset_size", 10000)  # Number of examples
        seq_length = params.get("seq_length", 512)
        
        gpu_info = self.get_gpu_info()
        vram_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        
        estimates = []
        
        # Full fine-tuning (FP16)
        full_ft_mem = (model_params * 2 + model_params * 2 + model_params * 8) / (1024**3)  # weights + grads + optimizer
        full_ft_mem += (model_params * 0.1 * seq_length / 512) / (1024**3)  # Activations estimate
        estimates.append({
            "method": "Full Fine-tuning (FP16)",
            "memory_gb": round(full_ft_mem, 1),
            "fits": full_ft_mem < vram_gb * 0.9,
            "trainable_params": self._format_params(int(model_params)),
            "trainable_pct": "100%",
            "notes": "Highest quality, but requires most memory",
        })
        
        # LoRA
        lora_params = model_params * 0.01  # ~1% trainable
        lora_mem = (model_params * 2 + lora_params * 2 + lora_params * 8) / (1024**3)
        lora_mem += (model_params * 0.05 * seq_length / 512) / (1024**3)
        estimates.append({
            "method": "LoRA (r=16, alpha=32)",
            "memory_gb": round(lora_mem, 1),
            "fits": lora_mem < vram_gb * 0.9,
            "trainable_params": self._format_params(int(lora_params)),
            "trainable_pct": "~1%",
            "notes": "Great balance of quality and efficiency",
        })
        
        # QLoRA (INT4 base + FP16 adapters)
        qlora_mem = (model_params * 0.5 + lora_params * 2 + lora_params * 8) / (1024**3)
        qlora_mem += (model_params * 0.03 * seq_length / 512) / (1024**3)
        estimates.append({
            "method": "QLoRA (INT4 + LoRA)",
            "memory_gb": round(qlora_mem, 1),
            "fits": qlora_mem < vram_gb * 0.9,
            "trainable_params": self._format_params(int(lora_params)),
            "trainable_pct": "~1%",
            "notes": "Most memory efficient, some quality trade-off",
        })
        
        # Training time estimate
        tokens_total = dataset_size * seq_length * 3  # 3 epochs
        # Rough estimate: tokens/sec based on GPU and model size
        if model_params > 30e9:
            tokens_per_sec = 500
        elif model_params > 7e9:
            tokens_per_sec = 2000
        else:
            tokens_per_sec = 5000
        
        training_hours = tokens_total / tokens_per_sec / 3600
        
        return {
            "model_name": model_name,
            "params_display": self._format_params(int(model_params)),
            "gpu": gpu_info.get("name", "Unknown"),
            "vram_gb": round(vram_gb, 1),
            "dataset_size": dataset_size,
            "seq_length": seq_length,
            "estimates": estimates,
            "training_time_estimate": {
                "epochs": 3,
                "total_tokens": tokens_total,
                "estimated_hours": round(training_hours, 1),
            }
        }
    
    def get_llm_optimization_advice(self, params: dict) -> dict:
        """Get LLM-powered optimization recommendations (dynamic, not hardcoded)."""
        model_id = params.get("model_id", "")
        model_params = params.get("params", 0)
        model_name = params.get("name", "Model")
        use_case = params.get("use_case", "inference")
        constraints = params.get("constraints", {})
        
        gpu_info = self.get_gpu_info()
        vram_gb = (gpu_info.get("memory_total", 0) or 0) / 1024
        gpu_name = gpu_info.get("name", "Unknown")
        arch = self._detect_gpu_arch(gpu_name)
        
        context = {
            "model": {"id": model_id, "name": model_name, "params_billions": model_params / 1e9 if model_params else 0},
            "hardware": {"gpu": gpu_name, "vram_gb": round(vram_gb, 1), "architecture": arch},
            "use_case": use_case,
            "constraints": constraints,
        }
        engine = self._get_llm_engine()
        if not engine:
            return self._get_fallback_advice(context, "LLM engine unavailable")
        
        question = f"Provide optimization recommendations for {model_name} on {gpu_name} for {use_case}."
        llm_response = engine.ask(question, context)
        return {
            "success": True,
            "source": "llm",
            "model": model_name,
            "use_case": use_case,
            "hardware": context["hardware"],
            "recommendations": llm_response,
        }
    
    def _detect_gpu_arch(self, gpu_name: str) -> str:
        name_lower = gpu_name.lower()
        if "b100" in name_lower or "b200" in name_lower: return "blackwell"
        elif "h100" in name_lower or "h200" in name_lower: return "hopper"
        elif "a100" in name_lower: return "ampere"
        return "unknown"
    
    def _get_fallback_advice(self, context: dict, error: str) -> dict:
        recommendations = ["Use FP8/BF16 precision", "Enable Flash Attention", "Use continuous batching for inference"]
        return {"success": True, "source": "rule_based", "model": context["model"]["name"], "recommendations": "\n".join(recommendations), "note": f"LLM unavailable: {error}"}
    
    def calculate_compound_optimizations(self, params: dict) -> dict:
        """Calculate compound effects of stacking multiple optimizations (LLM-powered)."""
        model_params = params.get("params", 7e9)
        selected_opts = params.get("optimizations", ["bf16", "flash_attn"])
        engine = self._get_llm_engine()
        if engine:
            prompt = "Estimate compound effects of stacked optimizations."
            context = {
                "model_params": model_params,
                "optimizations": selected_opts,
            }
            try:
                response = engine.ask(prompt, context)
                return {"success": True, "llm_powered": True, "analysis": response}
            except Exception as exc:
                self._llm_init_error = str(exc)
                # fall through to heuristic path
        
        optimization_db = {
            "fp16": {"speedup": 1.8, "memory": 0.5}, "bf16": {"speedup": 1.7, "memory": 0.5},
            "fp8": {"speedup": 2.0, "memory": 0.75}, "int8": {"speedup": 1.5, "memory": 0.75},
            "flash_attn": {"speedup": 2.5, "memory": 0.8}, "continuous_batch": {"speedup": 2.5, "memory": 1.0},
            "cuda_graphs": {"speedup": 1.3, "memory": 1.0}, "torch_compile": {"speedup": 1.5, "memory": 1.0},
        }
        
        total_speedup, total_memory = 1.0, 1.0
        applied = []
        for opt_id in selected_opts:
            if opt_id in optimization_db:
                opt = optimization_db[opt_id]
                total_speedup *= opt["speedup"]
                total_memory *= opt["memory"]
                applied.append({"id": opt_id, "speedup": opt["speedup"]})
        
        base_mem = (model_params * 2) / (1024**3)
        return {
            "success": True,
            "llm_powered": False,
            "fallback_reason": "LLM engine unavailable",
            "total_speedup": round(total_speedup, 2),
            "memory_reduction": f"{(1-total_memory)*100:.0f}%",
            "base_memory_gb": round(base_mem, 1),
            "optimized_memory_gb": round(base_mem * total_memory, 1),
            "applied": applied
        }
    
    # =========================================================================
    # WEBHOOK SYSTEM
    # =========================================================================
    
    _webhooks_file = CODE_ROOT / "dashboard" / "webhooks.json"
    
    def _load_webhooks(self) -> list:
        """Load webhooks from persistent storage."""
        try:
            if self._webhooks_file.exists():
                return json.loads(self._webhooks_file.read_text())
        except Exception:
            pass
        return []
    
    def _save_webhooks(self, webhooks: list) -> None:
        """Save webhooks to persistent storage."""
        try:
            self._webhooks_file.write_text(json.dumps(webhooks, indent=2))
        except Exception as e:
            print(f"Failed to save webhooks: {e}")
    
    def get_webhooks(self) -> dict:
        """Get all configured webhooks."""
        return {"webhooks": self._load_webhooks()}
    
    def save_webhooks(self, params: dict) -> dict:
        """Save webhooks to persistent storage."""
        webhooks = params.get("webhooks", [])
        self._save_webhooks(webhooks)
        return {"success": True, "count": len(webhooks)}
    
    def test_webhook(self, params: dict) -> dict:
        """Test a webhook by sending a test notification."""
        url = params.get("url", "")
        platform = params.get("platform", "slack")
        name = params.get("name", "Test Webhook")
        
        if not url:
            return {"success": False, "error": "No URL provided"}
        
        # Build test payload based on platform
        if platform == "slack":
            payload = {
                "text": f"🧪 Test notification from AI Performance Dashboard\n*Webhook:* {name}\n*Status:* Connection successful!"
            }
        elif platform == "discord":
            payload = {
                "content": f"🧪 Test notification from AI Performance Dashboard\n**Webhook:** {name}\n**Status:** Connection successful!"
            }
        elif platform == "teams":
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "00f5d4",
                "summary": "AI Performance Dashboard Test",
                "sections": [{
                    "activityTitle": "🧪 Test Notification",
                    "facts": [
                        {"name": "Webhook", "value": name},
                        {"name": "Status", "value": "Connection successful!"}
                    ],
                }]
            }
        else:
            payload = {"message": f"Test notification from AI Performance Dashboard - {name}"}
        
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return {
                    "success": True,
                    "status_code": response.status,
                    "message": "Webhook test successful!"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send test notification"
            }
    
    def send_webhook_notification(self, params: dict) -> dict:
        """Send a notification to a webhook."""
        url = params.get("url", "")
        message_type = params.get("message_type", "custom")
        platform = params.get("type", "slack")
        custom_message = params.get("message", "")
        
        if not url:
            return {"success": False, "error": "No URL provided"}
        
        # Build payload based on message type
        if message_type == "summary":
            # Get benchmark summary
            data = self.load_benchmark_data()
            summary = data.get("summary", {})
            
            summary_text = (
                f"📊 *Performance Summary*\n"
                f"• Total Benchmarks: {summary.get('total', 0)}\n"
                f"• Succeeded: {summary.get('succeeded', 0)}\n"
                f"• Failed: {summary.get('failed', 0)}\n"
                f"• Avg Speedup: {summary.get('avg_speedup', 0):.2f}x\n"
                f"• Max Speedup: {summary.get('max_speedup', 0):.2f}x"
            )
            message = summary_text
        elif message_type == "regression":
            message = "⚠️ Performance regression detected! Check the dashboard for details."
        elif message_type == "optimization_complete":
            message = "✅ Optimization job completed successfully!"
        else:
            message = custom_message or "Notification from AI Performance Dashboard"
        
        # Format for platform
        if platform == "slack":
            payload = {"text": message}
        elif platform == "discord":
            payload = {"content": message.replace("*", "**")}
        elif platform == "teams":
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "00f5d4",
                "summary": "AI Performance Dashboard",
                "sections": [{"activityTitle": message}]
            }
        else:
            payload = {"message": message}
        
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return {
                    "success": True,
                    "status_code": response.status,
                    "message": "Notification sent successfully!"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send notification"
            }
    
    # =========================================================================
    # THEME SYSTEM
    # =========================================================================
    
    def run_benchmark(self, params: dict) -> dict:
        """Run a specific benchmark and return results."""
        chapter = params.get('chapter', '')
        name = params.get('name', '')
        run_baseline = params.get('run_baseline', True)
        run_optimized = params.get('run_optimized', True)
        
        if not chapter or not name:
            return {"success": False, "error": "Missing chapter or name"}
        
        try:
            # Build the command against the new Typer CLI
            cmd = [
                sys.executable,
                "-m",
                "cli.aisp",
                "bench",
                "run",
                "--targets",
                f"{chapter}:{name}",
                "--format",
                "json",
            ]
            # The new CLI runs baseline + optimized by default; skip flags are not exposed yet.
            # Future: add --skip-baseline/--skip-optimized support upstream if needed.

            # Run the benchmark with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(CODE_ROOT)
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or "Benchmark failed",
                    "stdout": result.stdout
                }
            
            # Try to parse JSON output
            try:
                output = json.loads(result.stdout)
                return {
                    "success": True,
                    "baseline_ms": output.get('baseline_time_ms'),
                    "optimized_ms": output.get('optimized_time_ms'),
                    "speedup": output.get('speedup'),
                    "output": output
                }
            except json.JSONDecodeError:
                # If not JSON, just return success with raw output
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "message": "Benchmark completed (non-JSON output)"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Benchmark timed out after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_code_diff(self, chapter: str, name: str) -> dict:
        """Get baseline and optimized code for a benchmark."""
        import urllib.parse
        name = urllib.parse.unquote(name)
        
        chapter_dir = CODE_ROOT / chapter
        if not chapter_dir.exists():
            return {"error": f"Chapter directory not found: {chapter}"}

        code_pair = find_code_pair(chapter_dir, name)
        baseline_code = code_pair.get("baseline_code")
        optimized_code = code_pair.get("optimized_code")

        if not baseline_code and not optimized_code:
            return {
                "error": "Code files not found",
                "hint": f"Looking in {chapter_dir}",
                "baseline": None,
                "optimized": None
            }

        diff_summary = {}
        if baseline_code and optimized_code:
            diff_summary = summarize_diff(baseline_code, optimized_code)

        return {
            "baseline": baseline_code,
            "optimized": optimized_code,
            "chapter": chapter,
            "name": name,
            **diff_summary
        }

    # =========================================================================
    # GPU CONTROL METHODS
    # =========================================================================
    
    def get_gpu_control_state(self) -> dict:
        """Get current GPU clock and power state."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=clocks.gr,clocks.max.gr,clocks.mem,clocks.max.mem,clocks.sm,power.draw,power.limit,power.max_limit,persistence_mode',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 9:
                    return {
                        "clocks": {
                            "graphics": int(parts[0]) if parts[0].strip() else None,
                            "graphics_max": int(parts[1]) if parts[1].strip() else None,
                            "memory": int(parts[2]) if parts[2].strip() else None,
                            "memory_max": int(parts[3]) if parts[3].strip() else None,
                            "sm": int(parts[4]) if parts[4].strip() else None,
                        },
                        "power": {
                            "current": float(parts[5]) if parts[5].strip() else None,
                            "limit": float(parts[6]) if parts[6].strip() else None,
                            "max_limit": float(parts[7]) if parts[7].strip() else None,
                        },
                        "persistence_mode": parts[8].strip().lower() == 'enabled',
                        "clocks_locked": False  # Would need separate check
                    }
        except Exception as e:
            pass
        return {"error": "Could not query GPU state"}
    
    def get_gpu_topology(self) -> dict:
        """Get multi-GPU topology information."""
        return super().get_gpu_topology()

    def get_nvlink_status(self) -> dict:
        """Get detailed NVLink status."""
        return super().get_nvlink_status()
    
    def get_cuda_environment(self) -> dict:
        """Get CUDA environment information."""
        env_info = {
            "cuda_version": os.environ.get('CUDA_VERSION', 'Unknown'),
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
            "torch_compile_debug": os.environ.get('TORCH_COMPILE_DEBUG'),
            "cublas_workspace": os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        }
        
        # Try to get PyTorch info
        try:
            result = subprocess.run(
                [sys.executable, '-c', '''
import torch
import json
info = {
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cudnn_enabled": torch.backends.cudnn.enabled,
    "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
    "flash_attention": hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
    "deterministic": torch.are_deterministic_algorithms_enabled() if hasattr(torch, 'are_deterministic_algorithms_enabled') else False,
}
print(json.dumps(info))
'''],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                pytorch_info = json.loads(result.stdout.strip())
                env_info.update(pytorch_info)
        except Exception as e:
            pass
        
        return env_info
    
    def set_gpu_power_limit(self, params: dict) -> dict:
        """Set GPU power limit (requires root)."""
        power_limit = params.get('power_limit')
        if not power_limit:
            return {"success": False, "error": "No power limit specified"}
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '-pl', str(power_limit)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return {"success": True, "power_limit": power_limit}
            return {"success": False, "error": result.stderr or "Failed to set power limit"}
        except Exception as e:
            return {"success": False, "error": str(e), "command": f"sudo nvidia-smi -pl {power_limit}"}
    
    def set_gpu_clock_pin(self, params: dict) -> dict:
        """Pin GPU clocks to max (requires root)."""
        pin = params.get('pin', True)
        
        try:
            if pin:
                # First get max clocks
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=clocks.max.gr', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    max_clock = result.stdout.strip()
                    result = subprocess.run(
                        ['nvidia-smi', '-lgc', max_clock],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        return {"success": True, "clocks_locked": True, "clock": max_clock}
            else:
                result = subprocess.run(
                    ['nvidia-smi', '-rgc'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return {"success": True, "clocks_locked": False}
            
            return {"success": False, "error": "Failed to modify clock settings"}
        except Exception as e:
            cmd = "nvidia-smi -lgc MAX" if pin else "nvidia-smi -rgc"
            return {"success": False, "error": str(e), "command": f"sudo {cmd}"}
    
    def set_gpu_persistence(self, params: dict) -> dict:
        """Enable/disable GPU persistence mode (requires root)."""
        enabled = params.get('enabled', True)
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '-pm', '1' if enabled else '0'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return {"success": True, "persistence_mode": enabled}
            return {"success": False, "error": result.stderr or "Failed to set persistence mode"}
        except Exception as e:
            cmd = f"sudo nvidia-smi -pm {'1' if enabled else '0'}"
            return {"success": False, "error": str(e), "command": cmd}
    
    def apply_gpu_preset(self, params: dict) -> dict:
        """Apply a GPU performance preset."""
        preset = params.get('preset', 'balanced')
        
        commands = []
        if preset == 'max':
            commands = [
                'nvidia-smi -pm 1',
                'nvidia-smi -lgc MAX',
                'nvidia-smi -pl MAX'
            ]
        elif preset == 'balanced':
            commands = [
                'nvidia-smi -pm 1',
                'nvidia-smi -rgc'
            ]
        elif preset == 'quiet':
            commands = [
                'nvidia-smi -pm 0',
                'nvidia-smi -rgc',
                'nvidia-smi -pl 200'
            ]
        
        results = []
        all_success = True
        
        for cmd in commands:
            try:
                parts = cmd.split()
                result = subprocess.run(parts, capture_output=True, text=True, timeout=5)
                results.append({"cmd": cmd, "success": result.returncode == 0})
                if result.returncode != 0:
                    all_success = False
            except Exception as e:
                results.append({"cmd": cmd, "success": False, "error": str(e)})
                all_success = False
        
        return {
            "success": all_success,
            "preset": preset,
            "results": results,
            "commands": [f"sudo {cmd}" for cmd in commands]
        }

    def get_available_themes(self) -> dict:
        """Get available UI themes."""
        return {
            "themes": [
                {
                    "id": "dark-purple",
                    "name": "Dark Purple (Default)",
                    "description": "Deep purple accents on dark background",
                    "colors": {
                        "bg_primary": "#0f0f14",
                        "bg_card": "#1a1a24",
                        "accent_primary": "#8854d0",
                        "accent_success": "#22c55e",
                        "accent_warning": "#f59e0b",
                        "accent_danger": "#ef4444",
                    }
                },
                {
                    "id": "dark-blue",
                    "name": "Dark Blue",
                    "description": "Professional blue theme",
                    "colors": {
                        "bg_primary": "#0a0f1a",
                        "bg_card": "#111827",
                        "accent_primary": "#3b82f6",
                        "accent_success": "#10b981",
                        "accent_warning": "#f59e0b",
                        "accent_danger": "#ef4444",
                    }
                },
                {
                    "id": "dark-green",
                    "name": "Matrix Green",
                    "description": "Hacker-style green theme",
                    "colors": {
                        "bg_primary": "#0a0f0a",
                        "bg_card": "#0f1a0f",
                        "accent_primary": "#22c55e",
                        "accent_success": "#4ade80",
                        "accent_warning": "#facc15",
                        "accent_danger": "#f87171",
                    }
                },
                {
                    "id": "light",
                    "name": "Light Mode",
                    "description": "Light background for daytime use",
                    "colors": {
                        "bg_primary": "#f8fafc",
                        "bg_card": "#ffffff",
                        "accent_primary": "#7c3aed",
                        "accent_success": "#16a34a",
                        "accent_warning": "#d97706",
                        "accent_danger": "#dc2626",
                        "text_primary": "#1e293b",
                        "text_secondary": "#475569",
                    }
                },
                {
                    "id": "high-contrast",
                    "name": "High Contrast",
                    "description": "Maximum readability",
                    "colors": {
                        "bg_primary": "#000000",
                        "bg_card": "#1a1a1a",
                        "accent_primary": "#00ffff",
                        "accent_success": "#00ff00",
                        "accent_warning": "#ffff00",
                        "accent_danger": "#ff0000",
                    }
                },
                {
                    "id": "nvidia",
                    "name": "NVIDIA Green",
                    "description": "Official NVIDIA colors",
                    "colors": {
                        "bg_primary": "#1a1a1a",
                        "bg_card": "#2d2d2d",
                        "accent_primary": "#76b900",
                        "accent_success": "#76b900",
                        "accent_warning": "#f5a623",
                        "accent_danger": "#e74c3c",
                    }
                },
            ],
            "current": "dark-purple",
        }
    
    # =========================================================================
    # LIVE OPTIMIZATION STREAMING (SSE)
    # =========================================================================
    
    def stream_optimization_events(self, job_id: str):
        """Stream optimization events using Server-Sent Events (SSE)."""
        global _job_events
        
        if job_id not in _job_events:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found"}).encode())
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        event_queue = _job_events[job_id]
        
        try:
            while True:
                try:
                    # Wait for events with timeout
                    event = event_queue.get(timeout=30)
                    
                    if event is None:
                        # Job completed
                        self.wfile.write(b"event: complete\ndata: {}\n\n")
                        self.wfile.flush()
                        break
                    
                    # Send event
                    event_type = event.get("type", "message")
                    event_data = json.dumps(event)
                    self.wfile.write(f"event: {event_type}\ndata: {event_data}\n\n".encode())
                    self.wfile.flush()
                    
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected
    
    def start_optimization_job(self, params: dict) -> dict:
        """Start a new optimization job with live streaming."""
        global _optimization_jobs, _job_events
        
        job_id = str(uuid.uuid4())[:8]
        target = params.get("target", "")
        
        if not target:
            return {"error": "No target specified. Provide 'target' as chapter:example or chapter"}
        
        # Create event queue for this job
        _job_events[job_id] = queue.Queue()
        
        # Create job record
        job = {
            "id": job_id,
            "target": target,
            "status": "starting",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "llm_analysis": params.get("llm_analysis", True),
            "apply_patches": params.get("apply_patches", True),
            "rebenchmark": params.get("rebenchmark", True),
            "deep_profile": params.get("deep_profile", False),
            "events": [],
        }
        _optimization_jobs[job_id] = job
        
        # Start optimization in background thread
        def run_optimization():
            self._run_optimization_job(job_id, params)
        
        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()
        
        return {
            "job_id": job_id,
            "status": "started",
            "stream_url": f"/api/optimize/stream/{job_id}",
        }
    
    def _run_optimization_job(self, job_id: str, params: dict):
        """Run the optimization job and emit events."""
        global _optimization_jobs, _job_events
        
        job = _optimization_jobs.get(job_id)
        event_queue = _job_events.get(job_id)
        
        if not job or not event_queue:
            return
        
        def emit(event_type: str, data: dict):
            event = {"type": event_type, "timestamp": time.strftime("%H:%M:%S"), **data}
            job["events"].append(event)
            event_queue.put(event)
        
        try:
            emit("status", {"message": "🚀 Starting optimization job...", "status": "running"})
            job["status"] = "running"
            
            target = params.get("target", "")
            llm_analysis = params.get("llm_analysis", True)
            apply_patches = params.get("apply_patches", True)
            rebenchmark = params.get("rebenchmark", True)
            deep_profile = params.get("deep_profile", False)
            
            # Build the bench command via aisp
            cmd = [sys.executable, "-m", "cli.aisp", "bench", "run", "-t", target]
            
            if llm_analysis:
                cmd.append("--llm-analysis")
                emit("info", {"message": "📊 LLM analysis enabled"})
            if apply_patches:
                cmd.append("--apply-llm-patches")
                emit("info", {"message": "🔧 Patch application enabled"})
            if rebenchmark:
                cmd.append("--rebenchmark-llm-patches")
                emit("info", {"message": "⏱️ Rebenchmarking enabled"})
            if deep_profile:
                cmd.extend(["--profile", "deep_dive"])
                emit("info", {"message": "🔬 Deep profiling enabled (nsys/ncu/PyTorch)"})
            
            emit("command", {"message": f"Running: {' '.join(cmd)}"})
            
            # Run the command and stream output
            process = subprocess.Popen(
                cmd,
                cwd=str(CODE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            job["pid"] = process.pid
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if not line:
                    continue
                
                # Categorize output
                event_type = "output"
                if "🧠" in line or "LLM" in line.upper():
                    event_type = "llm"
                elif "📊" in line or "BASELINE" in line or "BENCHMARK" in line:
                    event_type = "benchmark"
                elif "🔧" in line or "PATCH" in line.upper():
                    event_type = "patch"
                elif "🔬" in line or "PROFIL" in line.upper() or "NSYS" in line.upper() or "NCU" in line.upper() or "ROOFLINE" in line.upper():
                    event_type = "profile"
                elif "✅" in line or "SUCCEEDED" in line.upper():
                    event_type = "success"
                elif "❌" in line or "FAILED" in line.upper() or "ERROR" in line.upper():
                    event_type = "error"
                elif "⚡" in line or "SPEEDUP" in line.upper():
                    event_type = "speedup"
                
                emit(event_type, {"message": line})
            
            process.wait()
            
            if process.returncode == 0:
                emit("complete", {"message": "✅ Optimization completed successfully!", "status": "completed"})
                job["status"] = "completed"
            else:
                emit("error", {"message": f"❌ Optimization failed with exit code {process.returncode}", "status": "failed"})
                job["status"] = "failed"
            
        except Exception as e:
            emit("error", {"message": f"❌ Error: {str(e)}", "status": "error"})
            job["status"] = "error"
        finally:
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            event_queue.put(None)  # Signal completion
    
    def stop_optimization_job(self, job_id: str) -> dict:
        """Stop a running optimization job."""
        global _optimization_jobs, _job_events
        
        if not job_id or job_id not in _optimization_jobs:
            return {"error": "Job not found"}
        
        job = _optimization_jobs[job_id]
        
        if job.get("pid"):
            try:
                import signal
                os.kill(job["pid"], signal.SIGTERM)
                job["status"] = "stopped"
                
                if job_id in _job_events:
                    _job_events[job_id].put({"type": "stopped", "message": "Job stopped by user"})
                    _job_events[job_id].put(None)
                
                return {"status": "stopped", "job_id": job_id}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "Job has no active process"}
    
    def list_optimization_jobs(self) -> dict:
        """List all optimization jobs."""
        global _optimization_jobs
        
        jobs = []
        for job_id, job in _optimization_jobs.items():
            jobs.append({
                "id": job_id,
                "target": job.get("target"),
                "status": job.get("status"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "event_count": len(job.get("events", [])),
            })
        
        return {"jobs": jobs, "count": len(jobs)}
    
    # =========================================================================
    # Multi-Metric Analysis Methods
    # =========================================================================
    
    def get_pareto_frontier(self) -> dict:
        """Calculate Pareto-optimal benchmarks across speed and memory dimensions."""
        return self.analyzer.get_pareto_frontier()
    
    def get_tradeoff_analysis(self) -> dict:
        """Analyze speed vs memory trade-offs for all benchmarks."""
        return self.analyzer.get_tradeoff_analysis()
    
    def get_constraint_recommendations(self) -> dict:
        """Provide recommendations based on common constraint scenarios."""
        return self.analyzer.get_constraint_recommendations()
    
    def get_categorized_leaderboards(self) -> dict:
        """Return separate leaderboards for each optimization category."""
        return self.analyzer.get_categorized_leaderboards()
    
    def get_whatif_recommendations(self, params: dict) -> dict:
        """What-If Constraint Solver: Find optimizations matching user constraints."""
        return self.analyzer.get_whatif_recommendations(params)
    
    # =========================================================================
    # ADVANCED SYSTEM ANALYSIS METHODS (NEW!)
    # =========================================================================
    
    def get_cpu_memory_analysis(self) -> dict:
        """Get CPU/memory hierarchy analysis (caches, NUMA, TLB, hugepages)."""
        return advanced_wrappers.cpu_memory_analysis()
    
    def get_system_parameters(self) -> dict:
        """Get kernel/system parameters affecting performance."""
        return advanced_wrappers.system_parameters()
    
    def get_container_limits(self) -> dict:
        """Get container/cgroups limits detection."""
        return advanced_wrappers.container_limits()
    
    def analyze_warp_divergence(self, code: str = "") -> dict:
        """Analyze code for warp divergence patterns."""
        return advanced_wrappers.warp_divergence(code)
    
    def analyze_bank_conflicts(self, stride: int = 1, element_size: int = 4) -> dict:
        """Analyze shared memory bank conflicts."""
        return advanced_wrappers.bank_conflicts(stride, element_size)
    
    def analyze_memory_access(self, stride: int = 1, element_size: int = 4) -> dict:
        """Analyze memory access patterns for coalescing."""
        return advanced_wrappers.memory_access(stride, element_size)
    
    def run_auto_tuning(self, kernel_type: str = "matmul", max_configs: int = 50) -> dict:
        """Run auto-tuning for kernel parameters."""
        return advanced_wrappers.auto_tuning(kernel_type, max_configs)
    
    def get_full_system_analysis(self) -> dict:
        """Get complete system analysis for optimization."""
        return {
            "cpu_memory": advanced_wrappers.cpu_memory_analysis(),
            "system_params": advanced_wrappers.system_parameters(),
            "container": advanced_wrappers.container_limits(),
            "optimizations_available": len(optimization_stack.get_all_optimizations().get("optimizations", [])),
            "playbooks_available": optimization_stack.get_optimization_playbooks().get("count", 0),
            "recommendations": self._generate_comprehensive_recommendations(),
        }
    
    def _generate_comprehensive_recommendations(self) -> list:
        """Generate comprehensive optimization recommendations."""
        recs = []
        
        cpu_mem = advanced_wrappers.cpu_memory_analysis()
        sys_params = advanced_wrappers.system_parameters()
        container = advanced_wrappers.container_limits()

        recs.extend(cpu_mem.get("recommendations", []))
        recs.extend(sys_params.get("recommendations", []))
        recs.extend(container.get("recommendations", []))

        # Add GPU-specific recommendations
        sw_info = self.get_software_info()
        if sw_info.get("compute_capability"):
            cc = sw_info["compute_capability"]
            if cc >= "8.9":
                recs.append("FP8 supported! Use Transformer Engine for 2x throughput.")
            if cc >= "9.0":
                recs.append("Hopper architecture detected! Use TMA and WGMMA for best performance.")
            if cc >= "10.0":
                recs.append("Blackwell architecture detected! Enable FP4 and DSMEM for maximum performance.")
        
        return recs[:10]  # Top 10 recommendations
    
    def predict_hardware_scaling(self, from_gpu: str, to_gpu: str, workload: str) -> dict:
        """Predict performance scaling between GPUs."""
        return advanced_wrappers.predict_hardware_scaling(from_gpu, to_gpu, workload)
    
    def analyze_energy_efficiency(self, gpu: str, power_limit: int = None) -> dict:
        """Analyze GPU energy efficiency."""
        return advanced_wrappers.energy_efficiency(gpu, power_limit)
    
    def estimate_multi_gpu_scaling(self, gpus: int, nvlink: bool, workload: str) -> dict:
        """Estimate multi-GPU scaling efficiency."""
        return advanced_wrappers.multi_gpu_scaling(gpus, nvlink, workload)
    
    def get_optimization_stacking(self) -> dict:
        """Analyze which optimizations can be combined (stacked)."""
        return optimization_stack.get_optimization_stacking(self.analyzer)
    
    def get_all_optimizations(self) -> dict:
        """Get all available optimization techniques."""
        return optimization_stack.get_all_optimizations()
    
    def get_optimization_playbooks(self) -> dict:
        """Get pre-defined optimization playbooks."""
        return optimization_stack.get_optimization_playbooks()
    
    def calculate_compound_optimization(self, optimizations: list) -> dict:
        """Calculate compound effect of multiple optimizations."""
        software_info = self.get_software_info()
        return optimization_stack.calculate_compound_optimization(optimizations, software_info)
    
    def get_optimal_optimization_stack(self, target_speedup: float, max_difficulty: str) -> dict:
        """Find optimal optimization stack for target speedup."""
        software_info = self.get_software_info()
        return optimization_stack.get_optimal_optimization_stack(target_speedup, max_difficulty, software_info)
    
    def calculate_occupancy(self, threads: int, shared: int, registers: int) -> dict:
        """Calculate kernel occupancy."""
        try:
            from core.analysis.advanced_analysis import KernelAnalyzer
            
            # Get GPU specs
            software_info = self.get_software_info()
            sm_count = software_info.get("sm_count", 132)
            max_threads_per_sm = software_info.get("max_threads_per_sm", 2048)
            max_registers_per_sm = software_info.get("registers_per_sm", 65536)
            max_shared_per_sm = (software_info.get("shared_mem_per_sm_kb") or 228) * 1024
            
            analyzer = KernelAnalyzer()
            result = analyzer.estimate_from_code(
                threads_per_block=threads,
                shared_memory_bytes=shared,
                registers_per_thread=registers,
                sm_count=sm_count,
                max_threads_per_sm=max_threads_per_sm,
                max_registers_per_sm=max_registers_per_sm,
                max_shared_per_sm=max_shared_per_sm,
            )
            
            return {
                "success": True,
                "gpu_specs": {
                    "sm_count": sm_count,
                    "max_threads_per_sm": max_threads_per_sm,
                    "max_registers_per_sm": max_registers_per_sm,
                    "max_shared_per_sm_bytes": max_shared_per_sm,
                },
                **result,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_power_efficiency(self) -> dict:
        """Analyze power efficiency (ops/watt) of benchmarks."""
        return self.analyzer.get_power_efficiency()
    
    def get_scaling_analysis(self) -> dict:
        """Analyze how optimizations scale with workload size."""
        return self.analyzer.get_scaling_analysis()
    
    def get_cost_analysis(self, gpu: str = None, custom_rate: float = None) -> dict:
        """Calculate cost impact ($/token, $/hour savings).
        
        Args:
            gpu: GPU type ('B200', 'H100', 'A100', 'L40S', 'A10G', 'T4')
            custom_rate: Custom hourly rate in $/hr
        """
        return self.analyzer.get_cost_analysis(gpu=gpu, custom_rate=custom_rate)
    
    def run_warmup_audit(self, check_recommended: bool = False) -> dict:
        """Run the warmup audit script and return results."""
        return run_warmup_audit(CODE_ROOT, check_recommended)
    
    # =========================================================================
    # NEW LLM-POWERED ANALYSIS METHODS (using llm_advisor module)
    # =========================================================================
    
    def llm_analyze_bottlenecks(self) -> dict:
        """Use LLM to analyze bottlenecks from profiling data."""
        try:
            from core.analysis.llm_advisor import get_advisor, OptimizationContext
            
            # Gather context from existing analysis
            kernel_data = self.detect_bottlenecks()
            hw_info = self.get_hardware_capabilities()
            gpu_info = self.get_gpu_info()
            
            advisor = get_advisor()
            
            # Build optimization context
            context = OptimizationContext(
                gpu_name=gpu_info.get("gpu_name", "Unknown"),
                gpu_memory_gb=gpu_info.get("memory_total_gb", 0),
                compute_capability=tuple(hw_info.get("gpu", {}).get("compute_capability", [0, 0])),
                num_gpus=gpu_info.get("gpu_count", 1),
                nvlink_available=hw_info.get("gpu", {}).get("nvlink", False),
                bottleneck_categories=kernel_data.get("bottlenecks", []),
                kernel_times=kernel_data.get("kernel_summary", {}),
            )
            
            result = advisor.analyze_bottlenecks(context)
            result["context_used"] = {
                "gpu": context.gpu_name,
                "memory_gb": context.gpu_memory_gb,
                "num_gpus": context.num_gpus,
                "nvlink": context.nvlink_available,
            }
            
            return result
            
        except ImportError as e:
            return {"error": f"LLM advisor not available: {e}", "llm_available": False}
        except Exception as e:
            return {"error": str(e), "llm_available": False}
    
    def llm_distributed_recommendations(self, params: dict) -> dict:
        """Get LLM-powered distributed training recommendations."""
        try:
            from core.analysis.llm_advisor import get_advisor
            
            advisor = get_advisor()
            return advisor.get_distributed_recommendations(
                num_nodes=params.get("num_nodes", 1),
                gpus_per_node=params.get("gpus_per_node", 8),
                model_params_b=params.get("model_params_b", 70),
                interconnect=params.get("interconnect", "infiniband"),
            )
            
        except ImportError as e:
            return {"error": f"LLM advisor not available: {e}", "llm_available": False}
        except Exception as e:
            return {"error": str(e), "llm_available": False}
    
    def llm_inference_recommendations(self, params: dict) -> dict:
        """Get LLM-powered inference optimization recommendations."""
        try:
            from core.analysis.llm_advisor import get_advisor
            
            advisor = get_advisor()
            return advisor.get_inference_recommendations(
                model_name=params.get("model", "llama-3.1-70b"),
                target_latency_ms=params.get("target_latency_ms"),
                target_throughput=params.get("target_throughput"),
                max_batch_size=params.get("max_batch_size", 32),
                max_sequence_length=params.get("max_sequence_length", 4096),
            )
            
        except ImportError as e:
            return {"error": f"LLM advisor not available: {e}", "llm_available": False}
        except Exception as e:
            return {"error": str(e), "llm_available": False}
    
    def llm_rlhf_recommendations(self, params: dict) -> dict:
        """Get LLM-powered RLHF training recommendations."""
        try:
            from core.analysis.llm_advisor import get_advisor
            
            advisor = get_advisor()
            return advisor.get_rlhf_recommendations(
                policy_model_size_b=params.get("policy_size_b", 7),
                reward_model_size_b=params.get("reward_size_b", 7),
                num_gpus=params.get("num_gpus", 8),
            )
            
        except ImportError as e:
            return {"error": f"LLM advisor not available: {e}", "llm_available": False}
        except Exception as e:
            return {"error": str(e), "llm_available": False}
    
    def llm_custom_query(self, query: str) -> dict:
        """Send a custom query to the LLM advisor."""
        try:
            from core.analysis.llm_advisor import get_advisor, SYSTEM_PROMPT
            
            if not query.strip():
                return {"error": "Empty query", "llm_available": False}
            
            advisor = get_advisor()
            
            if not advisor.is_llm_available():
                return {
                    "error": "No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.",
                    "llm_available": False,
                    "suggestion": "Export your API key: export ANTHROPIC_API_KEY=your-key-here"
                }
            
            # Get context for the query
            gpu_info = self.get_gpu_info()
            context = f"""
Current hardware context:
- GPU: {gpu_info.get('gpu_name', 'Unknown')}
- Memory: {gpu_info.get('memory_total_gb', 0):.1f} GB
- GPUs: {gpu_info.get('gpu_count', 1)}

User question: {query}
"""
            
            response = advisor._call_llm(context, SYSTEM_PROMPT)
            
            return {
                "query": query,
                "response": response,
                "llm_available": True,
                "provider": advisor.config.provider,
                "model": advisor.config.model,
            }
            
        except ImportError as e:
            return {"error": f"LLM advisor not available: {e}", "llm_available": False}
        except Exception as e:
            return {"error": str(e), "llm_available": False}
    
    def log_message(self, format, *args):
        """Suppress logging for cleaner output."""
        pass


DashboardHandler = PerformanceCore  # Backwards compatibility alias


def create_handler(data_file: Optional[Path] = None):
    """Create a handler class with the data file bound."""
    def handler(*args, **kwargs):
        return PerformanceCore(*args, data_file=data_file, **kwargs)
    return handler


def serve_dashboard(port: int = 6970, data_file: Optional[Path] = None, open_browser: bool = True):
    """Start the dashboard server."""
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    handler = create_handler(data_file)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   ⚡ GPU Performance Lab Dashboard                                     ║
║                                                                        ║
║   Server running at: {url:<50} ║
║   Data source: {str(data_file or 'benchmark_test_results.json')[:50]:<50} ║
║                                                                        ║
║   📊 Data APIs:                                                        ║
║   • GET /api/data              - Benchmark results                     ║
║   • GET /api/gpu               - Live GPU status                       ║
║   • GET /api/llm-analysis      - LLM insights & explanations           ║
║   • GET /api/profiles          - Available profile data                ║
║                                                                        ║
║   🔬 Deep Profile Comparison (NEW!):                                   ║
║   • GET /api/deep-profile/list        - List comparable profiles       ║
║   • GET /api/deep-profile/compare/:ch - nsys/ncu metrics comparison    ║
║   • GET /api/deep-profile/recommendations - Analysis & recommendations ║
║                                                                        ║
║   🚀 Live Optimization Console (NEW!):                                 ║
║   • POST /api/optimize/start   - Start optimization with streaming     ║
║   • GET /api/optimize/stream/:id - SSE stream for live updates         ║
║   • GET /api/optimize/jobs     - List all optimization jobs            ║
║                                                                        ║
║   Press Ctrl+C to stop                                                 ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
        """)
        
        if open_browser:
            # Open browser after a short delay
            def open_delayed():
                time.sleep(0.5)
                webbrowser.open(url)
            threading.Thread(target=open_delayed, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard server stopped.")


app = typer.Typer(help="GPU Performance Lab Dashboard Server")


@app.command("serve")
def cli_serve(
    port: int = typer.Option(6970, "--port", "-p", help="Port to run the server on"),
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to benchmark results JSON file"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open browser automatically"),
) -> None:
    """Start the dashboard server."""
    serve_dashboard(port=port, data_file=data, open_browser=not no_browser)


def main() -> None:
    app()


if __name__ == '__main__':
    main()
