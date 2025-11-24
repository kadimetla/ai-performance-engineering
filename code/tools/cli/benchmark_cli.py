"""Unified CLI for benchmark execution and management using Typer."""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

try:
    import typer
    from typer import Option

    TYPER_AVAILABLE = True
except ImportError:  # pragma: no cover - Typer is optional for docs builds
    TYPER_AVAILABLE = False
    typer = None  # type: ignore
    Option = None  # type: ignore
    Argument = None  # type: ignore
    Context = None  # type: ignore

# Suppress CUDA capability warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

UTILITY_SCRIPTS = {
    "kv-cache": repo_root / "tools" / "utilities" / "kv_cache_calc.py",
    "cost-per-token": repo_root / "tools" / "utilities" / "calculate_cost_per_token.py",
    "compare-precision": repo_root / "tools" / "utilities" / "compare_precision_accuracy.py",
    "detect-cutlass": repo_root / "tools" / "utilities" / "detect_cutlass_info.py",
    "dump-hw": repo_root / "tools" / "utilities" / "dump_hardware_capabilities.py",
    "probe-hw": repo_root / "tools" / "utilities" / "probe_hardware_capabilities.py",
}


def _expand_multi_value_option(option_names: List[str]) -> None:
    """Allow passing `--option value1 value2` by rewriting argv."""
    argv = sys.argv
    if not set(option_names).intersection(argv):
        return
    new_argv = [argv[0]]
    i = 1
    option_set = set(option_names)
    while i < len(argv):
        token = argv[i]
        if token in option_set:
            option = token
            i += 1
            consumed = False
            while i < len(argv) and not argv[i].startswith("-"):
                new_argv.append(option)
                new_argv.append(argv[i])
                i += 1
                consumed = True
            if not consumed:
                new_argv.append(option)
            continue
        new_argv.append(token)
        i += 1
    sys.argv = new_argv


_expand_multi_value_option(["--targets", "-t"])

from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities
from common.python.logger import setup_logging, get_logger
from common.python.artifact_manager import ArtifactManager
from tools.verification.verify_all_benchmarks import resolve_target_chapters, run_verification
from common.python import profiler_config as profiler_config_mod
from common.python.discovery import chapter_slug, discover_all_chapters

apply_env_defaults()


def _validate_output_format(fmt: str | None) -> str:
    if fmt is None:
        return "both"
    normalized = fmt.strip().lower()
    valid = {"json", "markdown", "both"}
    if normalized not in valid:
        message = "Output format must be one of json, markdown, or both"
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _validate_ncu_metric_set(metric_set: str) -> str:
    normalized = metric_set.strip().lower()
    valid = {"auto", "deep_dive", "roofline", "minimal"}
    if normalized not in valid:
        message = (
            f"Invalid Nsight Compute metric set '{metric_set}'. "
            "Choose from 'auto', 'deep_dive', 'roofline', or 'minimal'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    if normalized != "auto":
        profiler_config_mod.set_default_profiler_metric_set(normalized)
    return normalized


def _validate_profile_type(profile: str | None) -> str:
    if profile is None:
        return "none"
    normalized = profile.strip().lower()
    valid = {"none", "minimal", "deep_dive", "roofline"}
    if normalized not in valid:
        message = (
            f"Invalid profile choice '{profile}'. "
            "Choose from 'none', 'minimal', 'deep_dive', or 'roofline'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _parse_target_extra_args(entries: Optional[List[str]]) -> Dict[str, List[str]]:
    """Parse --target-extra-arg entries of the form target="--flag value"."""
    parsed: Dict[str, List[str]] = {}
    for entry in entries or []:
        target, sep, args = entry.partition("=")
        if not sep or not target or not args:
            continue
        parsed[target.strip()] = shlex.split(args)
    return parsed


def _apply_suite_timeout(seconds: Optional[int]) -> None:
    """Install a SIGALRM to stop the suite after a timeout."""
    if seconds is None or seconds <= 0:
        return

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Benchmark suite exceeded timeout of {seconds} seconds")

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(seconds)


def _run_utility(tool: str, tool_args: Optional[List[str]]) -> int:
    """Execute a utility script with passthrough arguments."""
    script_path = UTILITY_SCRIPTS.get(tool)
    if script_path is None:
        raise ValueError(f"Unknown utility '{tool}'.")
    if not script_path.exists():
        raise FileNotFoundError(f"Utility script not found at {script_path}")

    extra_args = tool_args or []
    cmd = [sys.executable, str(script_path), *extra_args]
    result = subprocess.run(cmd)
    return result.returncode


# Import architecture optimizations early
try:
    import arch_config  # noqa: F401
except ImportError:
    pass

# Import benchmark functionality
try:
    import torch  # noqa: F401
    from common.python.chapter_compare_template import discover_benchmarks
    from tools.testing.run_all_benchmarks import test_chapter, generate_markdown_report

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Test functions come from run_all_benchmarks; if that import failed above, this is False.
TEST_FUNCTIONS_AVAILABLE = BENCHMARK_AVAILABLE

# Typer app setup
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="benchmark",
        help="Unified benchmark execution and management CLI",
        add_completion=False,
    )
else:
    app = None


def _execute_benchmarks(
    targets: Optional[List[str]] = None,
    output_format: str = "both",
    profile_type: str = "none",
    suite_timeout: Optional[int] = 14400,
    timeout_multiplier: float = 1.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    force_pipeline: bool = False,
    artifacts_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    ncu_metric_set: str = "auto",
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    torchrun_env: Optional[List[str]] = None,
    target_extra_args: Optional[List[str]] = None,
) -> None:
    """Execute selected benchmarks with optional profiling."""
    parsed_extra_args = _parse_target_extra_args(target_extra_args)

    try:
        from common.python.cuda_capabilities import set_force_pipeline

        set_force_pipeline(force_pipeline)
    except Exception:
        pass

    artifact_manager = ArtifactManager(base_dir=Path(artifacts_dir) if artifacts_dir else None)
    if log_file is None:
        log_file = artifact_manager.get_log_path()

    setup_logging(level=log_level, log_file=log_file, log_format="json", use_rich=True)
    logger = get_logger(__name__)

    if not BENCHMARK_AVAILABLE or not TEST_FUNCTIONS_AVAILABLE:
        logger.error("Benchmark dependencies missing (torch/benchmark_harness or test functions).")
        sys.exit(1)

    try:
        dump_environment_and_capabilities()
    except Exception as exc:
        logger.error(f"Skipping environment/capabilities dump due to error: {exc}")

    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(targets)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"FOUND {len(chapter_dirs)} chapter(s)")

    enable_profiling = (profile_type or "none").lower() != "none"

    _apply_suite_timeout(suite_timeout)

    all_results = []
    for chapter_dir in chapter_dirs:
        chapter_id = chapter_slug(chapter_dir, repo_root)
        example_filters = chapter_filters.get(chapter_id)
        only_examples = sorted(example_filters) if example_filters else None
        result = test_chapter(
            chapter_dir=chapter_dir,
            enable_profiling=enable_profiling,
            profile_type=profile_type if enable_profiling else "none",
            timeout_multiplier=timeout_multiplier,
            reproducible=reproducible,
            cold_start=cold_start,
            iterations=iterations,
            warmup=warmup,
            only_examples=only_examples,
            accept_regressions=accept_regressions or update_expectations,
            ncu_metric_set=ncu_metric_set,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            env_passthrough=torchrun_env,
            target_extra_args=parsed_extra_args,
        )
        all_results.append(result)

    output_json = artifact_manager.get_result_path("benchmark_test_results.json")
    output_md = artifact_manager.get_report_path("benchmark_test_results.md")

    if output_format in ["json", "both"]:
        with open(output_json, "w") as f:
            json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": all_results}, f, indent=2)
        logger.info(f"JSON results saved to: {output_json}")
    if output_format in ["markdown", "both"]:
        generate_markdown_report(all_results, output_md)
        logger.info(f"Markdown report saved to: {output_md}")

    total_failed = sum(r.get("summary", {}).get("failed", 0) for r in all_results)
    if total_failed > 0:
        sys.exit(1)


if TYPER_AVAILABLE:

    @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
    def run(
        ctx: typer.Context,
        targets: Optional[List[str]] = Option(None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to run. Repeat the flag for multiple targets. Omit or use 'all' for every chapter."),
        output_format: str = Option("both", "--format", "-f", help="Output format: 'json', 'markdown', or 'both'", callback=_validate_output_format),
        profile_type: str = Option("none", "--profile", "-p", help="Profiling preset: none (default), minimal, deep_dive, or roofline. Non-'none' enables nsys/ncu/PyTorch profiling.", callback=_validate_profile_type),
        suite_timeout: Optional[int] = Option(14400, "--suite-timeout", help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)"),
        timeout_multiplier: float = Option(1.0, "--timeout-multiplier", help="Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts)"),
        reproducible: bool = Option(False, "--reproducible", help="Enable reproducible mode: set all seeds to 42 and force deterministic algorithms (uses slower fallbacks; ops without deterministic support may error)."),
        cold_start: bool = Option(False, "--cold-start", help="Reset GPU state between benchmarks for cold start measurements"),
        iterations: Optional[int] = Option(None, "--iterations", help="Number of benchmark iterations (default: chapter-specific)"),
        warmup: Optional[int] = Option(None, "--warmup", help="Number of warmup iterations (default: chapter-specific)"),
        force_pipeline: bool = Option(False, "--force-pipeline", help="Force enable CUDA Pipeline API even on compute capability 12.0+ (may cause instability on Blackwell GPUs)"),
        artifacts_dir: Optional[str] = Option(None, "--artifacts-dir", help="Directory for artifacts (default: ./artifacts)"),
        log_level: str = Option("INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
        log_file: Optional[str] = Option(None, "--log-file", help="Path to log file (default: artifacts/<run_id>/logs/benchmark.log)"),
        ncu_metric_set: str = Option("auto", "--ncu-metric-set", help="Nsight Compute metric preset: auto, minimal, deep_dive, or roofline. If auto, the profile type governs metric selection.", callback=_validate_ncu_metric_set),
        accept_regressions: bool = Option(False, "--accept-regressions", help="Update expectation files when improvements are detected instead of flagging regressions.", is_flag=True),
        update_expectations: bool = Option(False, "--update-expectations", help="Force-write observed metrics into expectation files (overrides regressions). Useful for refreshing baselines on new hardware.", is_flag=True),
        launch_via: str = Option("python", "--launch-via", help="Launcher to use for benchmarks: python or torchrun."),
        nproc_per_node: Optional[int] = Option(None, "--nproc-per-node", help="torchrun --nproc_per_node value."),
        nnodes: Optional[str] = Option(None, "--nnodes", help="torchrun --nnodes value."),
        rdzv_backend: Optional[str] = Option(None, "--rdzv-backend", help="torchrun rendezvous backend (defaults to c10d when nnodes is set)."),
        rdzv_endpoint: Optional[str] = Option(None, "--rdzv-endpoint", help="torchrun rendezvous endpoint (host:port)."),
        torchrun_env: Optional[List[str]] = Option(None, "--torchrun-env", help="Environment variables to forward into torchrun launches (repeatable)."),
        target_extra_args: Optional[List[str]] = Option(None, "--target-extra-arg", help='Per-target extra args, format: target="--flag value". Repeatable.'),
    ):
        """Run benchmarks - discover, run, and summarize results."""
        combined_targets: List[str] = []
        for arg in (list(targets) if targets else []):
            if arg:
                combined_targets.append(arg)
        for extra in ctx.args or []:
            if not extra:
                continue
            # Drop stray values that belong to other options when Click defers parsing.
            if extra.lower() in {"none", "minimal", "deep_dive", "roofline", "json", "markdown", "both"}:
                continue
            combined_targets.append(extra)
        # Final cleanup: drop any falsy or duplicate entries
        combined_targets = [t for t in combined_targets if t]
        # Deduplicate to avoid running the same target multiple times when
        # Click/Typer shuffles positional args.
        combined_targets = list(dict.fromkeys(combined_targets))
        _execute_benchmarks(
            targets=combined_targets or None,
            output_format=output_format,
            profile_type=profile_type,
            suite_timeout=suite_timeout,
            timeout_multiplier=timeout_multiplier,
            reproducible=reproducible,
            cold_start=cold_start,
            iterations=iterations,
            warmup=warmup,
            force_pipeline=force_pipeline,
            artifacts_dir=artifacts_dir,
            log_level=log_level,
            log_file=log_file,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            ncu_metric_set=ncu_metric_set,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            torchrun_env=torchrun_env,
            target_extra_args=target_extra_args,
        )

    @app.command()
    def verify(
        targets: Optional[List[str]] = Option(None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to verify. Repeat the flag for multiple targets. Omit or use 'all' for every chapter."),
    ):
        """Run the lightweight benchmark verification harness."""
        exit_code = run_verification(list(targets) if targets else None)
        raise typer.Exit(code=exit_code)

    @app.command("list-targets")
    def list_targets(
        chapter: Optional[str] = Option(
            None,
            "--chapter",
            "-c",
            help="Limit output to a single chapter (e.g., ch15 or labs/blackwell_matmul).",
        ),
    ):
        """List available benchmark targets in chapter:example format."""
        if chapter:
            chapter_dirs, _ = resolve_target_chapters([chapter])
        else:
            chapter_dirs = discover_all_chapters(repo_root)

        if not chapter_dirs:
            typer.echo("No chapter directories found.")
            raise typer.Exit(code=1)

        any_targets = False
        for chapter_dir in chapter_dirs:
            chapter_id = chapter_slug(chapter_dir, repo_root)
            pairs = discover_benchmarks(chapter_dir)
            if not pairs:
                continue
            any_targets = True
            for _, _, example in sorted(pairs, key=lambda entry: entry[2]):
                typer.echo(f"{chapter_id}:{example}")

        if not any_targets:
            typer.echo("No benchmark targets discovered.")

    @app.command("utils")
    def utils(
        tool: str = Option(
            ...,
            "--tool",
            "-u",
            help=f"Utility to run. Available: {', '.join(sorted(UTILITY_SCRIPTS))}",
        ),
        tool_args: Optional[List[str]] = typer.Argument(
            None,
            help="Arguments forwarded to the utility (use -- to separate).",
        ),
    ):
        """Run repository utilities (e.g., KV cache calculator) from one entrypoint."""
        try:
            exit_code = _run_utility(tool, tool_args)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        except FileNotFoundError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)

        raise typer.Exit(code=exit_code)


def main():
    """Entry point for CLI."""
    if not TYPER_AVAILABLE:
        print("ERROR: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)

    if app is None:
        print("ERROR: CLI not available")
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()
