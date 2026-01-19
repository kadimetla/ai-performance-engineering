"""Artifact manager for standardized benchmark artifact organization.

Manages directory structure for benchmark runs, ensuring consistent organization
of results, profiles, reports, logs, and manifests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


class ArtifactManager:
    """Manages artifact directory structure for benchmark runs.
    
    Creates and manages a standardized directory structure:
    artifacts/<timestamp>/
      - results/          # JSON results
      - profiles/         # nsys-rep, ncu-rep, torch traces
      - reports/          # markdown reports
      - logs/             # structured logs
      - manifest.json     # run manifest
    """
    
    def __init__(self, base_dir: Optional[Path] = None, run_id: Optional[str] = None):
        """Initialize artifact manager.
        
        Args:
            base_dir: Base directory for artifacts (defaults to ./artifacts)
            run_id: Optional run ID (defaults to timestamp)
        """
        if base_dir is None:
            base_dir = Path("artifacts")
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id
        
        # Create directory structure
        self._create_structure()
    
    def _create_structure(self) -> None:
        """Create the directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        (self.run_dir / "profiles").mkdir(exist_ok=True)
        (self.run_dir / "reports").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "progress").mkdir(exist_ok=True)
    
    @property
    def results_dir(self) -> Path:
        """Get path to results directory."""
        return self.run_dir / "results"
    
    @property
    def profiles_dir(self) -> Path:
        """Get path to profiles directory."""
        return self.run_dir / "profiles"
    
    @property
    def reports_dir(self) -> Path:
        """Get path to reports directory."""
        return self.run_dir / "reports"
    
    @property
    def logs_dir(self) -> Path:
        """Get path to logs directory."""
        return self.run_dir / "logs"

    @property
    def progress_dir(self) -> Path:
        """Get path to progress directory."""
        return self.run_dir / "progress"

    @property
    def manifest_path(self) -> Path:
        """Get path to manifest.json file."""
        return self.run_dir / "manifest.json"
    
    def get_result_path(self, filename: str = "benchmark_test_results.json") -> Path:
        """Get path for a result file.
        
        Args:
            filename: Name of the result file
        
        Returns:
            Path to result file
        """
        return self.results_dir / filename
    
    def get_profile_path(self, profiler: str, benchmark_name: str) -> Path:
        """Get path for a profiling artifact.
        
        Args:
            profiler: Profiler name ('nsys', 'ncu', 'torch')
            benchmark_name: Name of the benchmark
        
        Returns:
            Path to profile artifact
        """
        if profiler == "nsys":
            return self.profiles_dir / f"{benchmark_name}.nsys-rep"
        elif profiler == "ncu":
            return self.profiles_dir / f"{benchmark_name}.ncu-rep"
        elif profiler == "torch":
            return self.profiles_dir / f"{benchmark_name}_torch_trace.json"
        else:
            return self.profiles_dir / f"{benchmark_name}_{profiler}"
    
    def get_report_path(self, filename: str = "benchmark_report.md") -> Path:
        """Get path for a report file.
        
        Args:
            filename: Name of the report file
        
        Returns:
            Path to report file
        """
        return self.reports_dir / filename
    
    def get_log_path(self, filename: str = "benchmark.log") -> Path:
        """Get path for a log file.
        
        Args:
            filename: Name of the log file
        
        Returns:
            Path to log file
        """
        return self.logs_dir / filename
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.run_dir)
    
    def __repr__(self) -> str:
        """Representation."""
        return f"ArtifactManager(base_dir={self.base_dir}, run_id={self.run_id})"
