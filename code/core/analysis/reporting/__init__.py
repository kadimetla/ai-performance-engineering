"""
Report Generation Module

Generate polished PDF and HTML reports for:
- Benchmark results
- Profiling analysis
- Optimization summaries
- LLM insights
- Verification reports with anti-cheat status

Usage:
    from core.analysis.reporting import ReportGenerator
    
    generator = ReportGenerator()
    generator.generate_pdf(benchmark_data, "report.pdf")
    
    # Verification reports
    from core.analysis.reporting import generate_verification_report
    generate_verification_report(benchmark_data, "verification.html", gpu_name="H100")
"""

from .generator import ReportGenerator, ReportConfig, BenchmarkData, generate_report
from .templates import BenchmarkReport, ProfileReport, OptimizationReport
from .verification_report import (
    VerificationReportGenerator,
    VerificationReportData,
    VerificationBenchmarkResult,
    TheoreticalPeak,
    generate_verification_report,
    get_theoretical_peak,
    calculate_efficiency,
    calculate_roofline_bound,
    GPU_THEORETICAL_PEAKS,
)

__all__ = [
    # Core report generation
    'ReportGenerator',
    'ReportConfig',
    'BenchmarkData',
    'generate_report',
    'BenchmarkReport',
    'ProfileReport',
    'OptimizationReport',
    # Verification reports
    'VerificationReportGenerator',
    'VerificationReportData',
    'VerificationBenchmarkResult',
    'TheoreticalPeak',
    'generate_verification_report',
    'get_theoretical_peak',
    'calculate_efficiency',
    'calculate_roofline_bound',
    'GPU_THEORETICAL_PEAKS',
]
