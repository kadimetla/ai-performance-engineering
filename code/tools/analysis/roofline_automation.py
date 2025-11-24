#!/usr/bin/env python3
"""Automated roofline analysis for Blackwell GPUs.

Generates dual roofline plots showing SM compute and TMEM bandwidth limits
for kernel performance analysis.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.logger import get_logger
from common.python.hardware_capabilities import detect_capabilities

logger = get_logger(__name__)


class RooflineAnalyzer:
    """Automated roofline analysis for Blackwell."""
    
    # Hardware specifications (default to B200)
    HARDWARE_SPECS = {
        'B200': {
            'sm_tflops': 2500.0,  # BF16 Tensor Core
            'sm_bandwidth_gb_s': 8000.0,  # HBM3e
            'tmem_bandwidth_gb_s': 20000.0,  # TMEM aggregate
            'fp8_tflops': 5000.0,
            'fp32_tflops': 625.0,
        },
        'B300': {
            'sm_tflops': 2500.0,
            'sm_bandwidth_gb_s': 12000.0,
            'tmem_bandwidth_gb_s': 30000.0,
            'fp8_tflops': 5000.0,
            'fp32_tflops': 625.0,
        },
        'GB200': {
            'sm_tflops': 2500.0,
            'sm_bandwidth_gb_s': 8000.0,
            'tmem_bandwidth_gb_s': 20000.0,
            'fp8_tflops': 5000.0,
            'fp32_tflops': 625.0,
        },
        'GB300': {
            'sm_tflops': 2500.0,
            'sm_bandwidth_gb_s': 12000.0,
            'tmem_bandwidth_gb_s': 30000.0,
            'fp8_tflops': 5000.0,
            'fp32_tflops': 625.0,
        },
    }
    
    def __init__(self, hardware: str = 'auto'):
        """Initialize roofline analyzer.
        
        Args:
            hardware: Hardware type ('B200', 'B300', 'GB200', 'GB300', or 'auto')
        """
        if hardware == 'auto':
            self.hardware = self._detect_hardware()
        else:
            self.hardware = hardware
        
        self.specs = self.HARDWARE_SPECS.get(self.hardware, self.HARDWARE_SPECS['B200'])
        logger.info(f"Using hardware specs: {self.hardware}")
    
    def _detect_hardware(self) -> str:
        """Detect hardware from capabilities."""
        cap = detect_capabilities()
        if cap is None:
            logger.warning("Could not detect hardware, using B200 defaults")
            return 'B200'
        
        name = cap.name.upper()
        if 'B200' in name:
            return 'B200'
        elif 'B300' in name:
            return 'B300'
        elif 'GB200' in name:
            return 'GB200'
        elif 'GB300' in name:
            return 'GB300'
        else:
            logger.warning(f"Unknown hardware {name}, using B200 defaults")
            return 'B200'
    
    def calculate_roofline(
        self,
        min_ai: float = 0.01,
        max_ai: float = 1000.0,
        precision: str = 'bf16'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate roofline curves.
        
        Args:
            min_ai: Minimum arithmetic intensity (FLOPs/byte)
            max_ai: Maximum arithmetic intensity
            precision: Precision ('fp32', 'bf16', 'fp8')
        
        Returns:
            ai: Arithmetic intensity array
            sm_roofline: SM compute roofline
            tmem_roofline: TMEM bandwidth roofline
        """
        ai = np.logspace(np.log10(min_ai), np.log10(max_ai), 1000)
        
        # Select peak performance based on precision
        if precision == 'fp8':
            peak_tflops = self.specs['fp8_tflops']
        elif precision in ['bf16', 'fp16']:
            peak_tflops = self.specs['sm_tflops']
        else:  # fp32
            peak_tflops = self.specs['fp32_tflops']
        
        # SM roofline: limited by HBM bandwidth
        sm_bandwidth_tb_s = self.specs['sm_bandwidth_gb_s'] / 1000.0
        sm_roofline = np.minimum(ai * sm_bandwidth_tb_s, peak_tflops)
        
        # TMEM roofline: limited by TMEM bandwidth
        tmem_bandwidth_tb_s = self.specs['tmem_bandwidth_gb_s'] / 1000.0
        tmem_roofline = np.minimum(ai * tmem_bandwidth_tb_s, peak_tflops)
        
        return ai, sm_roofline, tmem_roofline
    
    def plot_roofline(
        self,
        kernels: Optional[List[Dict]] = None,
        output_path: Optional[Path] = None,
        precision: str = 'bf16',
        title: Optional[str] = None
    ):
        """Generate dual roofline plot.
        
        Args:
            kernels: List of kernel data dicts with keys:
                - name: Kernel name
                - ai: Arithmetic intensity (FLOPs/byte)
                - tflops: Achieved TFLOP/s
                - uses_tmem: Bool indicating TMEM usage
            output_path: Path to save plot (PNG)
            precision: Precision ('fp32', 'bf16', 'fp8')
            title: Optional plot title
        """
        ai, sm_roofline, tmem_roofline = self.calculate_roofline(precision=precision)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot rooflines
        ax.loglog(ai, sm_roofline, 'b-', linewidth=2, label='SM Compute Roofline (HBM)')
        ax.loglog(ai, tmem_roofline, 'r-', linewidth=2, label='TMEM Roofline')
        
        # Plot kernels if provided
        if kernels:
            for kernel in kernels:
                marker = 'o' if kernel.get('uses_tmem', False) else 's'
                color = 'red' if kernel.get('uses_tmem', False) else 'blue'
                
                ax.loglog(
                    kernel['ai'],
                    kernel['tflops'],
                    marker=marker,
                    markersize=10,
                    color=color,
                    label=kernel.get('name', 'Kernel')
                )
                
                # Annotate
                ax.annotate(
                    kernel.get('name', 'Kernel'),
                    xy=(kernel['ai'], kernel['tflops']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7)
                )
        
        # Labels and formatting
        ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
        ax.set_ylabel('Performance (TFLOP/s)', fontsize=12)
        
        if title is None:
            title = f'{self.hardware} Dual Roofline - {precision.upper()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        
        # Add hardware specs annotation
        specs_text = (
            f"Hardware: {self.hardware}\n"
            f"Peak: {self.specs['sm_tflops']:.0f} TFLOP/s ({precision.upper()})\n"
            f"HBM: {self.specs['sm_bandwidth_gb_s']:.0f} GB/s\n"
            f"TMEM: {self.specs['tmem_bandwidth_gb_s']:.0f} GB/s"
        )
        ax.text(
            0.02, 0.98, specs_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved roofline plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_kernel(
        self,
        flops: float,
        bytes_accessed: float,
        time_ms: float,
        uses_tmem: bool = False,
        name: str = "Kernel"
    ) -> Dict:
        """Analyze kernel performance.
        
        Args:
            flops: Total FLOPs executed
            bytes_accessed: Total bytes accessed
            time_ms: Execution time in milliseconds
            uses_tmem: Whether kernel uses TMEM
            name: Kernel name
        
        Returns:
            analysis: Dict with performance metrics and bottleneck info
        """
        time_s = time_ms / 1000.0
        
        # Calculate metrics
        tflops = (flops / time_s) / 1e12
        bandwidth_gb_s = (bytes_accessed / time_s) / 1e9
        ai = flops / bytes_accessed if bytes_accessed > 0 else 0
        
        # Determine bottleneck
        if uses_tmem:
            bandwidth_limit = self.specs['tmem_bandwidth_gb_s']
            roofline_name = 'TMEM'
        else:
            bandwidth_limit = self.specs['sm_bandwidth_gb_s']
            roofline_name = 'SM/HBM'
        
        peak_tflops = self.specs['sm_tflops']
        
        # Ridge point: AI where we transition from memory-bound to compute-bound
        ridge_ai = peak_tflops * 1000 / bandwidth_limit  # TFLOP/s to GFLOP/s, GB/s
        
        if ai < ridge_ai * 0.8:
            bottleneck = f"Memory-bound ({roofline_name})"
            efficiency = (bandwidth_gb_s / bandwidth_limit) * 100
        else:
            bottleneck = "Compute-bound"
            efficiency = (tflops / peak_tflops) * 100
        
        analysis = {
            'name': name,
            'ai': ai,
            'tflops': tflops,
            'bandwidth_gb_s': bandwidth_gb_s,
            'uses_tmem': uses_tmem,
            'bottleneck': bottleneck,
            'efficiency_pct': efficiency,
            'ridge_ai': ridge_ai,
        }
        
        logger.info(f"\nKernel Analysis: {name}")
        logger.info(f"  Arithmetic Intensity: {ai:.2f} FLOPs/byte")
        logger.info(f"  Achieved: {tflops:.2f} TFLOP/s ({bandwidth_gb_s:.2f} GB/s)")
        logger.info(f"  Bottleneck: {bottleneck}")
        logger.info(f"  Efficiency: {efficiency:.1f}%")
        
        return analysis
    
    def load_kernel_data(self, csv_path: Path) -> List[Dict]:
        """Load kernel data from CSV.
        
        CSV format:
        name,flops,bytes,time_ms,uses_tmem
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            kernels: List of kernel dicts
        """
        import csv
        
        kernels = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                analysis = self.analyze_kernel(
                    flops=float(row['flops']),
                    bytes_accessed=float(row['bytes']),
                    time_ms=float(row['time_ms']),
                    uses_tmem=row.get('uses_tmem', 'false').lower() == 'true',
                    name=row['name']
                )
                kernels.append(analysis)
        
        logger.info(f"Loaded {len(kernels)} kernels from {csv_path}")
        return kernels


def main():
    parser = argparse.ArgumentParser(description="Automated Roofline Analysis")
    parser.add_argument('--hardware', type=str, default='auto',
                       choices=['auto', 'B200', 'B300', 'GB200', 'GB300'],
                       help='Hardware type')
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['fp32', 'bf16', 'fp16', 'fp8'],
                       help='Precision for peak performance')
    parser.add_argument('--kernels-csv', type=Path,
                       help='CSV file with kernel data')
    parser.add_argument('--output', type=Path,
                       help='Output path for plot (PNG)')
    parser.add_argument('--title', type=str,
                       help='Plot title')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = RooflineAnalyzer(hardware=args.hardware)
    
    # Load kernel data if provided
    kernels = None
    if args.kernels_csv:
        kernels = analyzer.load_kernel_data(args.kernels_csv)
    
    # Generate plot
    analyzer.plot_roofline(
        kernels=kernels,
        output_path=args.output,
        precision=args.precision,
        title=args.title
    )
    
    print(f"\n{'='*60}")
    print(f"Roofline Analysis Complete")
    print(f"{'='*60}")
    print(f"Hardware: {analyzer.hardware}")
    print(f"Precision: {args.precision}")
    if kernels:
        print(f"Kernels analyzed: {len(kernels)}")
    if args.output:
        print(f"Plot saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

