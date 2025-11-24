#!/usr/bin/env python3
"""CUTLASS Profiler integration for automated kernel selection.

Automates CUTLASS profiler to find optimal GEMM kernels for Blackwell.
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.logger import get_logger

logger = get_logger(__name__)


class CUTLASSProfilerIntegration:
    """Automated CUTLASS profiler integration."""
    
    def __init__(self, output_dir: Path = Path("artifacts/cutlass")):
        """Initialize CUTLASS profiler integration."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check CUTLASS availability
        self.cutlass_available = self._check_cutlass()
        
        if self.cutlass_available:
            logger.info("✓ CUTLASS profiler available")
        else:
            logger.warning("✗ CUTLASS profiler not found")
    
    def _check_cutlass(self) -> bool:
        """Check if CUTLASS profiler is available."""
        try:
            result = subprocess.run(
                ['cutlass_profiler', '--help'],
                capture_output=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def profile_gemm(
        self,
        m: int,
        n: int,
        k: int,
        data_type: str = 'f16',
        operation: str = 'gemm',
        output_name: Optional[str] = None
    ) -> Optional[Dict]:
        """Profile GEMM operation with CUTLASS.
        
        Args:
            m, n, k: GEMM dimensions
            data_type: Data type ('f16', 'bf16', 'f32', 'f8')
            operation: Operation type ('gemm', 'conv2d', etc.)
            output_name: Base name for output
        
        Returns:
            results: Dict with best kernel configuration
        """
        if not self.cutlass_available:
            logger.error("CUTLASS profiler not available")
            return None
        
        if output_name is None:
            output_name = f"{operation}_{m}x{n}x{k}_{data_type}"
        
        output_csv = self.output_dir / f"{output_name}.csv"
        
        # Build CUTLASS profiler command
        cmd = [
            'cutlass_profiler',
            f'--operation={operation}',
            f'--m={m}',
            f'--n={n}',
            f'--k={k}',
            f'--A={data_type}',
            f'--B={data_type}',
            f'--C={data_type}',
            '--providers=cutlass',
            '--output=' + str(output_csv),
        ]
        
        logger.info(f"Profiling {operation} {m}×{n}×{k} ({data_type})...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"CUTLASS profiler failed: {result.stderr}")
                return None
            
            # Parse results
            results = self._parse_results(output_csv)
            
            logger.info(f"Found {len(results)} kernel configurations")
            if results:
                best = results[0]
                logger.info(f"Best kernel: {best['kernel']}, {best['gflops']:.2f} GFLOP/s")
            
            return results[0] if results else None
        
        except subprocess.TimeoutExpired:
            logger.error("CUTLASS profiler timed out")
            return None
        except Exception as e:
            logger.error(f"Error running CUTLASS profiler: {e}")
            return None
    
    def _parse_results(self, csv_path: Path) -> List[Dict]:
        """Parse CUTLASS profiler CSV output."""
        import csv
        
        results = []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append({
                        'kernel': row.get('Kernel', ''),
                        'gflops': float(row.get('GFLOPs', 0)),
                        'runtime_ms': float(row.get('Runtime', 0)),
                        'provider': row.get('Provider', 'cutlass'),
                    })
            
            # Sort by performance
            results.sort(key=lambda x: x['gflops'], reverse=True)
        
        except Exception as e:
            logger.warning(f"Could not parse CUTLASS results: {e}")
        
        return results
    
    def batch_profile(
        self,
        shapes: List[Tuple[int, int, int]],
        data_type: str = 'f16'
    ) -> Dict[str, Dict]:
        """Profile multiple GEMM shapes.
        
        Args:
            shapes: List of (M, N, K) tuples
            data_type: Data type for all GEMMs
        
        Returns:
            all_results: Dict mapping shape to best kernel
        """
        all_results = {}
        
        for m, n, k in shapes:
            shape_key = f"{m}x{n}x{k}"
            logger.info(f"Profiling shape: {shape_key}")
            
            result = self.profile_gemm(m, n, k, data_type=data_type)
            if result:
                all_results[shape_key] = result
        
        logger.info(f"Batch profiling complete: {len(all_results)} shapes")
        
        # Save summary
        summary_path = self.output_dir / "batch_profile_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="CUTLASS Profiler Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile single GEMM
  python cutlass_profiler_integration.py --m 4096 --n 4096 --k 4096 --dtype bf16
  
  # Batch profile common shapes
  python cutlass_profiler_integration.py --batch-shapes \\
    2048,2048,2048 4096,4096,4096 8192,8192,8192
        """
    )
    
    parser.add_argument('--m', type=int, help='M dimension')
    parser.add_argument('--n', type=int, help='N dimension')
    parser.add_argument('--k', type=int, help='K dimension')
    parser.add_argument('--dtype', type=str, default='bf16',
                       choices=['f16', 'bf16', 'f32', 'f8'],
                       help='Data type')
    parser.add_argument('--batch-shapes', nargs='+',
                       help='Batch mode: list of M,N,K shapes')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/cutlass'))
    
    args = parser.parse_args()
    
    # Create integration
    profiler = CUTLASSProfilerIntegration(output_dir=args.output_dir)
    
    # Batch mode
    if args.batch_shapes:
        shapes = []
        for shape_str in args.batch_shapes:
            m, n, k = map(int, shape_str.split(','))
            shapes.append((m, n, k))
        
        results = profiler.batch_profile(shapes, data_type=args.dtype)
        
        print(f"\n{'='*60}")
        print(f"CUTLASS Batch Profiling Results")
        print(f"{'='*60}")
        for shape, result in results.items():
            print(f"{shape}: {result['gflops']:.2f} GFLOP/s ({result['kernel']})")
        print(f"{'='*60}\n")
        return
    
    # Single profile mode
    if args.m is None or args.n is None or args.k is None:
        parser.error("--m, --n, --k required (or use --batch-shapes)")
    
    result = profiler.profile_gemm(
        m=args.m,
        n=args.n,
        k=args.k,
        data_type=args.dtype
    )
    
    if result:
        print(f"\n{'='*60}")
        print(f"CUTLASS Profiling Result")
        print(f"{'='*60}")
        print(f"Shape: {args.m}×{args.n}×{args.k}")
        print(f"Data type: {args.dtype}")
        print(f"Best kernel: {result['kernel']}")
        print(f"Performance: {result['gflops']:.2f} GFLOP/s")
        print(f"Runtime: {result['runtime_ms']:.4f} ms")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
