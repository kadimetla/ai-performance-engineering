#!/bin/bash
# Build the CUTLASS Blackwell GEMM library
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "Building CUTLASS Blackwell GEMM..."

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build . --parallel $(nproc)

echo ""
echo "Build complete! Library: ${BUILD_DIR}/cutlass_blackwell_gemm.so"
echo ""
echo "To use in Python:"
echo "  import torch"
echo "  torch.ops.load_library('${BUILD_DIR}/cutlass_blackwell_gemm.so')"
echo "  # Or: import ctypes; ctypes.CDLL('${BUILD_DIR}/cutlass_blackwell_gemm.so')"

