#pragma once

/**
 * cuda13_demo_runner.cuh
 *
 * Lightweight façade that exposes two helper utilities, giving the
 * chapter code a consistent way to demo CUDA 13.0 features without duplicating
 * the heavy-weight implementation.  The full demos live under
 * core/common/headers/cuda13_demos.cuh; this wrapper simply provides a stable API
 * (`cuda13_demo_runner::run_stream_ordered_demo()` and
 * `cuda13_demo_runner::run_tma_demo()`) while handling availability checks and
 * friendly logging.
 */

#include <cstdio>

#include "headers/cuda13_demos.cuh"

namespace cuda13_demo_runner {

inline bool cuda13_available() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    return true;
#else
    return false;
#endif
}

inline void run_stream_ordered_demo() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    cuda13_demos::run_stream_ordered_memory_demo();
#else
    std::printf("[CUDA 13] Stream-ordered memory helper skipped (CUDA 13.0+ required)\n");
#endif
}

inline void run_tma_demo() {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
    cuda13_demos::run_simple_tma_demo();
#else
    std::printf("[CUDA 13] TMA helper skipped (CUDA 13.0+ required)\n");
#endif
}

}  // namespace cuda13_demo_runner
