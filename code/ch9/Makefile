# Architecture switching support
# Target architecture: Blackwell B200/B300 (SM100)
ARCH ?= sm_100

# CUDA configuration
CUDA_VERSION = 13.0
NVCC = nvcc
# Use -gencode to embed both cubin (sm_100) and PTX (compute_100) for forward compatibility
NVCC_FLAGS = -O3 -std=c++17 -gencode arch=compute_100,code=[sm_100,compute_100] --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION) -Xcompiler -fPIC

# Find all CUDA source files
CUDA_SOURCES = $(wildcard *.cu)
TARGETS = $(CUDA_SOURCES:.cu=)

# Default target
all: $(TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"
	@echo "Built targets: $(TARGETS)"

# Build each target
$(TARGETS): %: %.cu
	@echo "Building $@..."
	@if [ "$@" = "cutlass_gemm_example" ]; then \
		$(NVCC) $(NVCC_FLAGS) -lcublas -lcuda -o $@ $<; \
	else \
		$(NVCC) $(NVCC_FLAGS) -lcuda -o $@ $<; \
	fi

# Architecture-specific targets
blackwell: ARCH=sm_100
blackwell: $(TARGETS)
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	perf record -g -o perf.data_$(ARCH) ./$(firstword $(TARGETS))
	perf report -i perf.data_$(ARCH)

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$(ARCH) ./$(firstword $(TARGETS))
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$(ARCH) ./$(firstword $(TARGETS))
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$(ARCH) ./$(firstword $(TARGETS))
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$(ARCH) ./$(firstword $(TARGETS))

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(TARGETS)
