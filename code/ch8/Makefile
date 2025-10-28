# Architecture switching support
# Target architecture: Blackwell B200/B300 (SM100)
ARCH ?= sm_100

# CUDA configuration
CUDA_VERSION = 13.0
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)
NVCC_FLAGS += --threads=0

# Find all CUDA source files
CUDA_SOURCES = $(wildcard *.cu)
CUDA_TARGETS = $(CUDA_SOURCES:.cu=)

# Default target - build all CUDA examples
all: $(CUDA_TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"
	@echo "✓ Built $(words $(CUDA_TARGETS)) CUDA examples"

# Build individual targets
$(CUDA_TARGETS): %: %.cu
	@echo "Compiling $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda

# Architecture-specific targets
blackwell: ARCH=sm_100
blackwell: all
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: all
	@echo "Nsight Systems timeline profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o nsys_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-ncu: all
	@echo "Nsight Compute kernel profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe..."; \
		ncu --section=SpeedOfLight --metrics breakdown:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all -o ncu_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o hta_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe with perf..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || echo "perf failed for $$exe"; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(CUDA_TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o comprehensive_timeline_$$exe_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --section=SpeedOfLight --metrics breakdown:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all -o comprehensive_kernel_$$exe_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "3. Perf system-level analysis..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || true; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

clean:
	rm -f $(CUDA_TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(CUDA_TARGETS)
