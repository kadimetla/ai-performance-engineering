// Minimal stub library so builds can link with -lnvToolsExt even on CUDA 13+ where
// NVTX is header-only. This translation unit intentionally defines a single symbol
// to produce a valid static archive that satisfies the linker requirement.

extern "C" void __nvtx_stub_keep_alive() {}
