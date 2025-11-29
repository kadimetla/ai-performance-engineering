/**
 * True Warp Specialization v2 - Simplified
 * =========================================
 * 
 * Clean implementation with proper producer/consumer separation.
 * Based on CUTLASS patterns but simplified for clarity.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>

using namespace cute;

namespace true_warp_spec2_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kStages = 4;

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct TrueWarpSpecSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A[kStages];
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B[kStages];
  alignas(16) cute::uint64_t full_barrier[kStages];
  alignas(16) cute::uint64_t empty_barrier[kStages];
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int stage) {
    return make_tensor(make_smem_ptr(smem_A[stage].begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB(int stage) {
    return make_tensor(make_smem_ptr(smem_B[stage].begin()), BSmemLayout{});
  }
};

using MmaTag = SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>;
using TmaLoad = SM90_TMA_LOAD;

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
__global__ void __launch_bounds__(128, 1)
gemm_true_warp_spec2(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
                     MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
                     int grid_m, int grid_n,
                     CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                     CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;
  bool is_warp0 = (warp_idx == 0);
  bool is_lane0 = (lane_idx == 0);

  int tile_m = blockIdx.x;
  int tile_n = blockIdx.y;
  auto mma_coord = make_coord(tile_m, tile_n, _);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageT& storage = *reinterpret_cast<SharedStorageT*>(shared_memory);

  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (is_warp0 && is_lane0) {
    tmem_allocator.allocate(decltype(tmem_allocator)::Sm100TmemCapacityColumns, &storage.tmem_base_ptr);
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(storage.full_barrier[s], 1);
      cute::initialize_barrier(storage.empty_barrier[s], 1);
    }
    cute::initialize_barrier(storage.mma_barrier, 1);
  }
  __syncthreads();

  uint32_t tmem_base = storage.tmem_base_ptr;
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
  tCtAcc.data() = tmem_base;

  Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
  Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
  Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

  auto tCsA_0 = storage.tensor_sA(0); auto tCsA_1 = storage.tensor_sA(1);
  auto tCsA_2 = storage.tensor_sA(2); auto tCsA_3 = storage.tensor_sA(3);
  auto tCsB_0 = storage.tensor_sB(0); auto tCsB_1 = storage.tensor_sB(1);
  auto tCsB_2 = storage.tensor_sB(2); auto tCsB_3 = storage.tensor_sB(3);

  auto tCrA_0 = cta_mma.make_fragment_A(tCsA_0);
  auto tCrA_1 = cta_mma.make_fragment_A(tCsA_1);
  auto tCrA_2 = cta_mma.make_fragment_A(tCsA_2);
  auto tCrA_3 = cta_mma.make_fragment_A(tCsA_3);
  auto tCrB_0 = cta_mma.make_fragment_B(tCsB_0);
  auto tCrB_1 = cta_mma.make_fragment_B(tCsB_1);
  auto tCrB_2 = cta_mma.make_fragment_B(tCsB_2);
  auto tCrB_3 = cta_mma.make_fragment_B(tCsB_3);

  auto [tAgA_0, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsA_0), group_modes<0,3>(tCgCoordA));
  auto [tBgB_0, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsB_0), group_modes<0,3>(tCgCoordB));
  auto [tAgA_1, tAsA_1] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsA_1), group_modes<0,3>(tCgCoordA));
  auto [tBgB_1, tBsB_1] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsB_1), group_modes<0,3>(tCgCoordB));
  auto [tAgA_2, tAsA_2] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsA_2), group_modes<0,3>(tCgCoordA));
  auto [tBgB_2, tBsB_2] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsB_2), group_modes<0,3>(tCgCoordB));
  auto [tAgA_3, tAsA_3] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsA_3), group_modes<0,3>(tCgCoordA));
  auto [tBgB_3, tBsB_3] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsB_3), group_modes<0,3>(tCgCoordB));

  int tma_bytes = sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0));
  int num_k_tiles = size<3>(tCgA);

  int full_phase[kStages] = {0, 0, 0, 0};
  int empty_phase[kStages] = {0, 0, 0, 0};
  int mma_phase = 0;

  // Lambda to issue TMA
  auto issue_tma = [&](int s, int k) {
    cute::set_barrier_transaction_bytes(storage.full_barrier[s], tma_bytes);
    if (s == 0) { copy(tma_atom_A.with(storage.full_barrier[0]), tAgA_0(_, k), tAsA_0); copy(tma_atom_B.with(storage.full_barrier[0]), tBgB_0(_, k), tBsB_0); }
    if (s == 1) { copy(tma_atom_A.with(storage.full_barrier[1]), tAgA_1(_, k), tAsA_1); copy(tma_atom_B.with(storage.full_barrier[1]), tBgB_1(_, k), tBsB_1); }
    if (s == 2) { copy(tma_atom_A.with(storage.full_barrier[2]), tAgA_2(_, k), tAsA_2); copy(tma_atom_B.with(storage.full_barrier[2]), tBgB_2(_, k), tBsB_2); }
    if (s == 3) { copy(tma_atom_A.with(storage.full_barrier[3]), tAgA_3(_, k), tAsA_3); copy(tma_atom_B.with(storage.full_barrier[3]), tBgB_3(_, k), tBsB_3); }
  };

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // =================== WARP 0: PRODUCER ===================
  if (warp_idx == 0 && is_lane0) {
    // Prologue: fill first kStages-1
    for (int k = 0; k < min(kStages - 1, num_k_tiles); ++k) {
      issue_tma(k, k);
    }
    
    // Mainloop: produce remaining
    for (int k = kStages - 1; k < num_k_tiles; ++k) {
      int s = k % kStages;
      // Wait for consumer to release
      cute::wait_barrier(storage.empty_barrier[s], empty_phase[s]);
      empty_phase[s] ^= 1;
      issue_tma(s, k);
    }
  }
  
  // =================== WARP 1: CONSUMER ===================
  else if (warp_idx == 1 && is_lane0) {
    for (int k = 0; k < num_k_tiles; ++k) {
      int s = k % kStages;
      
      // Wait for producer
      cute::wait_barrier(storage.full_barrier[s], full_phase[s]);
      full_phase[s] ^= 1;
      
      auto& tCrA = (s == 0) ? tCrA_0 : (s == 1) ? tCrA_1 : (s == 2) ? tCrA_2 : tCrA_3;
      auto& tCrB = (s == 0) ? tCrB_0 : (s == 1) ? tCrB_1 : (s == 2) ? tCrB_2 : tCrB_3;

      for (int kb = 0; kb < size<2>(tCrA_0); ++kb) {
        gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      
      // Signal stage consumed
      cutlass::arch::umma_arrive(&storage.empty_barrier[s]);
    }
    
    cutlass::arch::umma_arrive(&storage.mma_barrier);
  }
  
  // All warps wait for MMA completion
  cute::wait_barrier(storage.mma_barrier, mma_phase);
  
  // Epilogue - all threads
  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
  Tensor tDrC = make_fragment_like(tDgC);
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  axpby(1.0f, tDrAcc, 0.0f, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();
  if (is_warp0 && is_lane0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_true_warp_spec2_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1));
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16);
  TORCH_CHECK(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto m = a_contig.size(0), k = a_contig.size(1), n = b_contig.size(0);

  TORCH_CHECK(m % 128 == 0 && n % 256 == 0 && k % 64 == 0);

  auto c_buffer = torch::zeros({m, n}, a.options().dtype(torch::kFloat32));
  auto d_buffer = torch::empty_like(c_buffer);

  auto tiled_mma = make_tiled_mma(MmaTag{});
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  using SharedStorageT = TrueWarpSpecSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(a_contig.data_ptr<at::Half>())), make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(b_contig.data_ptr<at::Half>())), make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()), make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(d_buffer.data_ptr<TypeD>()), make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_atom_A = make_tma_atom(TmaLoad{}, mA, sA_layout, select<0, 2>(mma_tiler));
  auto tma_atom_B = make_tma_atom(TmaLoad{}, mB, sB_layout, select<1, 2>(mma_tiler));

  dim3 dimBlock(128);
  dim3 dimGrid((m + size(bM) - 1) / size(bM), (n + size(bN) - 1) / size(bN));
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_true_warp_spec2<SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  kernel_ptr<<<dimGrid, dimBlock, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, mC, mD, mma_tiler, tiled_mma, dimGrid.x, dimGrid.y, tma_atom_A, tma_atom_B);

  return d_buffer.to(torch::kFloat16);
}

}  // namespace

torch::Tensor matmul_tcgen05_true_warp_spec2(torch::Tensor a, torch::Tensor b) {
  return true_warp_spec2_impl::run_true_warp_spec2_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_true_warp_spec2", &matmul_tcgen05_true_warp_spec2);
}

