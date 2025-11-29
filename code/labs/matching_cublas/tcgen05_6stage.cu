/**
 * Stage 12: 6-Stage Deep Pipeline tcgen05 GEMM
 * =============================================
 * 
 * Even deeper pipelining with 6 stages.
 * Prologue fills 5 stages, mainloop keeps 5 ahead.
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

using namespace cute;

namespace deep6_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

constexpr int kStages = 6;  // Very deep pipeline

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct Deep6SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A[kStages];
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B[kStages];
  alignas(16) cute::uint64_t tma_barrier[kStages];
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA(int stage) {
    return make_tensor(make_smem_ptr(smem_A[stage].begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB(int stage) {
    return make_tensor(make_smem_ptr(smem_B[stage].begin()), BSmemLayout{});
  }
};

using MmaTag =
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                         UMMA::Major::K, UMMA::Major::K>;

template <class SharedStorageT,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
__global__ void __launch_bounds__(128, 1)
gemm_6stage(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
            MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
            CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
            CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  
  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  auto mma_coord = make_coord(blockIdx.x, blockIdx.y, _);

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
  if (elect_one_warp) {
    tmem_allocator.allocate(
        decltype(tmem_allocator)::Sm100TmemCapacityColumns,
        &storage.tmem_base_ptr);
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

  // All 6 stages - arrays instead of individual variables
  cute::array<decltype(storage.tensor_sA(0)), kStages> tCsA_arr;
  cute::array<decltype(storage.tensor_sB(0)), kStages> tCsB_arr;
  
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    tCsA_arr[s] = storage.tensor_sA(s);
    tCsB_arr[s] = storage.tensor_sB(s);
  }

  // MMA fragments
  auto tCrA_0 = cta_mma.make_fragment_A(storage.tensor_sA(0));
  auto tCrA_1 = cta_mma.make_fragment_A(storage.tensor_sA(1));
  auto tCrA_2 = cta_mma.make_fragment_A(storage.tensor_sA(2));
  auto tCrA_3 = cta_mma.make_fragment_A(storage.tensor_sA(3));
  auto tCrA_4 = cta_mma.make_fragment_A(storage.tensor_sA(4));
  auto tCrA_5 = cta_mma.make_fragment_A(storage.tensor_sA(5));
  auto tCrB_0 = cta_mma.make_fragment_B(storage.tensor_sB(0));
  auto tCrB_1 = cta_mma.make_fragment_B(storage.tensor_sB(1));
  auto tCrB_2 = cta_mma.make_fragment_B(storage.tensor_sB(2));
  auto tCrB_3 = cta_mma.make_fragment_B(storage.tensor_sB(3));
  auto tCrB_4 = cta_mma.make_fragment_B(storage.tensor_sB(4));
  auto tCrB_5 = cta_mma.make_fragment_B(storage.tensor_sB(5));

  // TMA partitions for stage 0 (to get transaction bytes)
  auto [tAgA_0, tAsA_0] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsA_arr[0]), group_modes<0,3>(tCgCoordA));
  auto [tBgB_0, tBsB_0] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
      group_modes<0,3>(tCsB_arr[0]), group_modes<0,3>(tCgCoordB));

  int tma_bytes = sizeof(make_tensor_like(tAsA_0)) + sizeof(make_tensor_like(tBsB_0));

  // Init barriers
  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(storage.mma_barrier, 1);
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(storage.tma_barrier[s], 1);
    }
  }
  __syncthreads();

  int num_k_tiles = size<3>(tCgA);
  int tma_phase[kStages] = {0};
  int mma_phase = 0;

  // Lambda to issue TMA for any stage
  auto issue_tma = [&](int stage, int k_tile) {
    auto tCsA_s = storage.tensor_sA(stage);
    auto tCsB_s = storage.tensor_sB(stage);
    auto [tAgA_s, tAsA_s] = tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsA_s), group_modes<0,3>(tCgCoordA));
    auto [tBgB_s, tBsB_s] = tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsB_s), group_modes<0,3>(tCgCoordB));
    
    cute::set_barrier_transaction_bytes(storage.tma_barrier[stage], tma_bytes);
    copy(tma_atom_A.with(storage.tma_barrier[stage]), tAgA_s(_, k_tile), tAsA_s);
    copy(tma_atom_B.with(storage.tma_barrier[stage]), tBgB_s(_, k_tile), tBsB_s);
  };

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // Prologue: fill first 5 stages
  if (elect_one_warp && elect_one_thr) {
    for (int s = 0; s < kStages - 1 && s < num_k_tiles; ++s) {
      issue_tma(s, s);
    }
  }

  // Mainloop
  for (int k = 0; k < num_k_tiles; ++k) {
    int curr = k % kStages;
    int next_k = k + (kStages - 1);  // 5 ahead
    int next_s = next_k % kStages;

    cute::wait_barrier(storage.tma_barrier[curr], tma_phase[curr]);
    tma_phase[curr] ^= 1;

    if (next_k < num_k_tiles && elect_one_warp && elect_one_thr) {
      issue_tma(next_s, next_k);
    }

    if (elect_one_warp) {
      auto& tCrA = (curr == 0) ? tCrA_0 : (curr == 1) ? tCrA_1 : (curr == 2) ? tCrA_2 :
                   (curr == 3) ? tCrA_3 : (curr == 4) ? tCrA_4 : tCrA_5;
      auto& tCrB = (curr == 0) ? tCrB_0 : (curr == 1) ? tCrB_1 : (curr == 2) ? tCrB_2 :
                   (curr == 3) ? tCrB_3 : (curr == 4) ? tCrB_4 : tCrB_5;

      for (int kb = 0; kb < size<2>(tCrA_0); ++kb) {
        gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.mma_barrier);
    }

    cute::wait_barrier(storage.mma_barrier, mma_phase);
    mma_phase ^= 1;
  }

  // Epilogue
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
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_6stage_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(1));
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16);
  TORCH_CHECK(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto m = a_contig.size(0);
  auto k = a_contig.size(1);
  auto n = b_contig.size(0);

  TORCH_CHECK(m % 128 == 0 && n % 256 == 0 && k % 64 == 0,
              "Size must be divisible by tcgen05 tile");

  auto options = a.options().dtype(torch::kFloat32);
  auto c_buffer = torch::zeros({m, n}, options);
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

  using SharedStorageT = Deep6SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<TypeA const*>(a_contig.data_ptr<at::Half>())),
      make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<TypeB const*>(b_contig.data_ptr<at::Half>())),
      make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
  Tensor mC = make_tensor(make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
  Tensor mD = make_tensor(make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
      make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

  auto tma_atom_A = make_tma_atom(SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
  auto tma_atom_B = make_tma_atom(SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));

  dim3 dimBlock(128);
  dim3 dimGrid((m + size(bM) - 1) / size(bM), (n + size(bN) - 1) / size(bN));
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_6stage<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  gemm_6stage<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>
      <<<dimGrid, dimBlock, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
          mA, mB, mC, mD, mma_tiler, tiled_mma, tma_atom_A, tma_atom_B);
  AT_CUDA_CHECK(cudaGetLastError());

  return d_buffer.to(torch::kFloat16);
}

}  // namespace deep6_impl

torch::Tensor matmul_tcgen05_6stage(torch::Tensor a, torch::Tensor b) {
  return deep6_impl::run_6stage_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_6stage", &matmul_tcgen05_6stage);
}

