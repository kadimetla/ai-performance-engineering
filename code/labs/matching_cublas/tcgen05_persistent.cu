/**
 * Stage 8: Persistent tcgen05 GEMM Kernel  
 * =======================================
 * 
 * CTAs stay resident and process MULTIPLE output tiles.
 * Barriers initialized ONCE, phase bits continue flipping across tiles.
 * 
 * Key: Don't re-initialize barriers! Just keep flipping phase bits.
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

namespace persistent_impl {

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;
using TypeD = float;
using Accumulator = float;

template <class TypeA_, class TypeB_, class ASmemLayout, class BSmemLayout>
struct PersistentSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA_, cute::cosize_v<ASmemLayout>> smem_A;
  alignas(128) cute::ArrayEngine<TypeB_, cute::cosize_v<BSmemLayout>> smem_B;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE auto tensor_sA() {
    return make_tensor(make_smem_ptr(smem_A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE auto tensor_sB() {
    return make_tensor(make_smem_ptr(smem_B.begin()), BSmemLayout{});
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
gemm_persistent(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
                MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
                int num_tiles_m, int num_tiles_n, int num_k_tiles,
                CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B) {
  
  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  extern __shared__ char shared_memory[];
  SharedStorageT& storage = *reinterpret_cast<SharedStorageT*>(shared_memory);

  // TMEM allocation - ONCE per CTA lifetime
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(
        decltype(tmem_allocator)::Sm100TmemCapacityColumns,
        &storage.tmem_base_ptr);
  }
  __syncthreads();
  uint32_t tmem_base = storage.tmem_base_ptr;

  // Initialize barriers ONCE at the start - NOT per tile!
  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(storage.mma_barrier, 1);
    cute::initialize_barrier(storage.tma_barrier, 1);
  }
  __syncthreads();

  // Phase bits persist across ALL tiles
  int mma_phase = 0;
  int tma_phase = 0;

  auto cta_mma = tiled_mma.get_slice(Int<0>{});
  int total_tiles = num_tiles_m * num_tiles_n;
  int num_ctas = gridDim.x;

  // SMEM tensors - reused for all tiles
  Tensor tCsA = storage.tensor_sA();
  Tensor tCsB = storage.tensor_sB();
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);

  // Work distribution: each CTA processes tiles in stride
  for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += num_ctas) {
    int tile_m = tile_idx % num_tiles_m;
    int tile_n = tile_idx / num_tiles_m;
    
    auto mma_coord = make_coord(tile_m, tile_n, _);

    Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
    Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

    Tensor tCgA = cta_mma.partition_A(gA);
    Tensor tCgB = cta_mma.partition_B(gB);
    Tensor tCgC = cta_mma.partition_C(gC);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
    tCtAcc.data() = tmem_base;

    // TMA coordinates for this tile
    Tensor tma_coord_A = tma_atom_A.get_tma_tensor(shape(mA));
    Tensor tma_coord_B = tma_atom_B.get_tma_tensor(shape(mB));
    Tensor gCoordA = local_tile(tma_coord_A, mma_tiler, mma_coord, Step<_1, X, _1>{});
    Tensor gCoordB = local_tile(tma_coord_B, mma_tiler, mma_coord, Step<X, _1, _1>{});
    Tensor tCgCoordA = cta_mma.partition_A(gCoordA);
    Tensor tCgCoordB = cta_mma.partition_B(gCoordB);

    // TMA partitions
    auto [tAgA, tAsA] = tma_partition(
        tma_atom_A, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsA), group_modes<0,3>(tCgCoordA));
    auto [tBgB, tBsB] = tma_partition(
        tma_atom_B, Int<0>{}, Layout<_1>{},
        group_modes<0,3>(tCsB), group_modes<0,3>(tCgCoordB));

    int tma_bytes = sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    // K-loop for this output tile
    for (int k = 0; k < num_k_tiles; ++k) {
      // Issue TMA loads
      if (elect_one_warp && elect_one_thr) {
        cute::set_barrier_transaction_bytes(storage.tma_barrier, tma_bytes);
        copy(tma_atom_A.with(storage.tma_barrier), tAgA(_, k), tAsA);
        copy(tma_atom_B.with(storage.tma_barrier), tBgB(_, k), tBsB);
      }

      // Wait for TMA
      cute::wait_barrier(storage.tma_barrier, tma_phase);
      tma_phase ^= 1;

      // Compute MMA
      if (elect_one_warp) {
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&storage.mma_barrier);
      }

      // Wait for MMA
      cute::wait_barrier(storage.mma_barrier, mma_phase);
      mma_phase ^= 1;
    }

    // Epilogue: TMEM -> global
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

    // Sync before next tile (ensure epilogue complete)
    __syncthreads();
  }

  // Free TMEM at the very end
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, decltype(tmem_allocator)::Sm100TmemCapacityColumns);
  }
}

torch::Tensor run_persistent_matmul(torch::Tensor a, torch::Tensor b) {
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

  using SharedStorageT = PersistentSharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

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

  // PERSISTENT: Launch exactly num_sms CTAs
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  
  int num_tiles_m = (m + size(bM) - 1) / size(bM);
  int num_tiles_n = (n + size(bN) - 1) / size(bN);
  int num_k_tiles = (k + size(bK) - 1) / size(bK);

  int total_tiles = num_tiles_m * num_tiles_n;
  int num_ctas = std::min(num_sms, total_tiles);

  dim3 dimBlock(128);
  dim3 dimGrid(num_ctas);
  int smem_bytes = sizeof(SharedStorageT);

  auto* kernel_ptr = &gemm_persistent<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>;

  AT_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  gemm_persistent<
      SharedStorageT, decltype(mA), decltype(mB), decltype(mC), decltype(mD),
      decltype(mma_tiler), decltype(tiled_mma), decltype(tma_atom_A), decltype(tma_atom_B)>
      <<<dimGrid, dimBlock, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
          mA, mB, mC, mD, mma_tiler, tiled_mma, num_tiles_m, num_tiles_n, num_k_tiles,
          tma_atom_A, tma_atom_B);
  AT_CUDA_CHECK(cudaGetLastError());

  return d_buffer.to(torch::kFloat16);
}

}  // namespace persistent_impl

torch::Tensor matmul_tcgen05_persistent(torch::Tensor a, torch::Tensor b) {
  return persistent_impl::run_persistent_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_persistent", &matmul_tcgen05_persistent);
}
