#include "jakal_net_native_cuda.h"

#include <cmath>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/autograd/autograd.h>
#include <mma.h>

bool jakal_net_compiled_with_cuda_source() {
  return true;
}

bool jakal_net_query_topk_reduce_cuda_available() {
  return true;
}

bool jakal_net_low_rank_pairwise_topk_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_topk_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_window_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_window_signed_abs_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_causal_dense_signed_abs_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_causal_dense_signed_abs_backward_cuda_available() {
  return true;
}

bool jakal_net_bilinear_propagation_causal_dense_signed_abs_backward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_multihead_max_propagation_causal_dense_signed_abs_cuda_available() {
  return true;
}

bool jakal_net_diagonal_propagation_causal_dense_signed_abs_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_dense_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_dense_scores_tf32_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_dense_tf32_forward_cuda_available() {
  return true;
}

bool jakal_net_low_rank_propagation_window_entmax15_forward_cuda_available() {
  return true;
}

namespace {

constexpr int kPairwiseTopkForwardThreads = 128;
constexpr int kPairwiseTopkForwardMaxK = 64;
constexpr int kDenseForwardValTile = 16;
constexpr float kNegativeInfinity = -INFINITY;
constexpr int kWmmaTf32TileM = 16;
constexpr int kWmmaTf32TileN = 16;
constexpr int kWmmaTf32TileK = 8;



int multihead_causal_dense_threads() {
  int value = kPairwiseTopkForwardThreads;
  if (const char* raw = std::getenv("JAKAL_NET_MULTIHEAD_DENSE_THREADS")) {
    try {
      value = std::stoi(raw);
    } catch (...) {
      value = kPairwiseTopkForwardThreads;
    }
  }
  if (value <= 64) {
    return 64;
  }
  if (value <= 128) {
    return 128;
  }
  if (value <= 256) {
    return 256;
  }
  return 512;
}

int64_t scan_cuda_dense_block_size() {
  if (const char* raw = std::getenv("JAKAL_NET_FUSED_SCAN_DENSE_BLOCK_SIZE")) {
    try {
      return std::max<int64_t>(16, std::stoll(raw));
    } catch (...) {
      return 128;
    }
  }
  return 128;
}

torch::ScalarType scan_cuda_accumulator_dtype(torch::ScalarType input_dtype) {
  if (const char* raw = std::getenv("JAKAL_NET_DENSE_ACCUMULATOR_DTYPE")) {
    const std::string value(raw);
    if (value == "input" || value == "none") {
      return input_dtype;
    }
  }
  if (input_dtype == torch::kFloat16 || input_dtype == torch::kBFloat16) {
    return torch::kFloat32;
  }
  return input_dtype;
}
constexpr int64_t kCompressSoftmax = 0;
constexpr int64_t kCompressSignedAbsSoftmax = 1;
constexpr int64_t kCompressSignedEntmax15 = 2;

__device__ inline void insertion_sort_descending(float* values, int64_t n);
__device__ inline float entmax15_tau_from_shifted(float* shifted_scores, int64_t n);

__device__ inline void insert_descending_topk(
    float score,
    int64_t index,
    float* best_scores,
    int64_t* best_indices,
    int64_t k) {
  if (score <= best_scores[k - 1]) {
    return;
  }
  int64_t write = k - 1;
  while (write > 0 && score > best_scores[write - 1]) {
    best_scores[write] = best_scores[write - 1];
    best_indices[write] = best_indices[write - 1];
    --write;
  }
  best_scores[write] = score;
  best_indices[write] = index;
}

__device__ inline float softsignf_device(float value) {
  return value / (1.0f + fabsf(value));
}

__device__ inline void atomicMaxFloat(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    if (__int_as_float(assumed) >= value) {
      break;
    }
    old = atomicCAS(address_as_int, assumed, __float_as_int(value));
  } while (assumed != old);
}

__global__ void low_rank_dense_scores_tf32_kernel(
    const float* __restrict__ weighted_projected_source,
    const float* __restrict__ projected_target,
    float* __restrict__ scores,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim) {
  using namespace nvcuda;
  const int64_t target_tile = blockIdx.x * kWmmaTf32TileM;
  const int64_t source_tile = blockIdx.y * kWmmaTf32TileN;
  const int64_t batch = blockIdx.z;

  wmma::fragment<wmma::matrix_a, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, wmma::precision::tf32, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  const float* target_base =
      projected_target + (batch * target_nodes + target_tile) * rank_dim;
  const float* source_base =
      weighted_projected_source + (batch * source_nodes + source_tile) * rank_dim;

  for (int64_t rank_start = 0; rank_start < rank_dim; rank_start += kWmmaTf32TileK) {
    wmma::load_matrix_sync(a_frag, target_base + rank_start, rank_dim);
    wmma::load_matrix_sync(b_frag, source_base + rank_start, rank_dim);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  float* score_base = scores + (batch * target_nodes + target_tile) * source_nodes + source_tile;
  wmma::store_matrix_sync(score_base, acc_frag, source_nodes, wmma::mem_row_major);
}

__device__ inline void low_rank_dense_wmma_score_tile(
    const float* __restrict__ weighted_projected_source,
    const float* __restrict__ projected_target,
    float* __restrict__ shared_scores,
    int64_t batch,
    int64_t target_tile,
    int64_t source_tile,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, wmma::precision::tf32, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);
  const float* target_base =
      projected_target + (batch * target_nodes + target_tile) * rank_dim;
  const float* source_base =
      weighted_projected_source + (batch * source_nodes + source_tile) * rank_dim;
  for (int64_t rank_start = 0; rank_start < rank_dim; rank_start += kWmmaTf32TileK) {
    wmma::load_matrix_sync(a_frag, target_base + rank_start, rank_dim);
    wmma::load_matrix_sync(b_frag, source_base + rank_start, rank_dim);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  wmma::store_matrix_sync(shared_scores, acc_frag, kWmmaTf32TileN, wmma::mem_row_major);
}

__global__ void low_rank_propagation_dense_tf32_stats_kernel(
    const float* __restrict__ weighted_projected_source,
    const float* __restrict__ projected_target,
    float* __restrict__ row_max,
    float* __restrict__ row_denom,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    float score_bias) {
  const int tid = threadIdx.x;
  const int64_t target_tile = blockIdx.x * kWmmaTf32TileM;
  const int64_t batch = blockIdx.y;
  extern __shared__ float shared[];
  float* shared_scores = shared;
  float* shared_rows = shared + kWmmaTf32TileM * kWmmaTf32TileN;

  if (tid < kWmmaTf32TileM) {
    shared_rows[tid] = 0.0f;
  }
  __syncthreads();
  for (int64_t source_tile = 0; source_tile < source_nodes; source_tile += kWmmaTf32TileN) {
    if (tid < 32) {
      low_rank_dense_wmma_score_tile(
          weighted_projected_source,
          projected_target,
          shared_scores,
          batch,
          target_tile,
          source_tile,
          target_nodes,
          source_nodes,
          rank_dim);
    }
    __syncthreads();
    if (tid < kWmmaTf32TileM) {
      float row_max_local = shared_rows[tid];
      for (int col = 0; col < kWmmaTf32TileN; ++col) {
        row_max_local = fmaxf(
            row_max_local,
            fabsf(shared_scores[tid * kWmmaTf32TileN + col] + score_bias));
      }
      shared_rows[tid] = row_max_local;
    }
    __syncthreads();
  }
  if (tid < kWmmaTf32TileM) {
    row_max[batch * target_nodes + target_tile + tid] = shared_rows[tid];
    shared_rows[kWmmaTf32TileM + tid] = 0.0f;
  }
  __syncthreads();
  for (int64_t source_tile = 0; source_tile < source_nodes; source_tile += kWmmaTf32TileN) {
    if (tid < 32) {
      low_rank_dense_wmma_score_tile(
          weighted_projected_source,
          projected_target,
          shared_scores,
          batch,
          target_tile,
          source_tile,
          target_nodes,
          source_nodes,
          rank_dim);
    }
    __syncthreads();
    if (tid < kWmmaTf32TileM) {
      const float max_stat = shared_rows[tid];
      float denom = shared_rows[kWmmaTf32TileM + tid];
      for (int col = 0; col < kWmmaTf32TileN; ++col) {
        denom += expf(fabsf(shared_scores[tid * kWmmaTf32TileN + col] + score_bias) - max_stat);
      }
      shared_rows[kWmmaTf32TileM + tid] = denom;
    }
    __syncthreads();
  }
  if (tid < kWmmaTf32TileM) {
    const float denom = shared_rows[kWmmaTf32TileM + tid];
    row_denom[batch * target_nodes + target_tile + tid] = denom > 0.0f ? denom : 1.0f;
  }
}

__global__ void low_rank_propagation_dense_tf32_accum_kernel(
    const float* __restrict__ weighted_projected_source,
    const float* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    float score_bias) {
  const int tid = threadIdx.x;
  const int64_t target_tile = blockIdx.x * kWmmaTf32TileM;
  const int64_t val_tile = blockIdx.y * kWmmaTf32TileN;
  const int64_t batch = blockIdx.z;
  extern __shared__ float shared[];
  float* shared_scores = shared;
  float* shared_state_acc = shared + kWmmaTf32TileM * kWmmaTf32TileN;
  float local_val_acc = 0.0f;
  const int row = tid / kWmmaTf32TileN;
  const int out_col = tid - row * kWmmaTf32TileN;

  if (val_tile == 0 && tid < kWmmaTf32TileM) {
    shared_state_acc[tid] = 0.0f;
  }
  __syncthreads();

  for (int64_t source_tile = 0; source_tile < source_nodes; source_tile += kWmmaTf32TileN) {
    if (tid < 32) {
      low_rank_dense_wmma_score_tile(
          weighted_projected_source,
          projected_target,
          shared_scores,
          batch,
          target_tile,
          source_tile,
          target_nodes,
          source_nodes,
          rank_dim);
    }
    __syncthreads();
    if (tid < kWmmaTf32TileM * kWmmaTf32TileN) {
      const int64_t target = target_tile + row;
      const int64_t out = val_tile + out_col;
      const float max_stat = row_max[batch * target_nodes + target];
      const float denom = row_denom[batch * target_nodes + target];
      for (int col = 0; col < kWmmaTf32TileN; ++col) {
        const int64_t source = source_tile + col;
        const float score = shared_scores[row * kWmmaTf32TileN + col] + score_bias;
        float edge = expf(fabsf(score) - max_stat) / denom;
        if (score < 0.0f) {
          edge = -edge;
        } else if (score == 0.0f) {
          edge = 0.0f;
        }
        if (out < out_dim) {
          local_val_acc += edge * projected_val[(batch * source_nodes + source) * out_dim + out];
        }
        if (val_tile == 0 && out_col == 0) {
          shared_state_acc[row] += edge * projected_state[batch * source_nodes + source];
        }
      }
    }
    __syncthreads();
  }
  if (tid < kWmmaTf32TileM * kWmmaTf32TileN) {
    const int64_t target = target_tile + row;
    const int64_t out = val_tile + out_col;
    if (out < out_dim) {
      delta_val[(batch * target_nodes + target) * out_dim + out] = local_val_acc;
    }
  }
  if (val_tile == 0 && tid < kWmmaTf32TileM) {
    delta_state[batch * target_nodes + target_tile + tid] = shared_state_acc[tid];
  }
}

__global__ void low_rank_propagation_dense_tf32_shared_edges_kernel(
    const float* __restrict__ weighted_projected_source,
    const float* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    float score_bias) {
  const int tid = threadIdx.x;
  const int64_t target_tile = blockIdx.x * kWmmaTf32TileM;
  const int64_t batch = blockIdx.y;
  extern __shared__ float shared[];
  float* shared_edges = shared;
  float* shared_scores = shared_edges + kWmmaTf32TileM * source_nodes;
  float* row_max = shared_scores + kWmmaTf32TileM * kWmmaTf32TileN;
  float* row_denom = row_max + kWmmaTf32TileM;

  for (int row = tid; row < kWmmaTf32TileM; row += blockDim.x) {
    row_max[row] = 0.0f;
    row_denom[row] = 0.0f;
  }
  __syncthreads();

  for (int64_t source_tile = 0; source_tile < source_nodes; source_tile += kWmmaTf32TileN) {
    if (tid < 32) {
      low_rank_dense_wmma_score_tile(
          weighted_projected_source,
          projected_target,
          shared_scores,
          batch,
          target_tile,
          source_tile,
          target_nodes,
          source_nodes,
          rank_dim);
    }
    __syncthreads();
    for (int linear = tid; linear < kWmmaTf32TileM * kWmmaTf32TileN; linear += blockDim.x) {
      const int row = linear / kWmmaTf32TileN;
      const int col = linear - row * kWmmaTf32TileN;
      const int64_t source = source_tile + col;
      const float score = shared_scores[linear] + score_bias;
      shared_edges[row * source_nodes + source] = score;
      atomicMaxFloat(&row_max[row], fabsf(score));
    }
    __syncthreads();
  }

  for (int linear = tid; linear < kWmmaTf32TileM * source_nodes; linear += blockDim.x) {
    const int row = linear / source_nodes;
    float score = shared_edges[linear];
    atomicAdd(&row_denom[row], expf(fabsf(score) - row_max[row]));
  }
  __syncthreads();

  for (int linear = tid; linear < kWmmaTf32TileM * source_nodes; linear += blockDim.x) {
    const int row = linear / source_nodes;
    const float score = shared_edges[linear];
    const float denom = row_denom[row] > 0.0f ? row_denom[row] : 1.0f;
    float edge = expf(fabsf(score) - row_max[row]) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    shared_edges[linear] = edge;
  }
  __syncthreads();

  for (int row = tid; row < kWmmaTf32TileM; row += blockDim.x) {
    float acc = 0.0f;
    for (int64_t source = 0; source < source_nodes; ++source) {
      acc += shared_edges[row * source_nodes + source] *
             projected_state[batch * source_nodes + source];
    }
    delta_state[batch * target_nodes + target_tile + row] = acc;
  }

  if (tid < 32) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, wmma::precision::tf32, wmma::row_major> edge_frag;
    wmma::fragment<wmma::matrix_b, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, wmma::precision::tf32, wmma::row_major> val_frag;
    wmma::fragment<wmma::accumulator, kWmmaTf32TileM, kWmmaTf32TileN, kWmmaTf32TileK, float> acc_frag;
    for (int64_t out_tile = 0; out_tile < out_dim; out_tile += kWmmaTf32TileN) {
      wmma::fill_fragment(acc_frag, 0.0f);
      for (int64_t source_tile = 0; source_tile < source_nodes; source_tile += kWmmaTf32TileK) {
        wmma::load_matrix_sync(
            edge_frag,
            shared_edges + source_tile,
            source_nodes);
        wmma::load_matrix_sync(
            val_frag,
            projected_val + (batch * source_nodes + source_tile) * out_dim + out_tile,
            out_dim);
        wmma::mma_sync(acc_frag, edge_frag, val_frag, acc_frag);
      }
      wmma::store_matrix_sync(
          delta_val + (batch * target_nodes + target_tile) * out_dim + out_tile,
          acc_frag,
          out_dim,
          wmma::mem_row_major);
    }
  }
}

template <typename scalar_t>
__global__ void scan_signed_softmax_state_kernel(
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ delta_state,
    scalar_t* __restrict__ output,
    int64_t batch_flat,
    int64_t nodes) {
  const int64_t batch = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  float local_sum = 0.0f;
  float local_sq_sum = 0.0f;
  const int64_t row_base = batch * nodes;
  for (int64_t node = tid; node < nodes; node += blockDim.x) {
    float value = static_cast<float>(state[row_base + node]) +
                  static_cast<float>(delta_state[row_base + node]);
    if (!isfinite(value)) {
      value = 0.0f;
    }
    local_sum += value;
    local_sq_sum += value * value;
  }
  shared[tid] = local_sum;
  shared[blockDim.x + tid] = local_sq_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
      shared[blockDim.x + tid] += shared[blockDim.x + tid + stride];
    }
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(nodes);
  const float variance = fmaxf(shared[blockDim.x] / static_cast<float>(nodes) - mean * mean, 0.0f);
  const float inv_std = rsqrtf(variance + 1.0e-5f);

  float local_max = 0.0f;
  for (int64_t node = tid; node < nodes; node += blockDim.x) {
    float value = static_cast<float>(state[row_base + node]) +
                  static_cast<float>(delta_state[row_base + node]);
    if (!isfinite(value)) {
      value = 0.0f;
    }
    const float normalized = (value - mean) * inv_std;
    local_max = fmaxf(local_max, fabsf(normalized));
  }
  shared[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  const float max_abs = shared[0];

  float local_denom = 0.0f;
  for (int64_t node = tid; node < nodes; node += blockDim.x) {
    float value = static_cast<float>(state[row_base + node]) +
                  static_cast<float>(delta_state[row_base + node]);
    if (!isfinite(value)) {
      value = 0.0f;
    }
    const float normalized = (value - mean) * inv_std;
    local_denom += expf(fabsf(normalized) - max_abs);
  }
  shared[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float denom = shared[0] > 0.0f ? shared[0] : 1.0f;
  const float state_mass = static_cast<float>(nodes);

  for (int64_t node = tid; node < nodes; node += blockDim.x) {
    float value = static_cast<float>(state[row_base + node]) +
                  static_cast<float>(delta_state[row_base + node]);
    if (!isfinite(value)) {
      value = 0.0f;
    }
    const float normalized = (value - mean) * inv_std;
    float sign = 0.0f;
    if (normalized > 0.0f) {
      sign = 1.0f;
    } else if (normalized < 0.0f) {
      sign = -1.0f;
    }
    output[row_base + node] =
        static_cast<scalar_t>(sign * expf(fabsf(normalized) - max_abs) / denom * state_mass);
  }
}

template <typename scalar_t>
__global__ void scan_val_add_layer_norm_kernel(
    const scalar_t* __restrict__ val,
    const scalar_t* __restrict__ delta_val,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t dim,
    bool has_weight,
    bool has_bias) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  float local_sum = 0.0f;
  float local_sq_sum = 0.0f;
  const int64_t row_base = row * dim;
  for (int64_t feature = tid; feature < dim; feature += blockDim.x) {
    float value = static_cast<float>(val[row_base + feature]) +
                  static_cast<float>(delta_val[row_base + feature]);
    local_sum += value;
    local_sq_sum += value * value;
  }
  shared[tid] = local_sum;
  shared[blockDim.x + tid] = local_sq_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
      shared[blockDim.x + tid] += shared[blockDim.x + tid + stride];
    }
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(dim);
  const float variance = fmaxf(shared[blockDim.x] / static_cast<float>(dim) - mean * mean, 0.0f);
  const float inv_std = rsqrtf(variance + 1.0e-5f);

  for (int64_t feature = tid; feature < dim; feature += blockDim.x) {
    float normalized =
        (static_cast<float>(val[row_base + feature]) +
         static_cast<float>(delta_val[row_base + feature]) - mean) * inv_std;
    if (has_weight) {
      normalized *= static_cast<float>(weight[feature]);
    }
    if (has_bias) {
      normalized += static_cast<float>(bias[feature]);
    }
    output[row_base + feature] = static_cast<scalar_t>(normalized);
  }
}

template <typename scalar_t>
__global__ void scan_bias_gelu_kernel(
    scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ bias,
    int64_t total,
    int64_t dim,
    bool has_bias) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  float value = static_cast<float>(hidden[linear]);
  if (has_bias) {
    value += static_cast<float>(bias[linear % dim]);
  }
  value = 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
  hidden[linear] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void scan_residual_bias_add_kernel(
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ bias,
    int64_t total,
    int64_t dim,
    bool has_bias) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  float value = static_cast<float>(delta[linear]);
  if (has_bias) {
    value += static_cast<float>(bias[linear % dim]);
  }
  delta[linear] = static_cast<scalar_t>(static_cast<float>(residual[linear]) + value);
}

template <typename scalar_t>
__global__ void query_topk_reduce_state_kernel(
    const scalar_t* __restrict__ edges,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ projected_state,
    scalar_t* __restrict__ delta_state,
    int64_t batch_flat,
    int64_t query_nodes,
    int64_t source_nodes,
    int64_t k) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch_flat * query_nodes;
  if (linear >= total) {
    return;
  }

  const int64_t batch = linear / query_nodes;
  const int64_t query = linear - batch * query_nodes;
  const int64_t edge_base = (batch * query_nodes + query) * k;
  const int64_t state_base = batch * source_nodes;

  float acc = 0.0f;
  for (int64_t rank = 0; rank < k; ++rank) {
    const int64_t source_index = indices[edge_base + rank];
    acc += static_cast<float>(edges[edge_base + rank]) *
           static_cast<float>(projected_state[state_base + source_index]);
  }
  delta_state[linear] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void query_topk_reduce_val_kernel(
    const scalar_t* __restrict__ edges,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ projected_val,
    scalar_t* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t query_nodes,
    int64_t source_nodes,
    int64_t out_dim,
    int64_t k) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch_flat * query_nodes * out_dim;
  if (linear >= total) {
    return;
  }

  const int64_t out = linear % out_dim;
  const int64_t query_linear = linear / out_dim;
  const int64_t batch = query_linear / query_nodes;
  const int64_t query = query_linear - batch * query_nodes;
  const int64_t edge_base = (batch * query_nodes + query) * k;
  const int64_t val_base = batch * source_nodes * out_dim;

  float acc = 0.0f;
  for (int64_t rank = 0; rank < k; ++rank) {
    const int64_t source_index = indices[edge_base + rank];
    acc += static_cast<float>(edges[edge_base + rank]) *
           static_cast<float>(projected_val[val_base + source_index * out_dim + out]);
  }
  delta_val[linear] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void low_rank_pairwise_topk_forward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ weighted_projected_state,
    const float* __restrict__ weighted_projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    float* __restrict__ topk_scores,
    int64_t* __restrict__ topk_indices,
    int64_t batch_flat,
    int64_t src_nodes,
    int64_t dst_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t k,
    float score_bias,
    int64_t compress_kind) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / src_nodes;
  const int64_t src = linear_row - batch * src_nodes;
  const int tid = threadIdx.x;

  float local_scores[kPairwiseTopkForwardMaxK];
  int64_t local_indices[kPairwiseTopkForwardMaxK];
  for (int64_t rank = 0; rank < kPairwiseTopkForwardMaxK; ++rank) {
    local_scores[rank] = kNegativeInfinity;
    local_indices[rank] = 0;
  }

  const int64_t source_offset = (batch * src_nodes + src) * rank_dim;
  for (int64_t dst = tid; dst < dst_nodes; dst += blockDim.x) {
    const int64_t target_offset = (batch * dst_nodes + dst) * rank_dim;
    float score = 0.0f;
    for (int64_t feature = 0; feature < rank_dim; ++feature) {
      score += static_cast<float>(weighted_projected_source[source_offset + feature]) *
               static_cast<float>(projected_target[target_offset + feature]);
    }
    score += score_bias;
    insert_descending_topk(score, dst, local_scores, local_indices, k);
  }

  extern __shared__ unsigned char shared_storage[];
  auto* shared_scores = reinterpret_cast<float*>(shared_storage);
  auto* shared_indices = reinterpret_cast<int32_t*>(shared_scores + blockDim.x * k);
  auto* shared_routes = reinterpret_cast<float*>(shared_indices + blockDim.x * k);
  auto* selected_indices = reinterpret_cast<int32_t*>(shared_routes + k);

  const int64_t thread_base = static_cast<int64_t>(tid) * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    shared_scores[thread_base + rank] = local_scores[rank];
    shared_indices[thread_base + rank] = static_cast<int32_t>(local_indices[rank]);
  }
  __syncthreads();

  if (tid == 0) {
    const bool signed_abs_softmax = compress_kind == kCompressSignedAbsSoftmax;
    const bool signed_entmax15 = compress_kind == kCompressSignedEntmax15;
    float merged_scores[kPairwiseTopkForwardMaxK];
    int64_t merged_indices[kPairwiseTopkForwardMaxK];
    for (int64_t rank = 0; rank < k; ++rank) {
      merged_scores[rank] = kNegativeInfinity;
      merged_indices[rank] = 0;
    }
    for (int64_t thread = 0; thread < blockDim.x; ++thread) {
      const int64_t base = thread * k;
      for (int64_t rank = 0; rank < k; ++rank) {
        insert_descending_topk(
            shared_scores[base + rank],
            static_cast<int64_t>(shared_indices[base + rank]),
            merged_scores,
            merged_indices,
            k);
      }
    }

    float denom = 0.0f;
    if (signed_entmax15) {
      float shifted_scores[kPairwiseTopkForwardMaxK];
      float max_abs_score = fabsf(merged_scores[0]);
      for (int64_t rank = 0; rank < k; ++rank) {
        max_abs_score = rank == 0 ? fabsf(merged_scores[rank]) : fmaxf(max_abs_score, fabsf(merged_scores[rank]));
      }
      for (int64_t rank = 0; rank < k; ++rank) {
        shifted_scores[rank] = (fabsf(merged_scores[rank]) - max_abs_score) * 0.5f;
      }
      const float tau = entmax15_tau_from_shifted(shifted_scores, k);
      for (int64_t rank = 0; rank < k; ++rank) {
        const float shifted = (fabsf(merged_scores[rank]) - max_abs_score) * 0.5f;
        const float positive = fmaxf(shifted - tau, 0.0f);
        const float route = positive * positive;
        shared_routes[rank] = route;
        denom += route;
      }
    } else {
      const float max_score = signed_abs_softmax ? fabsf(merged_scores[0]) : merged_scores[0];
      for (int64_t rank = 0; rank < k; ++rank) {
        const float normalized_score =
            signed_abs_softmax ? fabsf(merged_scores[rank]) : merged_scores[rank];
        const float route = expf(normalized_score - max_score);
        shared_routes[rank] = route;
        denom += route;
      }
    }
    denom = denom > 0.0f ? denom : 1.0f;
    const int64_t output_base = (batch * src_nodes + src) * k;
    for (int64_t rank = 0; rank < k; ++rank) {
      shared_routes[rank] /= denom;
      if (signed_abs_softmax || signed_entmax15) {
        if (merged_scores[rank] < 0.0f) {
          shared_routes[rank] = -shared_routes[rank];
        } else if (merged_scores[rank] == 0.0f) {
          shared_routes[rank] = 0.0f;
        }
      }
      topk_scores[output_base + rank] = merged_scores[rank];
      topk_indices[output_base + rank] = merged_indices[rank];
      selected_indices[rank] = static_cast<int32_t>(merged_indices[rank]);
    }

    const float state_contrib = weighted_projected_state[batch * src_nodes + src];
    const int64_t dst_base = batch * dst_nodes;
    for (int64_t rank = 0; rank < k; ++rank) {
      atomicAdd(&delta_state[dst_base + merged_indices[rank]], shared_routes[rank] * state_contrib);
    }
  }
  __syncthreads();

  const int64_t source_val_base = (batch * src_nodes + src) * out_dim;
  const int64_t delta_val_batch_base = batch * dst_nodes * out_dim;
  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    const float sender_val = weighted_projected_val[source_val_base + out];
    for (int64_t rank = 0; rank < k; ++rank) {
      const int64_t dst = static_cast<int64_t>(selected_indices[rank]);
      atomicAdd(
          &delta_val[delta_val_batch_base + dst * out_dim + out],
          shared_routes[rank] * sender_val);
    }
  }
}

template <typename scalar_t>
__global__ void low_rank_pairwise_topk_forward_multihead_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ weighted_projected_state,
    const float* __restrict__ weighted_projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    float* __restrict__ topk_scores,
    int64_t* __restrict__ topk_indices,
    int64_t batch_flat,
    int64_t num_heads,
    int64_t src_nodes,
    int64_t dst_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t k,
    float score_bias,
    int64_t compress_kind) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / src_nodes;
  const int64_t src = linear_row - batch * src_nodes;
  const int tid = threadIdx.x;

  float local_scores[kPairwiseTopkForwardMaxK];
  int64_t local_indices[kPairwiseTopkForwardMaxK];
  for (int64_t rank = 0; rank < kPairwiseTopkForwardMaxK; ++rank) {
    local_scores[rank] = kNegativeInfinity;
    local_indices[rank] = 0;
  }

  for (int64_t dst = tid; dst < dst_nodes; dst += blockDim.x) {
    float best_score = kNegativeInfinity;
    for (int64_t head = 0; head < num_heads; ++head) {
      const int64_t source_offset =
          (((batch * num_heads + head) * src_nodes + src) * rank_dim);
      const int64_t target_offset =
          (((batch * num_heads + head) * dst_nodes + dst) * rank_dim);
      float score = 0.0f;
      for (int64_t feature = 0; feature < rank_dim; ++feature) {
        score += static_cast<float>(weighted_projected_source[source_offset + feature]) *
                 static_cast<float>(projected_target[target_offset + feature]);
      }
      score += score_bias;
      best_score = fmaxf(best_score, score);
    }
    insert_descending_topk(best_score, dst, local_scores, local_indices, k);
  }

  extern __shared__ unsigned char shared_storage[];
  auto* shared_scores = reinterpret_cast<float*>(shared_storage);
  auto* shared_indices = reinterpret_cast<int32_t*>(shared_scores + blockDim.x * k);
  auto* shared_routes = reinterpret_cast<float*>(shared_indices + blockDim.x * k);
  auto* selected_indices = reinterpret_cast<int32_t*>(shared_routes + k);

  const int64_t thread_base = static_cast<int64_t>(tid) * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    shared_scores[thread_base + rank] = local_scores[rank];
    shared_indices[thread_base + rank] = static_cast<int32_t>(local_indices[rank]);
  }
  __syncthreads();

  if (tid == 0) {
    const bool signed_abs_softmax = compress_kind == kCompressSignedAbsSoftmax;
    const bool signed_entmax15 = compress_kind == kCompressSignedEntmax15;
    float merged_scores[kPairwiseTopkForwardMaxK];
    int64_t merged_indices[kPairwiseTopkForwardMaxK];
    for (int64_t rank = 0; rank < k; ++rank) {
      merged_scores[rank] = kNegativeInfinity;
      merged_indices[rank] = 0;
    }
    for (int64_t thread = 0; thread < blockDim.x; ++thread) {
      const int64_t base = thread * k;
      for (int64_t rank = 0; rank < k; ++rank) {
        insert_descending_topk(
            shared_scores[base + rank],
            static_cast<int64_t>(shared_indices[base + rank]),
            merged_scores,
            merged_indices,
            k);
      }
    }

    float denom = 0.0f;
    if (signed_entmax15) {
      float shifted_scores[kPairwiseTopkForwardMaxK];
      float max_abs_score = fabsf(merged_scores[0]);
      for (int64_t rank = 0; rank < k; ++rank) {
        max_abs_score = rank == 0 ? fabsf(merged_scores[rank]) : fmaxf(max_abs_score, fabsf(merged_scores[rank]));
      }
      for (int64_t rank = 0; rank < k; ++rank) {
        shifted_scores[rank] = (fabsf(merged_scores[rank]) - max_abs_score) * 0.5f;
      }
      const float tau = entmax15_tau_from_shifted(shifted_scores, k);
      for (int64_t rank = 0; rank < k; ++rank) {
        const float shifted = (fabsf(merged_scores[rank]) - max_abs_score) * 0.5f;
        const float positive = fmaxf(shifted - tau, 0.0f);
        const float route = positive * positive;
        shared_routes[rank] = route;
        denom += route;
      }
    } else {
      const float max_score = signed_abs_softmax ? fabsf(merged_scores[0]) : merged_scores[0];
      for (int64_t rank = 0; rank < k; ++rank) {
        const float normalized_score =
            signed_abs_softmax ? fabsf(merged_scores[rank]) : merged_scores[rank];
        const float route = expf(normalized_score - max_score);
        shared_routes[rank] = route;
        denom += route;
      }
    }
    denom = denom > 0.0f ? denom : 1.0f;
    const int64_t output_base = (batch * src_nodes + src) * k;
    for (int64_t rank = 0; rank < k; ++rank) {
      shared_routes[rank] /= denom;
      if (signed_abs_softmax || signed_entmax15) {
        if (merged_scores[rank] < 0.0f) {
          shared_routes[rank] = -shared_routes[rank];
        } else if (merged_scores[rank] == 0.0f) {
          shared_routes[rank] = 0.0f;
        }
      }
      topk_scores[output_base + rank] = merged_scores[rank];
      topk_indices[output_base + rank] = merged_indices[rank];
      selected_indices[rank] = static_cast<int32_t>(merged_indices[rank]);
    }

    const float state_contrib = weighted_projected_state[batch * src_nodes + src];
    const int64_t dst_base = batch * dst_nodes;
    for (int64_t rank = 0; rank < k; ++rank) {
      atomicAdd(&delta_state[dst_base + merged_indices[rank]], shared_routes[rank] * state_contrib);
    }
  }
  __syncthreads();

  const int64_t source_val_base = (batch * src_nodes + src) * out_dim;
  const int64_t delta_val_batch_base = batch * dst_nodes * out_dim;
  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    const float sender_val = weighted_projected_val[source_val_base + out];
    for (int64_t rank = 0; rank < k; ++rank) {
      const int64_t dst = static_cast<int64_t>(selected_indices[rank]);
      atomicAdd(
          &delta_val[delta_val_batch_base + dst * out_dim + out],
          shared_routes[rank] * sender_val);
    }
  }
}

template <typename scalar_t>
__global__ void low_rank_propagation_topk_forward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t k,
    float score_bias,
    int64_t compress_kind) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int tid = threadIdx.x;

  float local_scores[kPairwiseTopkForwardMaxK];
  int64_t local_indices[kPairwiseTopkForwardMaxK];
  for (int64_t rank = 0; rank < kPairwiseTopkForwardMaxK; ++rank) {
    local_scores[rank] = kNegativeInfinity;
    local_indices[rank] = 0;
  }

  const int64_t target_offset = (batch * target_nodes + target) * rank_dim;
  for (int64_t source = tid; source < source_nodes; source += blockDim.x) {
    const int64_t source_offset = (batch * source_nodes + source) * rank_dim;
    float score = 0.0f;
    for (int64_t feature = 0; feature < rank_dim; ++feature) {
      score += static_cast<float>(projected_target[target_offset + feature]) *
               static_cast<float>(weighted_projected_source[source_offset + feature]);
    }
    insert_descending_topk(score, source, local_scores, local_indices, k);
  }

  extern __shared__ unsigned char shared_storage[];
  auto* shared_scores = reinterpret_cast<float*>(shared_storage);
  auto* shared_indices = reinterpret_cast<int32_t*>(shared_scores + blockDim.x * k);
  auto* shared_edges = reinterpret_cast<float*>(shared_indices + blockDim.x * k);
  auto* selected_indices = reinterpret_cast<int32_t*>(shared_edges + k);

  const int64_t thread_base = static_cast<int64_t>(tid) * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    shared_scores[thread_base + rank] = local_scores[rank];
    shared_indices[thread_base + rank] = static_cast<int32_t>(local_indices[rank]);
  }
  __syncthreads();

  if (tid == 0) {
    const bool signed_abs_softmax = compress_kind == kCompressSignedAbsSoftmax;
    const bool signed_entmax15 = compress_kind == kCompressSignedEntmax15;
    float merged_scores[kPairwiseTopkForwardMaxK];
    int64_t merged_indices[kPairwiseTopkForwardMaxK];
    for (int64_t rank = 0; rank < k; ++rank) {
      merged_scores[rank] = kNegativeInfinity;
      merged_indices[rank] = 0;
    }
    for (int64_t thread = 0; thread < blockDim.x; ++thread) {
      const int64_t base = thread * k;
      for (int64_t rank = 0; rank < k; ++rank) {
        insert_descending_topk(
            shared_scores[base + rank],
            static_cast<int64_t>(shared_indices[base + rank]),
            merged_scores,
            merged_indices,
            k);
      }
    }

    float state_acc = 0.0f;
    const int64_t state_base = batch * source_nodes;
    if (signed_entmax15) {
      float shifted_scores[kPairwiseTopkForwardMaxK];
      float max_abs_score = fabsf(merged_scores[0] + score_bias);
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        max_abs_score = rank == 0 ? fabsf(signed_score) : fmaxf(max_abs_score, fabsf(signed_score));
      }
      float denom = 0.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        shifted_scores[rank] = (fabsf(merged_scores[rank] + score_bias) - max_abs_score) * 0.5f;
      }
      const float tau = entmax15_tau_from_shifted(shifted_scores, k);
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        const float shifted = (fabsf(signed_score) - max_abs_score) * 0.5f;
        const float positive = fmaxf(shifted - tau, 0.0f);
        const float edge = positive * positive;
        shared_edges[rank] = edge;
        denom += edge;
      }
      denom = denom > 0.0f ? denom : 1.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        float edge = shared_edges[rank] / denom;
        if (signed_score < 0.0f) {
          edge = -edge;
        } else if (signed_score == 0.0f) {
          edge = 0.0f;
        }
        const int64_t source = merged_indices[rank];
        shared_edges[rank] = edge;
        selected_indices[rank] = static_cast<int32_t>(source);
        state_acc += edge * projected_state[state_base + source];
      }
    } else if (signed_abs_softmax) {
      float max_score = fabsf(merged_scores[0] + score_bias);
      float denom = 0.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        const float edge = expf(fabsf(signed_score) - max_score);
        shared_edges[rank] = edge;
        denom += edge;
      }
      denom = denom > 0.0f ? denom : 1.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        float edge = shared_edges[rank] / denom;
        if (signed_score < 0.0f) {
          edge = -edge;
        } else if (signed_score == 0.0f) {
          edge = 0.0f;
        }
        const int64_t source = merged_indices[rank];
        shared_edges[rank] = edge;
        selected_indices[rank] = static_cast<int32_t>(source);
        state_acc += edge * projected_state[state_base + source];
      }
    } else {
      for (int64_t rank = 0; rank < k; ++rank) {
        const float edge = softsignf_device(merged_scores[rank] + score_bias);
        const int64_t source = merged_indices[rank];
        shared_edges[rank] = edge;
        selected_indices[rank] = static_cast<int32_t>(source);
        state_acc += edge * projected_state[state_base + source];
      }
    }
    delta_state[linear_row] = state_acc;
  }
  __syncthreads();

  const int64_t projected_val_base = batch * source_nodes * out_dim;
  const int64_t delta_val_base = linear_row * out_dim;
  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    float acc = 0.0f;
    for (int64_t rank = 0; rank < k; ++rank) {
      const int64_t source = static_cast<int64_t>(selected_indices[rank]);
      acc += shared_edges[rank] * projected_val[projected_val_base + source * out_dim + out];
    }
    delta_val[delta_val_base + out] = acc;
  }
}

template <typename scalar_t>
__global__ void low_rank_propagation_topk_forward_multihead_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t num_heads,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t k,
    float score_bias,
    int64_t compress_kind) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int tid = threadIdx.x;

  float local_scores[kPairwiseTopkForwardMaxK];
  int64_t local_indices[kPairwiseTopkForwardMaxK];
  for (int64_t rank = 0; rank < kPairwiseTopkForwardMaxK; ++rank) {
    local_scores[rank] = kNegativeInfinity;
    local_indices[rank] = 0;
  }

  for (int64_t source = tid; source < source_nodes; source += blockDim.x) {
    float best_score = kNegativeInfinity;
    for (int64_t head = 0; head < num_heads; ++head) {
      const int64_t target_offset =
          (((batch * num_heads + head) * target_nodes + target) * rank_dim);
      const int64_t source_offset =
          (((batch * num_heads + head) * source_nodes + source) * rank_dim);
      float score = 0.0f;
      for (int64_t feature = 0; feature < rank_dim; ++feature) {
        score += static_cast<float>(projected_target[target_offset + feature]) *
                 static_cast<float>(weighted_projected_source[source_offset + feature]);
      }
      best_score = fmaxf(best_score, score);
    }
    insert_descending_topk(best_score, source, local_scores, local_indices, k);
  }

  extern __shared__ unsigned char shared_storage[];
  auto* shared_scores = reinterpret_cast<float*>(shared_storage);
  auto* shared_indices = reinterpret_cast<int32_t*>(shared_scores + blockDim.x * k);
  auto* shared_edges = reinterpret_cast<float*>(shared_indices + blockDim.x * k);
  auto* selected_indices = reinterpret_cast<int32_t*>(shared_edges + k);

  const int64_t thread_base = static_cast<int64_t>(tid) * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    shared_scores[thread_base + rank] = local_scores[rank];
    shared_indices[thread_base + rank] = static_cast<int32_t>(local_indices[rank]);
  }
  __syncthreads();

  if (tid == 0) {
    const bool signed_abs_softmax = compress_kind == kCompressSignedAbsSoftmax;
    const bool signed_entmax15 = compress_kind == kCompressSignedEntmax15;
    float merged_scores[kPairwiseTopkForwardMaxK];
    int64_t merged_indices[kPairwiseTopkForwardMaxK];
    for (int64_t rank = 0; rank < k; ++rank) {
      merged_scores[rank] = kNegativeInfinity;
      merged_indices[rank] = 0;
    }
    for (int64_t thread = 0; thread < blockDim.x; ++thread) {
      const int64_t base = thread * k;
      for (int64_t rank = 0; rank < k; ++rank) {
        insert_descending_topk(
            shared_scores[base + rank],
            static_cast<int64_t>(shared_indices[base + rank]),
            merged_scores,
            merged_indices,
            k);
      }
    }

    float state_acc = 0.0f;
    const int64_t state_base = batch * source_nodes;
    if (signed_entmax15) {
      float shifted_scores[kPairwiseTopkForwardMaxK];
      float max_abs_score = fabsf(merged_scores[0] + score_bias);
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        max_abs_score = rank == 0 ? fabsf(signed_score) : fmaxf(max_abs_score, fabsf(signed_score));
      }
      float denom = 0.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        shifted_scores[rank] = (fabsf(merged_scores[rank] + score_bias) - max_abs_score) * 0.5f;
      }
      const float tau = entmax15_tau_from_shifted(shifted_scores, k);
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        const float shifted = (fabsf(signed_score) - max_abs_score) * 0.5f;
        const float positive = fmaxf(shifted - tau, 0.0f);
        const float edge = positive * positive;
        shared_edges[rank] = edge;
        denom += edge;
      }
      denom = denom > 0.0f ? denom : 1.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        float edge = shared_edges[rank] / denom;
        if (signed_score < 0.0f) {
          edge = -edge;
        } else if (signed_score == 0.0f) {
          edge = 0.0f;
        }
        const int64_t source = merged_indices[rank];
        shared_edges[rank] = edge;
        selected_indices[rank] = static_cast<int32_t>(source);
        state_acc += edge * projected_state[state_base + source];
      }
    } else if (signed_abs_softmax) {
      float max_score = fabsf(merged_scores[0] + score_bias);
      float denom = 0.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        const float edge = expf(fabsf(signed_score) - max_score);
        shared_edges[rank] = edge;
        denom += edge;
      }
      denom = denom > 0.0f ? denom : 1.0f;
      for (int64_t rank = 0; rank < k; ++rank) {
        const float signed_score = merged_scores[rank] + score_bias;
        float edge = shared_edges[rank] / denom;
        if (signed_score < 0.0f) {
          edge = -edge;
        } else if (signed_score == 0.0f) {
          edge = 0.0f;
        }
        const int64_t source = merged_indices[rank];
        shared_edges[rank] = edge;
        selected_indices[rank] = static_cast<int32_t>(source);
        state_acc += edge * projected_state[state_base + source];
      }
    } else {
      for (int64_t rank = 0; rank < k; ++rank) {
        const float edge = softsignf_device(merged_scores[rank] + score_bias);
        const int64_t source = merged_indices[rank];
        shared_edges[rank] = edge;
        selected_indices[rank] = static_cast<int32_t>(source);
        state_acc += edge * projected_state[state_base + source];
      }
    }
    delta_state[linear_row] = state_acc;
  }
  __syncthreads();

  const int64_t projected_val_base = batch * source_nodes * out_dim;
  const int64_t delta_val_base = linear_row * out_dim;
  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    float acc = 0.0f;
    for (int64_t rank = 0; rank < k; ++rank) {
      const int64_t source = static_cast<int64_t>(selected_indices[rank]);
      acc += shared_edges[rank] * projected_val[projected_val_base + source * out_dim + out];
    }
    delta_val[delta_val_base + out] = acc;
  }
}

template <typename scalar_t>
__device__ inline float low_rank_dense_score_at(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    int64_t batch,
    int64_t target,
    int64_t source,
    int64_t num_heads,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    bool multihead) {
  if (multihead) {
    float best_score = kNegativeInfinity;
    for (int64_t head = 0; head < num_heads; ++head) {
      const int64_t target_offset =
          (((batch * num_heads + head) * target_nodes + target) * rank_dim);
      const int64_t source_offset =
          (((batch * num_heads + head) * source_nodes + source) * rank_dim);
      float score = 0.0f;
      for (int64_t feature = 0; feature < rank_dim; ++feature) {
        score += static_cast<float>(projected_target[target_offset + feature]) *
                 static_cast<float>(weighted_projected_source[source_offset + feature]);
      }
      best_score = fmaxf(best_score, score);
    }
    return best_score;
  }
  const int64_t target_offset = (batch * target_nodes + target) * rank_dim;
  const int64_t source_offset = (batch * source_nodes + source) * rank_dim;
  float score = 0.0f;
  for (int64_t feature = 0; feature < rank_dim; ++feature) {
    score += static_cast<float>(projected_target[target_offset + feature]) *
             static_cast<float>(weighted_projected_source[source_offset + feature]);
  }
  return score;
}

template <typename scalar_t>
__global__ void low_rank_propagation_dense_stats_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    float* __restrict__ row_max,
    float* __restrict__ row_denom,
    int64_t batch_flat,
    int64_t num_heads,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t compress_kind,
    bool multihead) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  if (compress_kind != kCompressSignedAbsSoftmax) {
    if (tid == 0) {
      row_max[linear_row] = 0.0f;
      row_denom[linear_row] = 1.0f;
    }
    return;
  }

  float max_stat = compress_kind == kCompressSignedAbsSoftmax ? 0.0f : kNegativeInfinity;
  for (int64_t source = tid; source < source_nodes; source += blockDim.x) {
    max_stat = fmaxf(
        max_stat,
        fabsf(low_rank_dense_score_at(
            weighted_projected_source,
            projected_target,
            batch,
            target,
            source,
            num_heads,
            target_nodes,
            source_nodes,
            rank_dim,
            multihead)));
  }
  shared[tid] = max_stat;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  max_stat = shared[0];

  float local_denom = 0.0f;
  for (int64_t source = tid; source < source_nodes; source += blockDim.x) {
    local_denom += expf(
        fabsf(low_rank_dense_score_at(
            weighted_projected_source,
            projected_target,
            batch,
            target,
            source,
            num_heads,
            target_nodes,
            source_nodes,
            rank_dim,
            multihead)) -
        max_stat);
  }
  shared[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    row_max[linear_row] = max_stat;
    row_denom[linear_row] = shared[0] > 0.0f ? shared[0] : 1.0f;
  }
}

template <typename scalar_t>
__global__ void low_rank_propagation_dense_accum_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t num_heads,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t compress_kind,
    bool multihead) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int64_t tile_start = blockIdx.y * kDenseForwardValTile;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];
  const float max_stat = row_max[linear_row];
  const float denom = row_denom[linear_row];

  float state_acc = 0.0f;
  float val_acc[kDenseForwardValTile];
  for (int64_t tile = 0; tile < kDenseForwardValTile; ++tile) {
    val_acc[tile] = 0.0f;
  }
  const int64_t state_base = batch * source_nodes;
  const int64_t val_base = batch * source_nodes * out_dim;
  for (int64_t source = tid; source < source_nodes; source += blockDim.x) {
    const float score = low_rank_dense_score_at(
        weighted_projected_source,
        projected_target,
        batch,
        target,
        source,
        num_heads,
        target_nodes,
        source_nodes,
        rank_dim,
        multihead);
    float edge;
    if (compress_kind == kCompressSignedAbsSoftmax) {
      edge = expf(fabsf(score) - max_stat) / denom;
      if (score < 0.0f) {
        edge = -edge;
      } else if (score == 0.0f) {
        edge = 0.0f;
      }
    } else {
      edge = softsignf_device(score);
    }
    if (blockIdx.y == 0) {
      state_acc += edge * projected_state[state_base + source];
    }
    for (int64_t tile = 0; tile < kDenseForwardValTile; ++tile) {
      const int64_t out = tile_start + tile;
      if (out < out_dim) {
        val_acc[tile] += edge * projected_val[val_base + source * out_dim + out];
      }
    }
  }

  if (blockIdx.y == 0) {
    shared[tid] = state_acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        shared[tid] += shared[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      delta_state[linear_row] = shared[0];
    }
  } else {
    __syncthreads();
  }

  float* val_shared = shared;
  for (int64_t tile = 0; tile < kDenseForwardValTile; ++tile) {
    val_shared[tile * blockDim.x + tid] = val_acc[tile];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int64_t tile = 0; tile < kDenseForwardValTile; ++tile) {
        val_shared[tile * blockDim.x + tid] += val_shared[tile * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int64_t delta_val_base = linear_row * out_dim;
    for (int64_t tile = 0; tile < kDenseForwardValTile; ++tile) {
      const int64_t out = tile_start + tile;
      if (out < out_dim) {
        delta_val[delta_val_base + out] = val_shared[tile * blockDim.x];
      }
    }
  }
}

template <typename scalar_t>
__global__ void low_rank_propagation_window_forward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t window,
    float score_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int tid = threadIdx.x;

  const int64_t source_start = target > window ? target - window : 0;
  const int64_t source_end = source_nodes < (target + 1) ? source_nodes : (target + 1);
  const int64_t active_sources = source_end > source_start ? (source_end - source_start) : 0;
  const int64_t target_offset = (batch * target_nodes + target) * rank_dim;
  const int64_t state_base = batch * source_nodes;
  const int64_t projected_val_base = batch * source_nodes * out_dim;
  const int64_t delta_val_base = linear_row * out_dim;

  if (tid == 0) {
    float state_acc = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const int64_t source = source_start + offset;
      const int64_t source_offset = (batch * source_nodes + source) * rank_dim;
      float score = 0.0f;
      for (int64_t feature = 0; feature < rank_dim; ++feature) {
        score += static_cast<float>(projected_target[target_offset + feature]) *
                 static_cast<float>(weighted_projected_source[source_offset + feature]);
      }
      const float edge = softsignf_device(score + score_bias);
      state_acc += edge * projected_state[state_base + source];
    }
    delta_state[linear_row] = state_acc;
  }
  __syncthreads();

  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    float acc = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const int64_t source = source_start + offset;
      const int64_t source_offset = (batch * source_nodes + source) * rank_dim;
      float score = 0.0f;
      for (int64_t feature = 0; feature < rank_dim; ++feature) {
        score += static_cast<float>(projected_target[target_offset + feature]) *
                 static_cast<float>(weighted_projected_source[source_offset + feature]);
      }
      const float edge = softsignf_device(score + score_bias);
      acc += edge * projected_val[projected_val_base + source * out_dim + out];
    }
    delta_val[delta_val_base + out] = acc;
  }
}

template <typename scalar_t>
__global__ void low_rank_propagation_window_signed_abs_forward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t window,
    float score_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int tid = threadIdx.x;

  const int64_t source_start = target > window ? target - window : 0;
  const int64_t source_end = source_nodes < (target + 1) ? source_nodes : (target + 1);
  const int64_t active_sources = source_end > source_start ? (source_end - source_start) : 0;
  const int64_t target_offset = (batch * target_nodes + target) * rank_dim;
  const int64_t state_base = batch * source_nodes;
  const int64_t projected_val_base = batch * source_nodes * out_dim;
  const int64_t delta_val_base = linear_row * out_dim;

  extern __shared__ float shared[];
  float* shared_scores = shared;
  float* shared_edges = shared_scores + active_sources;
  float* reduce = shared_edges + active_sources;

  float local_max = 0.0f;
  for (int64_t offset = tid; offset < active_sources; offset += blockDim.x) {
    const int64_t source = source_start + offset;
    const int64_t source_offset = (batch * source_nodes + source) * rank_dim;
    float score = score_bias;
    for (int64_t feature = 0; feature < rank_dim; ++feature) {
      score += static_cast<float>(projected_target[target_offset + feature]) *
               static_cast<float>(weighted_projected_source[source_offset + feature]);
    }
    if (!isfinite(score)) {
      score = 0.0f;
    }
    shared_scores[offset] = score;
    local_max = fmaxf(local_max, fabsf(score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_stat = reduce[0];

  float local_sum = 0.0f;
  for (int64_t offset = tid; offset < active_sources; offset += blockDim.x) {
    local_sum += expf(fabsf(shared_scores[offset]) - max_stat);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  for (int64_t offset = tid; offset < active_sources; offset += blockDim.x) {
    const float score = shared_scores[offset];
    float edge = expf(fabsf(score) - max_stat) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    shared_edges[offset] = edge;
  }
  __syncthreads();

  float state_acc = 0.0f;
  for (int64_t offset = tid; offset < active_sources; offset += blockDim.x) {
    const int64_t source = source_start + offset;
    state_acc += shared_edges[offset] * projected_state[state_base + source];
  }
  reduce[tid] = state_acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    delta_state[linear_row] = reduce[0];
  }
  __syncthreads();

  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    float acc = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const int64_t source = source_start + offset;
      acc += shared_edges[offset] * projected_val[projected_val_base + source * out_dim + out];
    }
    delta_val[delta_val_base + out] = acc;
  }
}

template <typename scalar_t>
__global__ void diagonal_causal_signed_abs_forward_kernel(
    const scalar_t* __restrict__ layer_val,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const scalar_t* __restrict__ normalized_weight,
    const scalar_t* __restrict__ bias,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t nodes,
    int64_t dim,
    bool has_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* reduce = edges + active_sources;
  const int64_t val_base = batch * nodes * dim;
  const int64_t state_base = batch * nodes;
  const int64_t target_offset = val_base + target * dim;

  float local_max = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_offset = val_base + source * dim;
    float score = has_bias ? static_cast<float>(*bias) : 0.0f;
    for (int64_t feature = 0; feature < dim; ++feature) {
      score += static_cast<float>(layer_val[target_offset + feature]) *
               static_cast<float>(normalized_weight[feature]) *
               static_cast<float>(layer_val[source_offset + feature]);
    }
    if (!isfinite(score)) {
      score = 0.0f;
    }
    scores[source] = score;
    local_max = fmaxf(local_max, fabsf(score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    local_sum += expf(fabsf(scores[source]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  float state_acc = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
    state_acc += edge * projected_state[state_base + source];
  }
  reduce[tid] = state_acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    delta_state[linear_row] = reduce[0];
  }
  __syncthreads();

  const int64_t delta_val_base = linear_row * dim;
  for (int64_t feature = tid; feature < dim; feature += blockDim.x) {
    float acc = 0.0f;
    for (int64_t source = 0; source < active_sources; ++source) {
      acc += edges[source] * projected_val[val_base + source * dim + feature];
    }
    delta_val[delta_val_base + feature] = acc;
  }
}

template <typename scalar_t>
__global__ void low_rank_causal_signed_abs_backward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const scalar_t* __restrict__ core_weight,
    const float* __restrict__ grad_delta_state,
    const float* __restrict__ grad_delta_val,
    float* __restrict__ grad_projected_source,
    float* __restrict__ grad_projected_target,
    float* __restrict__ grad_projected_state,
    float* __restrict__ grad_projected_val,
    float* __restrict__ grad_core_weight,
    float* __restrict__ grad_bias,
    int64_t batch_flat,
    int64_t nodes,
    int64_t rank_dim,
    int64_t out_dim,
    float score_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* grad_edges = edges + active_sources;
  float* reduce = grad_edges + active_sources;
  const int64_t rank_base = batch * nodes * rank_dim;
  const int64_t state_base = batch * nodes;
  const int64_t val_base = batch * nodes * out_dim;
  const int64_t target_rank_offset = rank_base + target * rank_dim;
  const int64_t target_val_grad_offset = linear_row * out_dim;

  float local_max = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_rank_offset = rank_base + source * rank_dim;
    float score = score_bias;
    for (int64_t r = 0; r < rank_dim; ++r) {
      score += static_cast<float>(projected_target[target_rank_offset + r]) *
               static_cast<float>(weighted_projected_source[source_rank_offset + r]);
    }
    if (!isfinite(score)) {
      score = 0.0f;
    }
    scores[source] = score;
    local_max = fmaxf(local_max, fabsf(score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    local_sum += expf(fabsf(scores[source]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
  }
  __syncthreads();

  const float g_state = grad_delta_state[linear_row];
  float local_dot = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_val_offset = val_base + source * out_dim;
    float ge = g_state * projected_state[state_base + source];
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      ge += grad_delta_val[target_val_grad_offset + feature] *
            projected_val[source_val_offset + feature];
    }
    grad_edges[source] = ge;
    local_dot += ge * edges[source];
    atomicAdd(grad_projected_state + state_base + source, edges[source] * g_state);
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      atomicAdd(
          grad_projected_val + source_val_offset + feature,
          edges[source] * grad_delta_val[target_val_grad_offset + feature]);
    }
  }
  reduce[tid] = local_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float edge_dot = reduce[0];

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    const float sign = score > 0.0f ? 1.0f : (score < 0.0f ? -1.0f : 0.0f);
    const float prob = fabsf(edges[source]);
    const float grad_score =
        sign == 0.0f ? 0.0f : sign * prob * (sign * grad_edges[source] - edge_dot);
    atomicAdd(grad_bias, grad_score);
    const int64_t source_rank_offset = rank_base + source * rank_dim;
    for (int64_t r = 0; r < rank_dim; ++r) {
      const float target_r = static_cast<float>(projected_target[target_rank_offset + r]);
      const float source_r = static_cast<float>(projected_source[source_rank_offset + r]);
      const float weighted_source_r = static_cast<float>(weighted_projected_source[source_rank_offset + r]);
      const float core = static_cast<float>(core_weight[r]);
      atomicAdd(grad_projected_target + target_rank_offset + r, grad_score * weighted_source_r);
      atomicAdd(grad_projected_source + source_rank_offset + r, grad_score * core * target_r);
      atomicAdd(grad_core_weight + r, grad_score * target_r * source_r);
    }
  }
}



template <typename scalar_t>
__global__ void bilinear_causal_signed_abs_backward_kernel(
    const scalar_t* __restrict__ projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const float* __restrict__ grad_delta_state,
    const float* __restrict__ grad_delta_val,
    float* __restrict__ grad_projected_source,
    float* __restrict__ grad_projected_target,
    float* __restrict__ grad_projected_state,
    float* __restrict__ grad_projected_val,
    float* __restrict__ grad_bias,
    int64_t batch_flat,
    int64_t nodes,
    int64_t dim,
    int64_t out_dim,
    float score_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* grad_edges = edges + active_sources;
  float* reduce = grad_edges + active_sources;
  const int64_t dim_base = batch * nodes * dim;
  const int64_t state_base = batch * nodes;
  const int64_t val_base = batch * nodes * out_dim;
  const int64_t target_dim_offset = dim_base + target * dim;
  const int64_t target_val_grad_offset = linear_row * out_dim;

  float local_max = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_dim_offset = dim_base + source * dim;
    float score = score_bias;
    for (int64_t feature = 0; feature < dim; ++feature) {
      score += static_cast<float>(projected_target[target_dim_offset + feature]) *
               static_cast<float>(projected_source[source_dim_offset + feature]);
    }
    if (!isfinite(score)) {
      score = 0.0f;
    }
    scores[source] = score;
    local_max = fmaxf(local_max, fabsf(score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    local_sum += expf(fabsf(scores[source]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
  }
  __syncthreads();

  const float g_state = grad_delta_state[linear_row];
  float local_dot = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_val_offset = val_base + source * out_dim;
    float ge = g_state * projected_state[state_base + source];
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      ge += grad_delta_val[target_val_grad_offset + feature] * projected_val[source_val_offset + feature];
    }
    grad_edges[source] = ge;
    local_dot += ge * edges[source];
    atomicAdd(grad_projected_state + state_base + source, edges[source] * g_state);
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      atomicAdd(
          grad_projected_val + source_val_offset + feature,
          edges[source] * grad_delta_val[target_val_grad_offset + feature]);
    }
  }
  reduce[tid] = local_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  const float edge_dot = reduce[0];

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    const float sign = score > 0.0f ? 1.0f : (score < 0.0f ? -1.0f : 0.0f);
    const float prob = fabsf(edges[source]);
    const float grad_score = sign == 0.0f ? 0.0f : sign * prob * (sign * grad_edges[source] - edge_dot);
    atomicAdd(grad_bias, grad_score);
    const int64_t source_dim_offset = dim_base + source * dim;
    for (int64_t feature = 0; feature < dim; ++feature) {
      const float target_feature = static_cast<float>(projected_target[target_dim_offset + feature]);
      const float source_feature = static_cast<float>(projected_source[source_dim_offset + feature]);
      atomicAdd(grad_projected_target + target_dim_offset + feature, grad_score * source_feature);
      atomicAdd(grad_projected_source + source_dim_offset + feature, grad_score * target_feature);
    }
  }
}

template <typename scalar_t>
__device__ inline float low_rank_multihead_raw_score(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const scalar_t* __restrict__ biases,
    int64_t heads,
    int64_t batch_flat,
    int64_t nodes,
    int64_t rank_dim,
    int64_t batch,
    int64_t target,
    int64_t source,
    int64_t head,
    bool has_bias) {
  const int64_t head_base = ((head * batch_flat + batch) * nodes) * rank_dim;
  const int64_t target_offset = head_base + target * rank_dim;
  const int64_t source_offset = head_base + source * rank_dim;
  float score = has_bias ? static_cast<float>(biases[head]) : 0.0f;
  for (int64_t r = 0; r < rank_dim; ++r) {
    score += static_cast<float>(projected_target[target_offset + r]) *
             static_cast<float>(weighted_projected_source[source_offset + r]);
  }
  return isfinite(score) ? score : 0.0f;
}

template <typename scalar_t>
__device__ inline float low_rank_multihead_aggregate_score(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const scalar_t* __restrict__ biases,
    int64_t heads,
    int64_t batch_flat,
    int64_t nodes,
    int64_t rank_dim,
    int64_t batch,
    int64_t target,
    int64_t source,
    bool has_bias,
    int aggregate_mode,
    float* shared_head_scores,
    int* selected_head) {
  const bool smoothmax = aggregate_mode == 1;
  const bool signed_smoothmax = aggregate_mode == 2;
  float best_score = kNegativeInfinity;
  float head_max = kNegativeInfinity;
  int best_head = 0;
  for (int64_t head = 0; head < heads; ++head) {
    const float score = low_rank_multihead_raw_score(
        weighted_projected_source,
        projected_target,
        biases,
        heads,
        batch_flat,
        nodes,
        rank_dim,
        batch,
        target,
        source,
        head,
        has_bias);
    if ((smoothmax || signed_smoothmax) && shared_head_scores != nullptr) {
      shared_head_scores[source * heads + head] = score;
    }
    head_max = fmaxf(head_max, signed_smoothmax ? fabsf(score) : score);
    if (score > best_score) {
      best_score = score;
      best_head = static_cast<int>(head);
    }
  }
  if (smoothmax) {
    float head_denom = 0.0f;
    for (int64_t head = 0; head < heads; ++head) {
      const float score = shared_head_scores != nullptr
          ? shared_head_scores[source * heads + head]
          : low_rank_multihead_raw_score(
                weighted_projected_source,
                projected_target,
                biases,
                heads,
                batch_flat,
                nodes,
                rank_dim,
                batch,
                target,
                source,
                head,
                has_bias);
      head_denom += expf(score - head_max);
    }
    best_score = head_max + logf(fmaxf(head_denom, 1.0e-20f)) - logf(static_cast<float>(heads));
    best_head = 0;
  } else if (signed_smoothmax) {
    float head_denom = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t head = 0; head < heads; ++head) {
      const float score = shared_head_scores != nullptr
          ? shared_head_scores[source * heads + head]
          : low_rank_multihead_raw_score(
                weighted_projected_source,
                projected_target,
                biases,
                heads,
                batch_flat,
                nodes,
                rank_dim,
                batch,
                target,
                source,
                head,
                has_bias);
      const float prob_unnorm = expf(fabsf(score) - head_max);
      head_denom += prob_unnorm;
      weighted_sum += score * prob_unnorm;
    }
    best_score = weighted_sum / fmaxf(head_denom, 1.0e-20f);
    best_head = 0;
  }
  *selected_head = best_head;
  return best_score;
}

template <typename scalar_t>
__global__ void low_rank_multihead_max_causal_signed_abs_forward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const scalar_t* __restrict__ biases,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t heads,
    int64_t batch_flat,
    int64_t nodes,
    int64_t rank_dim,
    int64_t out_dim,
    bool has_bias,
    int aggregate_mode) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* reduce = edges + active_sources;
  int* best_heads = reinterpret_cast<int*>(reduce + blockDim.x);
  float* shared_head_scores = reinterpret_cast<float*>(best_heads + active_sources);
  const int64_t state_base = batch * nodes;
  const int64_t val_base = batch * nodes * out_dim;
  const int64_t delta_val_base = linear_row * out_dim;

  float local_max = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    int best_head = 0;
    const float best_score = low_rank_multihead_aggregate_score(
        weighted_projected_source,
        projected_target,
        biases,
        heads,
        batch_flat,
        nodes,
        rank_dim,
        batch,
        target,
        source,
        has_bias,
        aggregate_mode,
        shared_head_scores,
        &best_head);
    scores[source] = best_score;
    best_heads[source] = best_head;
    local_max = fmaxf(local_max, fabsf(best_score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    local_sum += expf(fabsf(scores[source]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  float state_acc = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
    state_acc += edge * projected_state[state_base + source];
  }
  reduce[tid] = state_acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    delta_state[linear_row] = reduce[0];
  }
  __syncthreads();

  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    float acc = 0.0f;
    for (int64_t source = 0; source < active_sources; ++source) {
      acc += edges[source] * projected_val[val_base + source * out_dim + out];
    }
    delta_val[delta_val_base + out] = acc;
  }
}

template <typename scalar_t>
__global__ void low_rank_multihead_max_causal_signed_abs_backward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const scalar_t* __restrict__ core_weights,
    const scalar_t* __restrict__ biases,
    const float* __restrict__ grad_delta_state,
    const float* __restrict__ grad_delta_val,
    float* __restrict__ grad_projected_source,
    float* __restrict__ grad_projected_target,
    float* __restrict__ grad_projected_state,
    float* __restrict__ grad_projected_val,
    float* __restrict__ grad_core_weights,
    float* __restrict__ grad_biases,
    int64_t heads,
    int64_t batch_flat,
    int64_t nodes,
    int64_t rank_dim,
    int64_t out_dim,
    bool has_bias,
    int aggregate_mode) {
  const bool smoothmax = aggregate_mode == 1;
  const bool signed_smoothmax = aggregate_mode == 2;
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* grad_edges = edges + active_sources;
  float* reduce = grad_edges + active_sources;
  int* best_heads = reinterpret_cast<int*>(reduce + blockDim.x);
  float* shared_head_scores = reinterpret_cast<float*>(best_heads + active_sources);
  float* shared_core_grads = shared_head_scores + ((smoothmax || signed_smoothmax) ? active_sources * heads : 0);
  float* shared_bias_grads = shared_core_grads + heads * rank_dim;
  const int64_t state_base = batch * nodes;
  const int64_t val_base = batch * nodes * out_dim;
  const int64_t target_val_grad_offset = linear_row * out_dim;

  for (int64_t index = tid; index < heads * rank_dim; index += blockDim.x) {
    shared_core_grads[index] = 0.0f;
  }
  for (int64_t head = tid; head < heads; head += blockDim.x) {
    shared_bias_grads[head] = 0.0f;
  }
  __syncthreads();

  float local_max = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    int best_head = 0;
    const float best_score = low_rank_multihead_aggregate_score(
        weighted_projected_source,
        projected_target,
        biases,
        heads,
        batch_flat,
        nodes,
        rank_dim,
        batch,
        target,
        source,
        has_bias,
        aggregate_mode,
        shared_head_scores,
        &best_head);
    scores[source] = best_score;
    best_heads[source] = best_head;
    local_max = fmaxf(local_max, fabsf(best_score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    local_sum += expf(fabsf(scores[source]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
  }
  __syncthreads();

  const float g_state = grad_delta_state[linear_row];
  float local_dot = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_val_offset = val_base + source * out_dim;
    float ge = g_state * projected_state[state_base + source];
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      ge += grad_delta_val[target_val_grad_offset + feature] *
            projected_val[source_val_offset + feature];
    }
    grad_edges[source] = ge;
    local_dot += ge * edges[source];
    atomicAdd(grad_projected_state + state_base + source, edges[source] * g_state);
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      atomicAdd(
          grad_projected_val + source_val_offset + feature,
          edges[source] * grad_delta_val[target_val_grad_offset + feature]);
    }
  }
  reduce[tid] = local_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float edge_dot = reduce[0];

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    const float sign = score > 0.0f ? 1.0f : (score < 0.0f ? -1.0f : 0.0f);
    const float prob = fabsf(edges[source]);
    const float grad_score =
        sign == 0.0f ? 0.0f : sign * prob * (sign * grad_edges[source] - edge_dot);
    float head_max = kNegativeInfinity;
    if (smoothmax || signed_smoothmax) {
      for (int64_t head = 0; head < heads; ++head) {
        const float head_score = shared_head_scores[source * heads + head];
        head_max = fmaxf(head_max, signed_smoothmax ? fabsf(head_score) : head_score);
      }
    }
    float head_denom = 1.0f;
    if (smoothmax || signed_smoothmax) {
      head_denom = 0.0f;
      for (int64_t head = 0; head < heads; ++head) {
        const float head_score = shared_head_scores[source * heads + head];
        head_denom += expf((signed_smoothmax ? fabsf(head_score) : head_score) - head_max);
      }
      head_denom = fmaxf(head_denom, 1.0e-20f);
    }
    const int64_t first_head = (smoothmax || signed_smoothmax) ? 0 : static_cast<int64_t>(best_heads[source]);
    const int64_t last_head = (smoothmax || signed_smoothmax) ? heads : first_head + 1;
    for (int64_t head = first_head; head < last_head; ++head) {
      const int64_t head_base = ((head * batch_flat + batch) * nodes) * rank_dim;
      const int64_t target_offset = head_base + target * rank_dim;
      const int64_t source_offset = head_base + source * rank_dim;
      const int64_t weight_offset = head * rank_dim;
      float head_grad = grad_score;
      if (smoothmax) {
        const float head_score = shared_head_scores[source * heads + head];
        head_grad *= expf(head_score - head_max) / head_denom;
      } else if (signed_smoothmax) {
        const float head_score = shared_head_scores[source * heads + head];
        const float head_prob = expf(fabsf(head_score) - head_max) / head_denom;
        const float head_sign = head_score > 0.0f ? 1.0f : (head_score < 0.0f ? -1.0f : 0.0f);
        head_grad *= head_prob * (1.0f + head_sign * (head_score - score));
      }
      if (has_bias) {
        atomicAdd(shared_bias_grads + head, head_grad);
      }
      for (int64_t r = 0; r < rank_dim; ++r) {
        const float target_r = static_cast<float>(projected_target[target_offset + r]);
        const float source_r = static_cast<float>(projected_source[source_offset + r]);
        const float weighted_source_r = static_cast<float>(weighted_projected_source[source_offset + r]);
        const float core = static_cast<float>(core_weights[weight_offset + r]);
        atomicAdd(grad_projected_target + target_offset + r, head_grad * weighted_source_r);
        atomicAdd(grad_projected_source + source_offset + r, head_grad * core * target_r);
        atomicAdd(shared_core_grads + weight_offset + r, head_grad * target_r * source_r);
      }
    }
  }
  __syncthreads();
  for (int64_t index = tid; index < heads * rank_dim; index += blockDim.x) {
    const float value = shared_core_grads[index];
    if (value != 0.0f) {
      atomicAdd(grad_core_weights + index, value);
    }
  }
  if (has_bias) {
    for (int64_t head = tid; head < heads; head += blockDim.x) {
      const float value = shared_bias_grads[head];
      if (value != 0.0f) {
        atomicAdd(grad_biases + head, value);
      }
    }
  }
}

template <typename scalar_t>
__global__ void low_rank_multihead_signed_smoothmax_causal_signed_abs_backward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const scalar_t* __restrict__ core_weights,
    const scalar_t* __restrict__ biases,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ grad_delta_state,
    const float* __restrict__ grad_delta_val,
    float* __restrict__ grad_projected_source,
    float* __restrict__ grad_projected_target,
    float* __restrict__ grad_projected_state,
    float* __restrict__ grad_projected_val,
    float* __restrict__ grad_core_weights,
    float* __restrict__ grad_biases,
    int64_t heads,
    int64_t batch_flat,
    int64_t nodes,
    int64_t rank_dim,
    int64_t out_dim,
    bool has_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* grad_edges = edges + active_sources;
  float* reduce = grad_edges + active_sources;
  float* shared_head_scores = reduce + blockDim.x;
  float* shared_core_grads = shared_head_scores + active_sources * heads;
  float* shared_bias_grads = shared_core_grads + heads * rank_dim;
  const int64_t state_base = batch * nodes;
  const int64_t val_base = batch * nodes * out_dim;
  const int64_t target_val_grad_offset = linear_row * out_dim;
  const float row_max_value = row_max[linear_row];
  const float row_denom_value = fmaxf(row_denom[linear_row], 1.0e-20f);

  for (int64_t index = tid; index < heads * rank_dim; index += blockDim.x) {
    shared_core_grads[index] = 0.0f;
  }
  for (int64_t head = tid; head < heads; head += blockDim.x) {
    shared_bias_grads[head] = 0.0f;
  }
  __syncthreads();

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    float head_max_abs = 0.0f;
    for (int64_t head = 0; head < heads; ++head) {
      const float head_score = low_rank_multihead_raw_score(
          weighted_projected_source,
          projected_target,
          biases,
          heads,
          batch_flat,
          nodes,
          rank_dim,
          batch,
          target,
          source,
          head,
          has_bias);
      shared_head_scores[source * heads + head] = head_score;
      head_max_abs = fmaxf(head_max_abs, fabsf(head_score));
    }
    float head_denom = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t head = 0; head < heads; ++head) {
      const float head_score = shared_head_scores[source * heads + head];
      const float head_prob_unnorm = expf(fabsf(head_score) - head_max_abs);
      head_denom += head_prob_unnorm;
      weighted_sum += head_score * head_prob_unnorm;
    }
    scores[source] = weighted_sum / fmaxf(head_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max_value) / row_denom_value;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
  }
  __syncthreads();

  const float g_state = grad_delta_state[linear_row];
  float local_dot = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_val_offset = val_base + source * out_dim;
    float ge = g_state * projected_state[state_base + source];
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      ge += grad_delta_val[target_val_grad_offset + feature] *
            projected_val[source_val_offset + feature];
    }
    grad_edges[source] = ge;
    local_dot += ge * edges[source];
    atomicAdd(grad_projected_state + state_base + source, edges[source] * g_state);
    for (int64_t feature = 0; feature < out_dim; ++feature) {
      atomicAdd(
          grad_projected_val + source_val_offset + feature,
          edges[source] * grad_delta_val[target_val_grad_offset + feature]);
    }
  }
  reduce[tid] = local_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float edge_dot = reduce[0];

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    const float sign = score > 0.0f ? 1.0f : (score < 0.0f ? -1.0f : 0.0f);
    const float prob = fabsf(edges[source]);
    const float grad_score =
        sign == 0.0f ? 0.0f : sign * prob * (sign * grad_edges[source] - edge_dot);

    float head_max_abs = 0.0f;
    for (int64_t head = 0; head < heads; ++head) {
      head_max_abs = fmaxf(head_max_abs, fabsf(shared_head_scores[source * heads + head]));
    }
    float head_denom = 0.0f;
    for (int64_t head = 0; head < heads; ++head) {
      head_denom += expf(fabsf(shared_head_scores[source * heads + head]) - head_max_abs);
    }
    head_denom = fmaxf(head_denom, 1.0e-20f);

    for (int64_t head = 0; head < heads; ++head) {
      const float head_score = shared_head_scores[source * heads + head];
      const float head_sign = head_score > 0.0f ? 1.0f : (head_score < 0.0f ? -1.0f : 0.0f);
      const float head_prob = expf(fabsf(head_score) - head_max_abs) / head_denom;
      const float head_grad = head_prob * (1.0f + head_sign * (head_score - score)) * grad_score;
      const int64_t head_base = ((head * batch_flat + batch) * nodes) * rank_dim;
      const int64_t target_offset = head_base + target * rank_dim;
      const int64_t source_offset = head_base + source * rank_dim;
      const int64_t weight_offset = head * rank_dim;
      if (has_bias) {
        atomicAdd(shared_bias_grads + head, head_grad);
      }
      for (int64_t r = 0; r < rank_dim; ++r) {
        const float target_r = static_cast<float>(projected_target[target_offset + r]);
        const float source_r = static_cast<float>(projected_source[source_offset + r]);
        const float weighted_source_r = static_cast<float>(weighted_projected_source[source_offset + r]);
        const float core = static_cast<float>(core_weights[weight_offset + r]);
        atomicAdd(grad_projected_target + target_offset + r, head_grad * weighted_source_r);
        atomicAdd(grad_projected_source + source_offset + r, head_grad * core * target_r);
        atomicAdd(shared_core_grads + weight_offset + r, head_grad * target_r * source_r);
      }
    }
  }
  __syncthreads();

  for (int64_t index = tid; index < heads * rank_dim; index += blockDim.x) {
    const float value = shared_core_grads[index];
    if (value != 0.0f) {
      atomicAdd(grad_core_weights + index, value);
    }
  }
  if (has_bias) {
    for (int64_t head = tid; head < heads; head += blockDim.x) {
      const float value = shared_bias_grads[head];
      if (value != 0.0f) {
        atomicAdd(grad_biases + head, value);
      }
    }
  }
}

template <typename scalar_t>
__global__ void diagonal_causal_signed_abs_backward_kernel(
    const scalar_t* __restrict__ layer_val,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    const scalar_t* __restrict__ normalized_weight,
    const scalar_t* __restrict__ bias,
    const float* __restrict__ grad_delta_state,
    const float* __restrict__ grad_delta_val,
    float* __restrict__ grad_layer_val,
    float* __restrict__ grad_projected_state,
    float* __restrict__ grad_projected_val,
    float* __restrict__ grad_normalized_weight,
    float* __restrict__ grad_bias,
    int64_t batch_flat,
    int64_t nodes,
    int64_t dim,
    bool has_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / nodes;
  const int64_t target = linear_row - batch * nodes;
  const int tid = threadIdx.x;
  const int64_t active_sources = target + 1;
  extern __shared__ float shared[];
  float* scores = shared;
  float* edges = scores + active_sources;
  float* grad_edges = edges + active_sources;
  float* reduce = grad_edges + active_sources;
  const int64_t val_base = batch * nodes * dim;
  const int64_t state_base = batch * nodes;
  const int64_t target_offset = val_base + target * dim;

  float local_max = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_offset = val_base + source * dim;
    float score = has_bias ? static_cast<float>(*bias) : 0.0f;
    for (int64_t feature = 0; feature < dim; ++feature) {
      score += static_cast<float>(layer_val[target_offset + feature]) *
               static_cast<float>(normalized_weight[feature]) *
               static_cast<float>(layer_val[source_offset + feature]);
    }
    if (!isfinite(score)) {
      score = 0.0f;
    }
    scores[source] = score;
    local_max = fmaxf(local_max, fabsf(score));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    local_sum += expf(fabsf(scores[source]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0] > 0.0f ? reduce[0] : 1.0f;

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    float edge = expf(fabsf(score) - row_max) / denom;
    if (score < 0.0f) {
      edge = -edge;
    } else if (score == 0.0f) {
      edge = 0.0f;
    }
    edges[source] = edge;
  }
  __syncthreads();

  const float g_state = grad_delta_state[linear_row];
  const int64_t grad_val_base = linear_row * dim;
  float local_dot = 0.0f;
  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const int64_t source_offset = val_base + source * dim;
    float ge = g_state * projected_state[state_base + source];
    for (int64_t feature = 0; feature < dim; ++feature) {
      ge += grad_delta_val[grad_val_base + feature] *
            projected_val[source_offset + feature];
    }
    grad_edges[source] = ge;
    local_dot += ge * edges[source];
    atomicAdd(grad_projected_state + state_base + source, edges[source] * g_state);
    for (int64_t feature = 0; feature < dim; ++feature) {
      atomicAdd(
          grad_projected_val + source_offset + feature,
          edges[source] * grad_delta_val[grad_val_base + feature]);
    }
  }
  reduce[tid] = local_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float edge_dot = reduce[0];

  for (int64_t source = tid; source < active_sources; source += blockDim.x) {
    const float score = scores[source];
    const float sign = score > 0.0f ? 1.0f : (score < 0.0f ? -1.0f : 0.0f);
    const float prob = fabsf(edges[source]);
    const float grad_score =
        sign == 0.0f ? 0.0f : sign * prob * (sign * grad_edges[source] - edge_dot);
    if (has_bias) {
      atomicAdd(grad_bias, grad_score);
    }
    const int64_t source_offset = val_base + source * dim;
    for (int64_t feature = 0; feature < dim; ++feature) {
      const float target_val = static_cast<float>(layer_val[target_offset + feature]);
      const float source_val = static_cast<float>(layer_val[source_offset + feature]);
      const float weight = static_cast<float>(normalized_weight[feature]);
      atomicAdd(grad_layer_val + target_offset + feature, grad_score * weight * source_val);
      atomicAdd(grad_layer_val + source_offset + feature, grad_score * weight * target_val);
      atomicAdd(grad_normalized_weight + feature, grad_score * target_val * source_val);
    }
  }
}

__device__ inline void insertion_sort_descending(float* values, int64_t n) {
  for (int64_t i = 1; i < n; ++i) {
    const float key = values[i];
    int64_t j = i - 1;
    while (j >= 0 && values[j] < key) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = key;
  }
}

__device__ inline float entmax15_tau_from_shifted(float* shifted_scores, int64_t n) {
  insertion_sort_descending(shifted_scores, n);
  float cumulative = 0.0f;
  float cumulative_sq = 0.0f;
  float tau = shifted_scores[0] - 1.0f;
  int64_t support = 1;
  for (int64_t rho = 1; rho <= n; ++rho) {
    const float value = shifted_scores[rho - 1];
    cumulative += value;
    cumulative_sq += value * value;
    const float rho_f = static_cast<float>(rho);
    const float mean = cumulative / rho_f;
    const float mean_sq = cumulative_sq / rho_f;
    const float support_var = rho_f * (mean_sq - mean * mean);
    const float delta = fmaxf((1.0f - support_var) / rho_f, 0.0f);
    const float tau_candidate = mean - sqrtf(delta);
    if (tau_candidate <= value) {
      tau = tau_candidate;
      support = rho;
    }
  }
  (void)support;
  return tau;
}

template <typename scalar_t>
__global__ void low_rank_propagation_window_entmax15_forward_kernel(
    const scalar_t* __restrict__ weighted_projected_source,
    const scalar_t* __restrict__ projected_target,
    const float* __restrict__ projected_state,
    const float* __restrict__ projected_val,
    float* __restrict__ delta_state,
    float* __restrict__ delta_val,
    int64_t batch_flat,
    int64_t target_nodes,
    int64_t source_nodes,
    int64_t rank_dim,
    int64_t out_dim,
    int64_t window,
    float score_bias) {
  const int64_t linear_row = blockIdx.x;
  const int64_t batch = linear_row / target_nodes;
  const int64_t target = linear_row - batch * target_nodes;
  const int tid = threadIdx.x;

  const int64_t source_start = target > window ? target - window : 0;
  const int64_t source_end = source_nodes < (target + 1) ? source_nodes : (target + 1);
  const int64_t active_sources = source_end > source_start ? (source_end - source_start) : 0;
  const int64_t target_offset = (batch * target_nodes + target) * rank_dim;
  const int64_t state_base = batch * source_nodes;
  const int64_t projected_val_base = batch * source_nodes * out_dim;
  const int64_t delta_val_base = linear_row * out_dim;

  extern __shared__ unsigned char shared_storage[];
  auto* shared_scores = reinterpret_cast<float*>(shared_storage);
  auto* shared_edges = shared_scores + active_sources;
  auto* shared_sorted = shared_edges + active_sources;

  if (tid == 0) {
    float max_abs_score = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const int64_t source = source_start + offset;
      const int64_t source_offset = (batch * source_nodes + source) * rank_dim;
      float score = score_bias;
      for (int64_t feature = 0; feature < rank_dim; ++feature) {
        score += static_cast<float>(projected_target[target_offset + feature]) *
                 static_cast<float>(weighted_projected_source[source_offset + feature]);
      }
      shared_scores[offset] = score;
      const float abs_score = fabsf(score);
      max_abs_score = offset == 0 ? abs_score : fmaxf(max_abs_score, abs_score);
    }

    float normalizer = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const float shifted = (fabsf(shared_scores[offset]) - max_abs_score) * 0.5f;
      shared_sorted[offset] = shifted;
    }
    const float tau = entmax15_tau_from_shifted(shared_sorted, active_sources);
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const float shifted = (fabsf(shared_scores[offset]) - max_abs_score) * 0.5f;
      const float positive = fmaxf(shifted - tau, 0.0f);
      const float magnitude = positive * positive;
      shared_edges[offset] = magnitude;
      normalizer += magnitude;
    }
    const float denom = normalizer > 0.0f ? normalizer : 1.0f;
    float state_acc = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      float edge = shared_edges[offset] / denom;
      const float score = shared_scores[offset];
      if (score < 0.0f) {
        edge = -edge;
      } else if (score == 0.0f) {
        edge = 0.0f;
      }
      shared_edges[offset] = edge;
      const int64_t source = source_start + offset;
      state_acc += edge * projected_state[state_base + source];
    }
    delta_state[linear_row] = state_acc;
  }
  __syncthreads();

  for (int64_t out = tid; out < out_dim; out += blockDim.x) {
    float acc = 0.0f;
    for (int64_t offset = 0; offset < active_sources; ++offset) {
      const int64_t source = source_start + offset;
      acc += shared_edges[offset] * projected_val[projected_val_base + source * out_dim + out];
    }
    delta_val[delta_val_base + out] = acc;
  }
}

template <typename scalar_t>
__global__ void query_topk_reduce_backward_state_kernel(
    const scalar_t* __restrict__ edges,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ projected_state,
    const scalar_t* __restrict__ grad_delta_state,
    float* __restrict__ grad_edges,
    float* __restrict__ grad_projected_state,
    int64_t batch_flat,
    int64_t query_nodes,
    int64_t source_nodes,
    int64_t k) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch_flat * query_nodes * k;
  if (linear >= total) {
    return;
  }

  const int64_t rank = linear % k;
  const int64_t query_linear = linear / k;
  const int64_t batch = query_linear / query_nodes;
  const int64_t query = query_linear - batch * query_nodes;
  const int64_t source_index = indices[linear];
  const float grad_state = static_cast<float>(grad_delta_state[query_linear]);
  const float edge = static_cast<float>(edges[linear]);
  const float selected_state =
      static_cast<float>(projected_state[batch * source_nodes + source_index]);

  grad_edges[linear] = grad_state * selected_state;
  atomicAdd(&grad_projected_state[batch * source_nodes + source_index], grad_state * edge);
}

template <typename scalar_t>
__global__ void query_topk_reduce_backward_val_kernel(
    const scalar_t* __restrict__ edges,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ projected_val,
    const scalar_t* __restrict__ grad_delta_val,
    float* __restrict__ grad_edges,
    float* __restrict__ grad_projected_val,
    int64_t batch_flat,
    int64_t query_nodes,
    int64_t source_nodes,
    int64_t out_dim,
    int64_t k) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch_flat * query_nodes * k * out_dim;
  if (linear >= total) {
    return;
  }

  const int64_t out = linear % out_dim;
  const int64_t rank_linear = linear / out_dim;
  const int64_t rank = rank_linear % k;
  const int64_t query_linear = rank_linear / k;
  const int64_t batch = query_linear / query_nodes;
  const int64_t source_index = indices[rank_linear];
  const float grad_val = static_cast<float>(grad_delta_val[query_linear * out_dim + out]);
  const float edge = static_cast<float>(edges[rank_linear]);
  const int64_t source_offset = batch * source_nodes * out_dim + source_index * out_dim + out;

  atomicAdd(&grad_edges[rank_linear], grad_val * static_cast<float>(projected_val[source_offset]));
  atomicAdd(&grad_projected_val[source_offset], grad_val * edge);
}

template <typename scalar_t>
__global__ void softsign_backward_kernel(
    const scalar_t* __restrict__ scores,
    const float* __restrict__ grad_edges,
    float* __restrict__ grad_scores,
    int64_t total) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const float score = static_cast<float>(scores[linear]);
  const float denom = 1.0f + (score < 0.0f ? -score : score);
  grad_scores[linear] = grad_edges[linear] / (denom * denom);
}

template <typename scalar_t>
__global__ void softmax_backward_kernel(
    const scalar_t* __restrict__ routes,
    const float* __restrict__ grad_routes,
    float* __restrict__ grad_scores,
    int64_t rows,
    int64_t k) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  float dot = 0.0f;
  const int64_t base = row * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    dot += static_cast<float>(routes[base + rank]) * grad_routes[base + rank];
  }
  for (int64_t rank = 0; rank < k; ++rank) {
    const int64_t offset = base + rank;
    grad_scores[offset] = static_cast<float>(routes[offset]) * (grad_routes[offset] - dot);
  }
}

template <typename scalar_t>
__global__ void diagonal_pairwise_topk_backward_kernel(
    const scalar_t* __restrict__ query_val,
    const scalar_t* __restrict__ source_val,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ indices,
    const float* __restrict__ grad_scores,
    float* __restrict__ grad_query,
    float* __restrict__ grad_source,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    float inv_temperature,
    int64_t batch_flat,
    int64_t query_nodes,
    int64_t source_nodes,
    int64_t dim,
    int64_t k) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch_flat * query_nodes * k * dim;
  if (linear >= total) {
    return;
  }

  const int64_t d = linear % dim;
  const int64_t rank_linear = linear / dim;
  const int64_t rank = rank_linear % k;
  const int64_t query_linear = rank_linear / k;
  const int64_t batch = query_linear / query_nodes;
  const int64_t query = query_linear - batch * query_nodes;
  const int64_t source_index = indices[rank_linear];
  const float g = grad_scores[rank_linear] * inv_temperature;
  const float q = static_cast<float>(query_val[(batch * query_nodes + query) * dim + d]);
  const float s =
      static_cast<float>(source_val[(batch * source_nodes + source_index) * dim + d]);
  const float w = static_cast<float>(weight[d]);

  atomicAdd(&grad_query[(batch * query_nodes + query) * dim + d], g * s * w);
  atomicAdd(&grad_source[(batch * source_nodes + source_index) * dim + d], g * q * w);
  atomicAdd(&grad_weight[d], g * q * s);
  if (d == 0) {
    atomicAdd(grad_bias, g);
  }
}

template <typename scalar_t>
__global__ void low_rank_pairwise_topk_backward_kernel(
    const scalar_t* __restrict__ query_val,
    const scalar_t* __restrict__ source_val,
    const scalar_t* __restrict__ source_weight,
    const scalar_t* __restrict__ target_weight,
    const scalar_t* __restrict__ core_weight,
    const scalar_t* __restrict__ projected_query,
    const scalar_t* __restrict__ projected_source,
    const int64_t* __restrict__ indices,
    const float* __restrict__ grad_scores,
    float* __restrict__ grad_query,
    float* __restrict__ grad_source,
    float* __restrict__ grad_source_weight,
    float* __restrict__ grad_target_weight,
    float* __restrict__ grad_core_weight,
    float* __restrict__ grad_bias,
    float inv_temperature,
    int64_t batch_flat,
    int64_t query_nodes,
    int64_t source_nodes,
    int64_t dim,
    int64_t rank_dim,
    int64_t k) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch_flat * query_nodes * k * rank_dim * dim;
  if (linear >= total) {
    return;
  }

  const int64_t d = linear % dim;
  const int64_t rank_feature_linear = linear / dim;
  const int64_t r = rank_feature_linear % rank_dim;
  const int64_t topk_linear = rank_feature_linear / rank_dim;
  const int64_t topk_rank = topk_linear % k;
  const int64_t query_linear = topk_linear / k;
  const int64_t batch = query_linear / query_nodes;
  const int64_t query = query_linear - batch * query_nodes;
  const int64_t source_index = indices[topk_linear];
  const float g = grad_scores[topk_linear] * inv_temperature;
  const float projected_q =
      static_cast<float>(projected_query[(batch * query_nodes + query) * rank_dim + r]);
  const float projected_s =
      static_cast<float>(projected_source[(batch * source_nodes + source_index) * rank_dim + r]);
  const float core = static_cast<float>(core_weight[r]);
  const float q = static_cast<float>(query_val[(batch * query_nodes + query) * dim + d]);
  const float s =
      static_cast<float>(source_val[(batch * source_nodes + source_index) * dim + d]);

  const float grad_projected_s = g * core * projected_q;
  const float grad_projected_q = g * core * projected_s;

  atomicAdd(&grad_source[(batch * source_nodes + source_index) * dim + d],
            grad_projected_s * static_cast<float>(source_weight[r * dim + d]));
  atomicAdd(&grad_query[(batch * query_nodes + query) * dim + d],
            grad_projected_q * static_cast<float>(target_weight[r * dim + d]));
  atomicAdd(&grad_source_weight[r * dim + d], grad_projected_s * s);
  atomicAdd(&grad_target_weight[r * dim + d], grad_projected_q * q);
  if (d == 0) {
    atomicAdd(&grad_core_weight[r], g * projected_s * projected_q);
    if (r == 0) {
      atomicAdd(grad_bias, g);
    }
  }
}

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor.");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous.");
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> jakal_net_query_topk_reduce_cuda(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val) {
  require_cuda_contiguous(edges, "edges");
  require_cuda_contiguous(indices, "indices");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (edges.dim() != 3) {
    throw std::runtime_error("edges must be shaped [batch, query_nodes, topk].");
  }
  if (indices.sizes() != edges.sizes()) {
    throw std::runtime_error("indices must have the same shape as edges.");
  }
  if (indices.scalar_type() != torch::kLong) {
    throw std::runtime_error("indices must use torch.long dtype.");
  }
  if (projected_state.dim() != 2) {
    throw std::runtime_error("projected_state must be shaped [batch, source_nodes].");
  }
  if (projected_val.dim() != 3) {
    throw std::runtime_error("projected_val must be shaped [batch, source_nodes, out_dim].");
  }
  if (projected_state.size(0) != edges.size(0) || projected_val.size(0) != edges.size(0)) {
    throw std::runtime_error("projected tensors must share the edge batch dimension.");
  }
  if (projected_state.size(1) != projected_val.size(1)) {
    throw std::runtime_error("projected_state and projected_val must share source_nodes.");
  }
  if (projected_state.scalar_type() != edges.scalar_type() ||
      projected_val.scalar_type() != edges.scalar_type()) {
    throw std::runtime_error("edges and projected tensors must share dtype.");
  }

  const auto batch_flat = edges.size(0);
  const auto query_nodes = edges.size(1);
  const auto k = edges.size(2);
  const auto source_nodes = projected_state.size(1);
  const auto out_dim = projected_val.size(2);

  auto delta_state = torch::empty({batch_flat, query_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, query_nodes, out_dim}, projected_val.options());

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t state_total = batch_flat * query_nodes;
  const int64_t val_total = batch_flat * query_nodes * out_dim;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      edges.scalar_type(),
      "query_topk_reduce_cuda",
      [&] {
        query_topk_reduce_state_kernel<scalar_t>
            <<<(state_total + threads - 1) / threads, threads, 0, stream>>>(
                edges.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                projected_state.data_ptr<scalar_t>(),
                delta_state.data_ptr<scalar_t>(),
                batch_flat,
                query_nodes,
                source_nodes,
                k);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        query_topk_reduce_val_kernel<scalar_t>
            <<<(val_total + threads - 1) / threads, threads, 0, stream>>>(
                edges.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                projected_val.data_ptr<scalar_t>(),
                delta_val.data_ptr<scalar_t>(),
                batch_flat,
                query_nodes,
                source_nodes,
                out_dim,
                k);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_low_rank_pairwise_topk_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& weighted_projected_state,
    const torch::Tensor& weighted_projected_val,
    int64_t topk,
    double score_bias,
    int64_t compress_kind) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(weighted_projected_state, "weighted_projected_state");
  require_cuda_contiguous(weighted_projected_val, "weighted_projected_val");
  if (topk <= 0 || topk > kPairwiseTopkForwardMaxK) {
    throw std::runtime_error("topk must be in the supported fused range [1, 64].");
  }
  if (compress_kind < kCompressSoftmax || compress_kind > kCompressSignedEntmax15) {
    throw std::runtime_error("compress_kind must be 0 (softmax), 1 (signed_abs_softmax), or 2 (signed_entmax15).");
  }
  const bool multihead =
      weighted_projected_source.dim() == 4 && projected_target.dim() == 4;
  if ((!multihead && (weighted_projected_source.dim() != 3 || projected_target.dim() != 3)) ||
      (multihead && (weighted_projected_source.dim() != 4 || projected_target.dim() != 4))) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank] or [batch, heads, nodes, rank].");
  }
  if (weighted_projected_state.dim() != 2 || weighted_projected_val.dim() != 3) {
    throw std::runtime_error(
        "weighted projected tensors must be shaped [batch, src_nodes] and [batch, src_nodes, out_dim].");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != weighted_projected_state.size(0) ||
      weighted_projected_source.size(0) != weighted_projected_val.size(0)) {
    throw std::runtime_error("All fused low-rank inputs must share batch_flat.");
  }
  const auto src_nodes = multihead ? weighted_projected_source.size(2) : weighted_projected_source.size(1);
  const auto dst_nodes = multihead ? projected_target.size(2) : projected_target.size(1);
  const auto rank_dim = multihead ? weighted_projected_source.size(3) : weighted_projected_source.size(2);
  if (src_nodes != weighted_projected_state.size(1) ||
      src_nodes != weighted_projected_val.size(1)) {
    throw std::runtime_error("weighted projected tensors must share src_nodes.");
  }
  if (rank_dim != (multihead ? projected_target.size(3) : projected_target.size(2))) {
    throw std::runtime_error("weighted_projected_source and projected_target must share rank_dim.");
  }
  if (weighted_projected_state.scalar_type() != torch::kFloat32 ||
      weighted_projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("weighted projected state/val must be float32 accumulators.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    projected_target = projected_target.to(weighted_projected_source.scalar_type());
  }

  const auto batch_flat = weighted_projected_source.size(0);
  const auto num_heads = multihead ? weighted_projected_source.size(1) : 1;
  const auto out_dim = weighted_projected_val.size(2);
  const auto k = std::min<int64_t>(topk, dst_nodes);
  if (k <= 0) {
    throw std::runtime_error("dst_nodes must be positive.");
  }

  auto delta_state = torch::zeros({batch_flat, dst_nodes}, weighted_projected_state.options());
  auto delta_val = torch::zeros({batch_flat, dst_nodes, out_dim}, weighted_projected_val.options());
  auto topk_scores = torch::empty({batch_flat, src_nodes, k}, weighted_projected_state.options());
  auto topk_indices =
      torch::empty({batch_flat, src_nodes, k}, weighted_projected_state.options().dtype(torch::kLong));

  const auto blocks = batch_flat * src_nodes;
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem =
      threads * k * (sizeof(float) + sizeof(int32_t)) + k * (sizeof(float) + sizeof(int32_t));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_pairwise_topk_forward_cuda",
      [&] {
        if (multihead) {
          low_rank_pairwise_topk_forward_multihead_kernel<scalar_t>
              <<<blocks, threads, shmem, stream>>>(
                  weighted_projected_source.data_ptr<scalar_t>(),
                  projected_target.data_ptr<scalar_t>(),
                  weighted_projected_state.data_ptr<float>(),
                  weighted_projected_val.data_ptr<float>(),
                  delta_state.data_ptr<float>(),
                  delta_val.data_ptr<float>(),
                  topk_scores.data_ptr<float>(),
                  topk_indices.data_ptr<int64_t>(),
                  batch_flat,
                  num_heads,
                  src_nodes,
                  dst_nodes,
                  rank_dim,
                  out_dim,
                  k,
                  static_cast<float>(score_bias),
                  compress_kind);
        } else {
          low_rank_pairwise_topk_forward_kernel<scalar_t>
              <<<blocks, threads, shmem, stream>>>(
                  weighted_projected_source.data_ptr<scalar_t>(),
                  projected_target.data_ptr<scalar_t>(),
                  weighted_projected_state.data_ptr<float>(),
                  weighted_projected_val.data_ptr<float>(),
                  delta_state.data_ptr<float>(),
                  delta_val.data_ptr<float>(),
                  topk_scores.data_ptr<float>(),
                  topk_indices.data_ptr<int64_t>(),
                  batch_flat,
                  src_nodes,
                  dst_nodes,
                  rank_dim,
                  out_dim,
                  k,
                  static_cast<float>(score_bias),
                  compress_kind);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {delta_state, delta_val, topk_scores, topk_indices};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_topk_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    double score_bias,
    int64_t compress_kind) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (topk <= 0 || topk > kPairwiseTopkForwardMaxK) {
    throw std::runtime_error("topk must be in the supported fused range [1, 64].");
  }
  if (compress_kind < 0 || compress_kind > 2) {
    throw std::runtime_error("compress_kind must be 0 (softsign), 1 (signed_abs_softmax), or 2 (signed_entmax15).");
  }
  const bool multihead =
      weighted_projected_source.dim() == 4 && projected_target.dim() == 4;
  if ((!multihead && (weighted_projected_source.dim() != 3 || projected_target.dim() != 3)) ||
      (multihead && (weighted_projected_source.dim() != 4 || projected_target.dim() != 4))) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank] or [batch, heads, nodes, rank].");
  }
  if (projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "projected_state and projected_val must be shaped [batch, source_nodes] and [batch, source_nodes, out_dim].");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != projected_state.size(0) ||
      weighted_projected_source.size(0) != projected_val.size(0)) {
    throw std::runtime_error("All fused low-rank inputs must share batch_flat.");
  }
  const auto source_nodes = multihead ? weighted_projected_source.size(2) : weighted_projected_source.size(1);
  const auto target_nodes = multihead ? projected_target.size(2) : projected_target.size(1);
  const auto rank_dim = multihead ? weighted_projected_source.size(3) : weighted_projected_source.size(2);
  if (source_nodes != projected_state.size(1) ||
      source_nodes != projected_val.size(1)) {
    throw std::runtime_error("projected_state/projected_val must share source_nodes.");
  }
  if (rank_dim != (multihead ? projected_target.size(3) : projected_target.size(2))) {
    throw std::runtime_error("weighted_projected_source and projected_target must share rank_dim.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state and projected_val must be float32 accumulators.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    projected_target = projected_target.to(weighted_projected_source.scalar_type());
  }

  const auto batch_flat = weighted_projected_source.size(0);
  const auto num_heads = multihead ? weighted_projected_source.size(1) : 1;
  const auto out_dim = projected_val.size(2);
  const auto k = std::min<int64_t>(topk, source_nodes);
  if (k <= 0) {
    throw std::runtime_error("source_nodes must be positive.");
  }

  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());

  const auto blocks = batch_flat * target_nodes;
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem =
      threads * k * (sizeof(float) + sizeof(int32_t)) + k * (sizeof(float) + sizeof(int32_t));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_topk_forward_cuda",
      [&] {
        if (multihead) {
          low_rank_propagation_topk_forward_multihead_kernel<scalar_t>
              <<<blocks, threads, shmem, stream>>>(
                  weighted_projected_source.data_ptr<scalar_t>(),
                  projected_target.data_ptr<scalar_t>(),
                  projected_state.data_ptr<float>(),
                  projected_val.data_ptr<float>(),
                  delta_state.data_ptr<float>(),
                  delta_val.data_ptr<float>(),
                  batch_flat,
                  num_heads,
                  target_nodes,
                  source_nodes,
                  rank_dim,
                  out_dim,
                  k,
                  static_cast<float>(score_bias),
                  compress_kind);
        } else {
          low_rank_propagation_topk_forward_kernel<scalar_t>
              <<<blocks, threads, shmem, stream>>>(
                  weighted_projected_source.data_ptr<scalar_t>(),
                  projected_target.data_ptr<scalar_t>(),
                  projected_state.data_ptr<float>(),
                  projected_val.data_ptr<float>(),
                  delta_state.data_ptr<float>(),
                  delta_val.data_ptr<float>(),
                  batch_flat,
                  target_nodes,
                  source_nodes,
                  rank_dim,
                  out_dim,
                  k,
                  static_cast<float>(score_bias),
                  compress_kind);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_dense_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t compress_kind) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (compress_kind != kCompressSignedAbsSoftmax && compress_kind != 0) {
    throw std::runtime_error("dense fused propagation supports signed_abs_softmax and softsign.");
  }
  const bool multihead =
      weighted_projected_source.dim() == 4 && projected_target.dim() == 4;
  if ((!multihead && (weighted_projected_source.dim() != 3 || projected_target.dim() != 3)) ||
      (multihead && (weighted_projected_source.dim() != 4 || projected_target.dim() != 4))) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank] or [batch, heads, nodes, rank].");
  }
  if (projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "projected_state and projected_val must be shaped [batch, source_nodes] and [batch, source_nodes, out_dim].");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != projected_state.size(0) ||
      weighted_projected_source.size(0) != projected_val.size(0)) {
    throw std::runtime_error("All fused low-rank dense inputs must share batch_flat.");
  }
  const auto source_nodes = multihead ? weighted_projected_source.size(2) : weighted_projected_source.size(1);
  const auto target_nodes = multihead ? projected_target.size(2) : projected_target.size(1);
  const auto rank_dim = multihead ? weighted_projected_source.size(3) : weighted_projected_source.size(2);
  if (source_nodes != projected_state.size(1) ||
      source_nodes != projected_val.size(1)) {
    throw std::runtime_error("projected_state/projected_val must share source_nodes.");
  }
  if (rank_dim != (multihead ? projected_target.size(3) : projected_target.size(2))) {
    throw std::runtime_error("weighted_projected_source and projected_target must share rank_dim.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state and projected_val must be float32 accumulators.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    projected_target = projected_target.to(weighted_projected_source.scalar_type()).contiguous();
  }

  const auto batch_flat = weighted_projected_source.size(0);
  const auto num_heads = multihead ? weighted_projected_source.size(1) : 1;
  const auto out_dim = projected_val.size(2);
  auto row_max = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto row_denom = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());

  constexpr int threads = kPairwiseTopkForwardThreads;
  const dim3 stats_blocks(batch_flat * target_nodes);
  const dim3 accum_blocks(batch_flat * target_nodes, (out_dim + kDenseForwardValTile - 1) / kDenseForwardValTile);
  const auto shmem = std::max<int64_t>(2 * threads, kDenseForwardValTile * threads) * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_dense_forward_cuda",
      [&] {
        low_rank_propagation_dense_stats_kernel<scalar_t>
            <<<stats_blocks, threads, threads * sizeof(float), stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                row_max.data_ptr<float>(),
                row_denom.data_ptr<float>(),
                batch_flat,
                num_heads,
                target_nodes,
                source_nodes,
                rank_dim,
                compress_kind,
                multihead);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        low_rank_propagation_dense_accum_kernel<scalar_t>
            <<<accum_blocks, threads, shmem, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                row_max.data_ptr<float>(),
                row_denom.data_ptr<float>(),
                delta_state.data_ptr<float>(),
                delta_val.data_ptr<float>(),
                batch_flat,
                num_heads,
                target_nodes,
                source_nodes,
                rank_dim,
                out_dim,
                compress_kind,
                multihead);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {delta_state, delta_val};
}

torch::Tensor jakal_net_low_rank_dense_scores_tf32_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "TF32 dense score tile kernel requires [batch, nodes, rank] tensors.");
  }
  if (weighted_projected_source.scalar_type() != torch::kFloat32 ||
      projected_target.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("TF32 dense score tile kernel requires float32 inputs.");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(2) != projected_target.size(2)) {
    throw std::runtime_error("TF32 dense score tile inputs must share batch and rank.");
  }
  const auto batch_flat = weighted_projected_source.size(0);
  const auto source_nodes = weighted_projected_source.size(1);
  const auto target_nodes = projected_target.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
  if (target_nodes % kWmmaTf32TileM != 0 ||
      source_nodes % kWmmaTf32TileN != 0 ||
      rank_dim % kWmmaTf32TileK != 0) {
    throw std::runtime_error(
        "TF32 dense score tile kernel requires target/source multiples of 16 and rank multiple of 8.");
  }
  auto scores = torch::empty(
      {batch_flat, target_nodes, source_nodes},
      projected_target.options().dtype(torch::kFloat32));
  const dim3 blocks(
      target_nodes / kWmmaTf32TileM,
      source_nodes / kWmmaTf32TileN,
      batch_flat);
  const auto stream = at::cuda::getCurrentCUDAStream();
  low_rank_dense_scores_tf32_kernel<<<blocks, 32, 0, stream>>>(
      weighted_projected_source.data_ptr<float>(),
      projected_target.data_ptr<float>(),
      scores.data_ptr<float>(),
      batch_flat,
      target_nodes,
      source_nodes,
      rank_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_dense_tf32_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "TF32 dense propagation kernel requires [batch, nodes, rank] score inputs.");
  }
  if (projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "TF32 dense propagation kernel requires projected_state [batch, nodes] and projected_val [batch, nodes, out_dim].");
  }
  if (weighted_projected_source.scalar_type() != torch::kFloat32 ||
      projected_target.scalar_type() != torch::kFloat32 ||
      projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("TF32 dense propagation kernel requires float32 inputs.");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != projected_state.size(0) ||
      weighted_projected_source.size(0) != projected_val.size(0) ||
      weighted_projected_source.size(1) != projected_state.size(1) ||
      weighted_projected_source.size(1) != projected_val.size(1) ||
      weighted_projected_source.size(2) != projected_target.size(2)) {
    throw std::runtime_error("TF32 dense propagation inputs have incompatible shapes.");
  }
  const auto batch_flat = weighted_projected_source.size(0);
  const auto source_nodes = weighted_projected_source.size(1);
  const auto target_nodes = projected_target.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
  const auto out_dim = projected_val.size(2);
  if (target_nodes % kWmmaTf32TileM != 0 ||
      source_nodes % kWmmaTf32TileN != 0 ||
      rank_dim % kWmmaTf32TileK != 0) {
    throw std::runtime_error(
        "TF32 dense propagation kernel requires target/source multiples of 16 and rank multiple of 8.");
  }
  auto row_max = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto row_denom = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (source_nodes <= 512) {
    const dim3 blocks(target_nodes / kWmmaTf32TileM, batch_flat);
    const auto shmem =
        (kWmmaTf32TileM * source_nodes +
         kWmmaTf32TileM * kWmmaTf32TileN +
         2 * kWmmaTf32TileM) *
        sizeof(float);
    low_rank_propagation_dense_tf32_shared_edges_kernel<<<blocks, 256, shmem, stream>>>(
        weighted_projected_source.data_ptr<float>(),
        projected_target.data_ptr<float>(),
        projected_state.data_ptr<float>(),
        projected_val.data_ptr<float>(),
        delta_state.data_ptr<float>(),
        delta_val.data_ptr<float>(),
        batch_flat,
        target_nodes,
        source_nodes,
        rank_dim,
        out_dim,
        static_cast<float>(score_bias));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {delta_state, delta_val};
  }
  const dim3 stats_blocks(target_nodes / kWmmaTf32TileM, batch_flat);
  const dim3 accum_blocks(
      target_nodes / kWmmaTf32TileM,
      (out_dim + kWmmaTf32TileN - 1) / kWmmaTf32TileN,
      batch_flat);
  const auto stats_shmem =
      (kWmmaTf32TileM * kWmmaTf32TileN + 2 * kWmmaTf32TileM) * sizeof(float);
  const auto accum_shmem =
      (kWmmaTf32TileM * kWmmaTf32TileN + kWmmaTf32TileM) * sizeof(float);
  low_rank_propagation_dense_tf32_stats_kernel<<<stats_blocks, 32, stats_shmem, stream>>>(
      weighted_projected_source.data_ptr<float>(),
      projected_target.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      batch_flat,
      target_nodes,
      source_nodes,
      rank_dim,
      static_cast<float>(score_bias));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  low_rank_propagation_dense_tf32_accum_kernel
      <<<accum_blocks, kWmmaTf32TileM * kWmmaTf32TileN, accum_shmem, stream>>>(
          weighted_projected_source.data_ptr<float>(),
          projected_target.data_ptr<float>(),
          projected_state.data_ptr<float>(),
          projected_val.data_ptr<float>(),
          row_max.data_ptr<float>(),
          row_denom.data_ptr<float>(),
          delta_state.data_ptr<float>(),
          delta_val.data_ptr<float>(),
          batch_flat,
          target_nodes,
          source_nodes,
          rank_dim,
          out_dim,
          static_cast<float>(score_bias));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (window < 0) {
    throw std::runtime_error("window must be non-negative.");
  }
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank].");
  }
  if (projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "projected_state and projected_val must be shaped [batch, source_nodes] and [batch, source_nodes, out_dim].");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != projected_state.size(0) ||
      weighted_projected_source.size(0) != projected_val.size(0)) {
    throw std::runtime_error("All fused low-rank inputs must share batch_flat.");
  }
  if (weighted_projected_source.size(1) != projected_state.size(1) ||
      weighted_projected_source.size(1) != projected_val.size(1)) {
    throw std::runtime_error("projected_state/projected_val must share source_nodes.");
  }
  if (weighted_projected_source.size(2) != projected_target.size(2)) {
    throw std::runtime_error("weighted_projected_source and projected_target must share rank_dim.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state and projected_val must be float32 accumulators.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    projected_target = projected_target.to(weighted_projected_source.scalar_type());
  }

  const auto batch_flat = weighted_projected_source.size(0);
  const auto source_nodes = weighted_projected_source.size(1);
  const auto target_nodes = projected_target.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
  const auto out_dim = projected_val.size(2);

  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());

  const auto blocks = batch_flat * target_nodes;
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_window_forward_cuda",
      [&] {
        low_rank_propagation_window_forward_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                delta_state.data_ptr<float>(),
                delta_val.data_ptr<float>(),
                batch_flat,
                target_nodes,
                source_nodes,
                rank_dim,
                out_dim,
                window,
                static_cast<float>(score_bias));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_signed_abs_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (window < 0) {
    throw std::runtime_error("window must be non-negative.");
  }
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank].");
  }
  if (projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "projected_state and projected_val must be shaped [batch, source_nodes] and [batch, source_nodes, out_dim].");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != projected_state.size(0) ||
      weighted_projected_source.size(0) != projected_val.size(0)) {
    throw std::runtime_error("All fused low-rank inputs must share batch_flat.");
  }
  if (weighted_projected_source.size(1) != projected_state.size(1) ||
      weighted_projected_source.size(1) != projected_val.size(1)) {
    throw std::runtime_error("projected_state/projected_val must share source_nodes.");
  }
  if (weighted_projected_source.size(2) != projected_target.size(2)) {
    throw std::runtime_error("weighted_projected_source and projected_target must share rank_dim.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state and projected_val must be float32 accumulators.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    projected_target = projected_target.to(weighted_projected_source.scalar_type());
  }

  const auto batch_flat = weighted_projected_source.size(0);
  const auto source_nodes = weighted_projected_source.size(1);
  const auto target_nodes = projected_target.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
  const auto out_dim = projected_val.size(2);

  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());

  const auto blocks = batch_flat * target_nodes;
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto width = std::min<int64_t>(window + 1, source_nodes);
  const auto shared_bytes = static_cast<size_t>((2 * width + threads) * sizeof(float));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_window_signed_abs_forward_cuda",
      [&] {
        low_rank_propagation_window_signed_abs_forward_kernel<scalar_t>
            <<<blocks, threads, shared_bytes, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                delta_state.data_ptr<float>(),
                delta_val.data_ptr<float>(),
                batch_flat,
                target_nodes,
                source_nodes,
                rank_dim,
                out_dim,
                window,
                static_cast<float>(score_bias));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_causal_dense_signed_abs_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank].");
  }
  const auto source_nodes = weighted_projected_source.size(1);
  return jakal_net_low_rank_propagation_window_signed_abs_forward_cuda(
      weighted_projected_source,
      projected_target,
      projected_state,
      projected_val,
      source_nodes,
      score_bias);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_causal_dense_signed_abs_backward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& core_weight,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val,
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_source, "projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(core_weight, "core_weight");
  require_cuda_contiguous(grad_delta_state, "grad_delta_state");
  require_cuda_contiguous(grad_delta_val, "grad_delta_val");
  if (weighted_projected_source.dim() != 3 || projected_source.dim() != 3 ||
      projected_target.dim() != 3 || projected_state.dim() != 2 ||
      projected_val.dim() != 3 || grad_delta_state.dim() != 2 ||
      grad_delta_val.dim() != 3) {
    throw std::runtime_error("low-rank causal dense backward received invalid ranks.");
  }
  if (weighted_projected_source.sizes() != projected_source.sizes() ||
      weighted_projected_source.sizes() != projected_target.sizes()) {
    throw std::runtime_error("projected low-rank tensors must share [batch, nodes, rank] shape.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32 ||
      grad_delta_state.scalar_type() != torch::kFloat32 ||
      grad_delta_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected/grad tensors must be float32.");
  }
  if (weighted_projected_source.scalar_type() != projected_source.scalar_type() ||
      weighted_projected_source.scalar_type() != projected_target.scalar_type() ||
      weighted_projected_source.scalar_type() != core_weight.scalar_type()) {
    throw std::runtime_error("low-rank score tensors and core_weight must share dtype.");
  }
  const auto batch_flat = weighted_projected_source.size(0);
  const auto nodes = weighted_projected_source.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
  const auto out_dim = projected_val.size(2);
  if (projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes ||
      grad_delta_state.size(0) != batch_flat || grad_delta_state.size(1) != nodes ||
      grad_delta_val.size(0) != batch_flat || grad_delta_val.size(1) != nodes ||
      grad_delta_val.size(2) != out_dim || core_weight.numel() != rank_dim) {
    throw std::runtime_error("low-rank causal dense backward input shapes are incompatible.");
  }
  auto grad_projected_source = torch::zeros({batch_flat, nodes, rank_dim}, projected_val.options());
  auto grad_projected_target = torch::zeros({batch_flat, nodes, rank_dim}, projected_val.options());
  auto grad_projected_state = torch::zeros({batch_flat, nodes}, projected_state.options());
  auto grad_projected_val = torch::zeros({batch_flat, nodes, out_dim}, projected_val.options());
  auto grad_core_weight = torch::zeros({rank_dim}, projected_val.options());
  auto grad_bias = torch::zeros({1}, projected_val.options());
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem = static_cast<size_t>((3 * nodes + threads) * sizeof(float));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_causal_dense_signed_abs_backward_cuda",
      [&] {
        low_rank_causal_signed_abs_backward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                core_weight.data_ptr<scalar_t>(),
                grad_delta_state.data_ptr<float>(),
                grad_delta_val.data_ptr<float>(),
                grad_projected_source.data_ptr<float>(),
                grad_projected_target.data_ptr<float>(),
                grad_projected_state.data_ptr<float>(),
                grad_projected_val.data_ptr<float>(),
                grad_core_weight.data_ptr<float>(),
                grad_bias.data_ptr<float>(),
                batch_flat,
                nodes,
                rank_dim,
                out_dim,
                static_cast<float>(score_bias));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {
      grad_projected_source,
      grad_projected_target,
      grad_projected_state,
      grad_projected_val,
      grad_core_weight,
      grad_bias};
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_bilinear_propagation_causal_dense_signed_abs_backward_cuda(
    const torch::Tensor& projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val,
    double score_bias) {
  require_cuda_contiguous(projected_source, "projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(grad_delta_state, "grad_delta_state");
  require_cuda_contiguous(grad_delta_val, "grad_delta_val");
  if (projected_source.dim() != 3 || projected_target.dim() != 3 ||
      projected_state.dim() != 2 || projected_val.dim() != 3 ||
      grad_delta_state.dim() != 2 || grad_delta_val.dim() != 3) {
    throw std::runtime_error("bilinear causal dense backward received invalid ranks.");
  }
  if (projected_source.sizes() != projected_target.sizes()) {
    throw std::runtime_error("projected_source and projected_target must share [batch,nodes,dim] shape.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32 ||
      grad_delta_state.scalar_type() != torch::kFloat32 ||
      grad_delta_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected/grad tensors must be float32.");
  }
  if (projected_source.scalar_type() != projected_target.scalar_type()) {
    throw std::runtime_error("projected_source and projected_target must share dtype.");
  }
  const auto batch_flat = projected_source.size(0);
  const auto nodes = projected_source.size(1);
  const auto dim = projected_source.size(2);
  const auto out_dim = projected_val.size(2);
  if (projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes ||
      grad_delta_state.size(0) != batch_flat || grad_delta_state.size(1) != nodes ||
      grad_delta_val.size(0) != batch_flat || grad_delta_val.size(1) != nodes ||
      grad_delta_val.size(2) != out_dim) {
    throw std::runtime_error("bilinear causal dense backward input shapes are incompatible.");
  }
  auto grad_projected_source = torch::zeros({batch_flat, nodes, dim}, projected_val.options());
  auto grad_projected_target = torch::zeros({batch_flat, nodes, dim}, projected_val.options());
  auto grad_projected_state = torch::zeros({batch_flat, nodes}, projected_state.options());
  auto grad_projected_val = torch::zeros({batch_flat, nodes, out_dim}, projected_val.options());
  auto grad_bias = torch::zeros({1}, projected_val.options());
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem = static_cast<size_t>((3 * nodes + threads) * sizeof(float));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      projected_source.scalar_type(),
      "bilinear_propagation_causal_dense_signed_abs_backward_cuda",
      [&] {
        bilinear_causal_signed_abs_backward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                grad_delta_state.data_ptr<float>(),
                grad_delta_val.data_ptr<float>(),
                grad_projected_source.data_ptr<float>(),
                grad_projected_target.data_ptr<float>(),
                grad_projected_state.data_ptr<float>(),
                grad_projected_val.data_ptr<float>(),
                grad_bias.data_ptr<float>(),
                batch_flat,
                nodes,
                dim,
                out_dim,
                static_cast<float>(score_bias));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_projected_source, grad_projected_target, grad_projected_state, grad_projected_val, grad_bias};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_multihead_max_propagation_causal_dense_signed_abs_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& biases,
    bool has_bias,
    const std::string& aggregate) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (has_bias) {
    require_cuda_contiguous(biases, "biases");
  }
  if (weighted_projected_source.dim() != 4 || projected_target.dim() != 4 ||
      projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error("multihead low-rank causal dense forward received invalid ranks.");
  }
  if (weighted_projected_source.sizes() != projected_target.sizes()) {
    throw std::runtime_error("weighted_projected_source and projected_target must share [heads,batch,nodes,rank].");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state/projected_val must be float32.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    throw std::runtime_error("multihead projected score tensors must share dtype.");
  }
  const auto heads = weighted_projected_source.size(0);
  const auto batch_flat = weighted_projected_source.size(1);
  const auto nodes = weighted_projected_source.size(2);
  const auto rank_dim = weighted_projected_source.size(3);
  const auto out_dim = projected_val.size(2);
  if (heads <= 0 || projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes) {
    throw std::runtime_error("multihead low-rank causal dense forward input shapes are incompatible.");
  }
  if (has_bias && (biases.dim() != 1 || biases.size(0) != heads ||
                   biases.scalar_type() != weighted_projected_source.scalar_type())) {
    throw std::runtime_error("biases must be [heads] and share score dtype.");
  }
  int aggregate_mode = 0;
  if (aggregate == "smoothmax") {
    aggregate_mode = 1;
  } else if (aggregate == "signed_smoothmax") {
    aggregate_mode = 2;
  } else if (aggregate != "max") {
    throw std::runtime_error("CUDA multihead low-rank fused path supports max, smoothmax, and signed_smoothmax aggregates.");
  }
  auto delta_state = torch::empty({batch_flat, nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, nodes, out_dim}, projected_val.options());
  const int threads = multihead_causal_dense_threads();
  const auto extra_head_scores = aggregate_mode != 0 ? heads * nodes * static_cast<int64_t>(sizeof(float)) : 0;
  const auto shmem = static_cast<size_t>(
      (2 * nodes + threads) * sizeof(float) +
      nodes * sizeof(int) +
      extra_head_scores);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_multihead_max_propagation_causal_dense_signed_abs_forward_cuda",
      [&] {
        low_rank_multihead_max_causal_signed_abs_forward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                has_bias ? biases.data_ptr<scalar_t>() : nullptr,
                delta_state.data_ptr<float>(),
                delta_val.data_ptr<float>(),
                heads,
                batch_flat,
                nodes,
                rank_dim,
                out_dim,
                has_bias,
                aggregate_mode);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_low_rank_multihead_max_propagation_causal_dense_signed_abs_backward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& core_weights,
    const torch::Tensor& biases,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val,
    bool has_bias,
    const std::string& aggregate) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_source, "projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(core_weights, "core_weights");
  require_cuda_contiguous(grad_delta_state, "grad_delta_state");
  require_cuda_contiguous(grad_delta_val, "grad_delta_val");
  if (has_bias) {
    require_cuda_contiguous(biases, "biases");
  }
  if (weighted_projected_source.dim() != 4 || projected_source.dim() != 4 ||
      projected_target.dim() != 4 || projected_state.dim() != 2 || projected_val.dim() != 3 ||
      grad_delta_state.dim() != 2 || grad_delta_val.dim() != 3 || core_weights.dim() != 2) {
    throw std::runtime_error("multihead low-rank causal dense backward received invalid ranks.");
  }
  if (weighted_projected_source.sizes() != projected_source.sizes() ||
      weighted_projected_source.sizes() != projected_target.sizes()) {
    throw std::runtime_error("multihead projected tensors must share [heads,batch,nodes,rank].");
  }
  if (projected_state.scalar_type() != torch::kFloat32 || projected_val.scalar_type() != torch::kFloat32 ||
      grad_delta_state.scalar_type() != torch::kFloat32 || grad_delta_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected/grad tensors must be float32.");
  }
  if (weighted_projected_source.scalar_type() != projected_source.scalar_type() ||
      weighted_projected_source.scalar_type() != projected_target.scalar_type() ||
      weighted_projected_source.scalar_type() != core_weights.scalar_type()) {
    throw std::runtime_error("multihead low-rank score tensors and core weights must share dtype.");
  }
  const auto heads = weighted_projected_source.size(0);
  const auto batch_flat = weighted_projected_source.size(1);
  const auto nodes = weighted_projected_source.size(2);
  const auto rank_dim = weighted_projected_source.size(3);
  const auto out_dim = projected_val.size(2);
  if (heads <= 0 || core_weights.size(0) != heads || core_weights.size(1) != rank_dim ||
      projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes ||
      grad_delta_state.size(0) != batch_flat || grad_delta_state.size(1) != nodes ||
      grad_delta_val.size(0) != batch_flat || grad_delta_val.size(1) != nodes ||
      grad_delta_val.size(2) != out_dim) {
    throw std::runtime_error("multihead low-rank causal dense backward input shapes are incompatible.");
  }
  if (has_bias && (biases.dim() != 1 || biases.size(0) != heads ||
                   biases.scalar_type() != weighted_projected_source.scalar_type())) {
    throw std::runtime_error("biases must be [heads] and share score dtype.");
  }
  int aggregate_mode = 0;
  if (aggregate == "smoothmax") {
    aggregate_mode = 1;
  } else if (aggregate == "signed_smoothmax") {
    aggregate_mode = 2;
  } else if (aggregate != "max") {
    throw std::runtime_error("CUDA multihead low-rank fused path supports max, smoothmax, and signed_smoothmax aggregates.");
  }
  auto grad_projected_source = torch::zeros({heads, batch_flat, nodes, rank_dim}, projected_val.options());
  auto grad_projected_target = torch::zeros({heads, batch_flat, nodes, rank_dim}, projected_val.options());
  auto grad_projected_state = torch::zeros({batch_flat, nodes}, projected_state.options());
  auto grad_projected_val = torch::zeros({batch_flat, nodes, out_dim}, projected_val.options());
  auto grad_core_weights = torch::zeros({heads, rank_dim}, projected_val.options());
  auto grad_biases = torch::zeros({heads}, projected_val.options());
  const int threads = multihead_causal_dense_threads();
  const auto extra_head_scores = aggregate_mode != 0 ? heads * nodes * static_cast<int64_t>(sizeof(float)) : 0;
  const auto extra_block_grads =
      (heads * rank_dim + heads) * static_cast<int64_t>(sizeof(float));
  const auto shmem = static_cast<size_t>(
      (3 * nodes + threads) * sizeof(float) +
      nodes * sizeof(int) +
      extra_head_scores +
      extra_block_grads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_multihead_max_propagation_causal_dense_signed_abs_backward_cuda",
      [&] {
        low_rank_multihead_max_causal_signed_abs_backward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                core_weights.data_ptr<scalar_t>(),
                has_bias ? biases.data_ptr<scalar_t>() : nullptr,
                grad_delta_state.data_ptr<float>(),
                grad_delta_val.data_ptr<float>(),
                grad_projected_source.data_ptr<float>(),
                grad_projected_target.data_ptr<float>(),
                grad_projected_state.data_ptr<float>(),
                grad_projected_val.data_ptr<float>(),
                grad_core_weights.data_ptr<float>(),
                grad_biases.data_ptr<float>(),
                heads,
                batch_flat,
                nodes,
                rank_dim,
                out_dim,
                has_bias,
                aggregate_mode);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_projected_source, grad_projected_target, grad_projected_state, grad_projected_val, grad_core_weights, grad_biases};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_low_rank_multihead_signed_smoothmax_propagation_causal_dense_signed_abs_backward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& core_weights,
    const torch::Tensor& biases,
    const torch::Tensor& row_max,
    const torch::Tensor& row_denom,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val,
    bool has_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_source, "projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(core_weights, "core_weights");
  require_cuda_contiguous(row_max, "row_max");
  require_cuda_contiguous(row_denom, "row_denom");
  require_cuda_contiguous(grad_delta_state, "grad_delta_state");
  require_cuda_contiguous(grad_delta_val, "grad_delta_val");
  if (has_bias) {
    require_cuda_contiguous(biases, "biases");
  }
  if (weighted_projected_source.dim() != 4 || projected_source.dim() != 4 ||
      projected_target.dim() != 4 || projected_state.dim() != 2 || projected_val.dim() != 3 ||
      row_max.dim() != 2 || row_denom.dim() != 2 ||
      grad_delta_state.dim() != 2 || grad_delta_val.dim() != 3 || core_weights.dim() != 2) {
    throw std::runtime_error("multihead low-rank signed_smoothmax backward received invalid ranks.");
  }
  if (weighted_projected_source.sizes() != projected_source.sizes() ||
      weighted_projected_source.sizes() != projected_target.sizes()) {
    throw std::runtime_error("multihead projected tensors must share [heads,batch,nodes,rank].");
  }
  if (projected_state.scalar_type() != torch::kFloat32 || projected_val.scalar_type() != torch::kFloat32 ||
      row_max.scalar_type() != torch::kFloat32 || row_denom.scalar_type() != torch::kFloat32 ||
      grad_delta_state.scalar_type() != torch::kFloat32 || grad_delta_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected/row/grad tensors must be float32.");
  }
  if (weighted_projected_source.scalar_type() != projected_source.scalar_type() ||
      weighted_projected_source.scalar_type() != projected_target.scalar_type() ||
      weighted_projected_source.scalar_type() != core_weights.scalar_type()) {
    throw std::runtime_error("multihead low-rank score tensors and core weights must share dtype.");
  }
  const auto heads = weighted_projected_source.size(0);
  const auto batch_flat = weighted_projected_source.size(1);
  const auto nodes = weighted_projected_source.size(2);
  const auto rank_dim = weighted_projected_source.size(3);
  const auto out_dim = projected_val.size(2);
  if (heads <= 0 || core_weights.size(0) != heads || core_weights.size(1) != rank_dim ||
      projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes ||
      row_max.size(0) != batch_flat || row_max.size(1) != nodes ||
      row_denom.size(0) != batch_flat || row_denom.size(1) != nodes ||
      grad_delta_state.size(0) != batch_flat || grad_delta_state.size(1) != nodes ||
      grad_delta_val.size(0) != batch_flat || grad_delta_val.size(1) != nodes ||
      grad_delta_val.size(2) != out_dim) {
    throw std::runtime_error("multihead low-rank signed_smoothmax backward input shapes are incompatible.");
  }
  if (has_bias && (biases.dim() != 1 || biases.size(0) != heads ||
                   biases.scalar_type() != weighted_projected_source.scalar_type())) {
    throw std::runtime_error("biases must be [heads] and share score dtype.");
  }

  auto grad_projected_source = torch::zeros({heads, batch_flat, nodes, rank_dim}, projected_val.options());
  auto grad_projected_target = torch::zeros({heads, batch_flat, nodes, rank_dim}, projected_val.options());
  auto grad_projected_state = torch::zeros({batch_flat, nodes}, projected_state.options());
  auto grad_projected_val = torch::zeros({batch_flat, nodes, out_dim}, projected_val.options());
  auto grad_core_weights = torch::zeros({heads, rank_dim}, projected_val.options());
  auto grad_biases = torch::zeros({heads}, projected_val.options());
  const int threads = multihead_causal_dense_threads();
  const auto extra_head_scores = heads * nodes * static_cast<int64_t>(sizeof(float));
  const auto extra_block_grads =
      (heads * rank_dim + heads) * static_cast<int64_t>(sizeof(float));
  const auto shmem = static_cast<size_t>(
      (3 * nodes + threads) * sizeof(float) +
      extra_head_scores +
      extra_block_grads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_multihead_signed_smoothmax_propagation_causal_dense_signed_abs_backward_cuda",
      [&] {
        low_rank_multihead_signed_smoothmax_causal_signed_abs_backward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                core_weights.data_ptr<scalar_t>(),
                has_bias ? biases.data_ptr<scalar_t>() : nullptr,
                row_max.data_ptr<float>(),
                row_denom.data_ptr<float>(),
                grad_delta_state.data_ptr<float>(),
                grad_delta_val.data_ptr<float>(),
                grad_projected_source.data_ptr<float>(),
                grad_projected_target.data_ptr<float>(),
                grad_projected_state.data_ptr<float>(),
                grad_projected_val.data_ptr<float>(),
                grad_core_weights.data_ptr<float>(),
                grad_biases.data_ptr<float>(),
                heads,
                batch_flat,
                nodes,
                rank_dim,
                out_dim,
                has_bias);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_projected_source, grad_projected_target, grad_projected_state, grad_projected_val, grad_core_weights, grad_biases};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_diagonal_propagation_causal_dense_signed_abs_forward_cuda(
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& normalized_weight,
    const torch::Tensor& bias) {
  require_cuda_contiguous(layer_val, "layer_val");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(normalized_weight, "normalized_weight");
  if (bias.defined() && bias.numel() != 0) {
    require_cuda_contiguous(bias, "bias");
  }
  if (layer_val.dim() != 3 || projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "diagonal causal dense forward expects layer_val [batch, nodes, dim], projected_state [batch, nodes], projected_val [batch, nodes, dim].");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state/projected_val must be float32.");
  }
  if (layer_val.scalar_type() != normalized_weight.scalar_type()) {
    throw std::runtime_error("layer_val and normalized_weight must share dtype.");
  }
  const auto batch_flat = layer_val.size(0);
  const auto nodes = layer_val.size(1);
  const auto dim = layer_val.size(2);
  if (projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes ||
      projected_val.size(2) != dim || normalized_weight.numel() != dim) {
    throw std::runtime_error("diagonal causal dense forward input shapes are incompatible.");
  }
  const bool has_bias = bias.defined() && bias.numel() != 0;
  if (has_bias && (bias.numel() != 1 || bias.scalar_type() != layer_val.scalar_type())) {
    throw std::runtime_error("bias must be a scalar with layer_val dtype.");
  }
  auto delta_state = torch::empty({batch_flat, nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, nodes, dim}, projected_val.options());
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem = static_cast<size_t>((2 * nodes + threads) * sizeof(float));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      layer_val.scalar_type(),
      "diagonal_propagation_causal_dense_signed_abs_forward_cuda",
      [&] {
        diagonal_causal_signed_abs_forward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                layer_val.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                normalized_weight.data_ptr<scalar_t>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                delta_state.data_ptr<float>(),
                delta_val.data_ptr<float>(),
                batch_flat,
                nodes,
                dim,
                has_bias);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_diagonal_propagation_causal_dense_signed_abs_backward_cuda(
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& normalized_weight,
    const torch::Tensor& bias,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val) {
  require_cuda_contiguous(layer_val, "layer_val");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(normalized_weight, "normalized_weight");
  require_cuda_contiguous(grad_delta_state, "grad_delta_state");
  require_cuda_contiguous(grad_delta_val, "grad_delta_val");
  if (bias.defined() && bias.numel() != 0) {
    require_cuda_contiguous(bias, "bias");
  }
  if (layer_val.dim() != 3 || projected_state.dim() != 2 || projected_val.dim() != 3 ||
      grad_delta_state.dim() != 2 || grad_delta_val.dim() != 3) {
    throw std::runtime_error("diagonal causal dense backward received invalid ranks.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32 ||
      grad_delta_state.scalar_type() != torch::kFloat32 ||
      grad_delta_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected/grad tensors must be float32.");
  }
  if (layer_val.scalar_type() != normalized_weight.scalar_type()) {
    throw std::runtime_error("layer_val and normalized_weight must share dtype.");
  }
  const auto batch_flat = layer_val.size(0);
  const auto nodes = layer_val.size(1);
  const auto dim = layer_val.size(2);
  if (projected_state.size(0) != batch_flat || projected_state.size(1) != nodes ||
      projected_val.size(0) != batch_flat || projected_val.size(1) != nodes ||
      projected_val.size(2) != dim ||
      grad_delta_state.size(0) != batch_flat || grad_delta_state.size(1) != nodes ||
      grad_delta_val.size(0) != batch_flat || grad_delta_val.size(1) != nodes ||
      grad_delta_val.size(2) != dim || normalized_weight.numel() != dim) {
    throw std::runtime_error("diagonal causal dense backward input shapes are incompatible.");
  }
  const bool has_bias = bias.defined() && bias.numel() != 0;
  if (has_bias && (bias.numel() != 1 || bias.scalar_type() != layer_val.scalar_type())) {
    throw std::runtime_error("bias must be a scalar with layer_val dtype.");
  }
  auto grad_layer_val = torch::zeros({batch_flat, nodes, dim}, projected_val.options());
  auto grad_projected_state = torch::zeros({batch_flat, nodes}, projected_state.options());
  auto grad_projected_val = torch::zeros({batch_flat, nodes, dim}, projected_val.options());
  auto grad_normalized_weight = torch::zeros({dim}, projected_val.options());
  auto grad_bias = torch::zeros({1}, projected_val.options());
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem = static_cast<size_t>((3 * nodes + threads) * sizeof(float));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      layer_val.scalar_type(),
      "diagonal_propagation_causal_dense_signed_abs_backward_cuda",
      [&] {
        diagonal_causal_signed_abs_backward_kernel<scalar_t>
            <<<batch_flat * nodes, threads, shmem, stream>>>(
                layer_val.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                normalized_weight.data_ptr<scalar_t>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                grad_delta_state.data_ptr<float>(),
                grad_delta_val.data_ptr<float>(),
                grad_layer_val.data_ptr<float>(),
                grad_projected_state.data_ptr<float>(),
                grad_projected_val.data_ptr<float>(),
                grad_normalized_weight.data_ptr<float>(),
                grad_bias.data_ptr<float>(),
                batch_flat,
                nodes,
                dim,
                has_bias);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_layer_val, grad_projected_state, grad_projected_val, grad_normalized_weight, grad_bias};
}

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_entmax15_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (window < 0) {
    throw std::runtime_error("window must be non-negative.");
  }
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank].");
  }
  if (projected_state.dim() != 2 || projected_val.dim() != 3) {
    throw std::runtime_error(
        "projected_state and projected_val must be shaped [batch, source_nodes] and [batch, source_nodes, out_dim].");
  }
  if (weighted_projected_source.size(0) != projected_target.size(0) ||
      weighted_projected_source.size(0) != projected_state.size(0) ||
      weighted_projected_source.size(0) != projected_val.size(0)) {
    throw std::runtime_error("All fused low-rank inputs must share batch_flat.");
  }
  if (weighted_projected_source.size(1) != projected_state.size(1) ||
      weighted_projected_source.size(1) != projected_val.size(1)) {
    throw std::runtime_error("projected_state/projected_val must share source_nodes.");
  }
  if (weighted_projected_source.size(2) != projected_target.size(2)) {
    throw std::runtime_error("weighted_projected_source and projected_target must share rank_dim.");
  }
  if (projected_state.scalar_type() != torch::kFloat32 ||
      projected_val.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("projected_state and projected_val must be float32 accumulators.");
  }
  if (weighted_projected_source.scalar_type() != projected_target.scalar_type()) {
    projected_target = projected_target.to(weighted_projected_source.scalar_type());
  }

  const auto batch_flat = weighted_projected_source.size(0);
  const auto source_nodes = weighted_projected_source.size(1);
  const auto target_nodes = projected_target.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
  const auto out_dim = projected_val.size(2);

  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());

  const auto blocks = batch_flat * target_nodes;
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto width = std::min<int64_t>(window + 1, source_nodes);
  const auto shared_bytes = static_cast<size_t>(3 * width * sizeof(float));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_window_entmax15_forward_cuda",
      [&] {
        low_rank_propagation_window_entmax15_forward_kernel<scalar_t>
            <<<blocks, threads, shared_bytes, stream>>>(
                weighted_projected_source.data_ptr<scalar_t>(),
                projected_target.data_ptr<scalar_t>(),
                projected_state.data_ptr<float>(),
                projected_val.data_ptr<float>(),
                delta_state.data_ptr<float>(),
                delta_val.data_ptr<float>(),
                batch_flat,
                target_nodes,
                source_nodes,
                rank_dim,
                out_dim,
                window,
                static_cast<float>(score_bias));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {delta_state, delta_val};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> jakal_net_query_topk_reduce_backward_cuda(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val) {
  require_cuda_contiguous(edges, "edges");
  require_cuda_contiguous(indices, "indices");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  require_cuda_contiguous(grad_delta_state, "grad_delta_state");
  require_cuda_contiguous(grad_delta_val, "grad_delta_val");
  if (edges.scalar_type() != projected_state.scalar_type() ||
      edges.scalar_type() != projected_val.scalar_type() ||
      edges.scalar_type() != grad_delta_state.scalar_type() ||
      edges.scalar_type() != grad_delta_val.scalar_type()) {
    throw std::runtime_error("query_topk_reduce_backward_cuda inputs must share dtype.");
  }

  const auto batch_flat = edges.size(0);
  const auto query_nodes = edges.size(1);
  const auto k = edges.size(2);
  const auto source_nodes = projected_state.size(1);
  const auto out_dim = projected_val.size(2);
  auto float_options = edges.options().dtype(torch::kFloat32);
  auto grad_edges = torch::empty(edges.sizes(), float_options);
  auto grad_projected_state = torch::zeros(projected_state.sizes(), float_options);
  auto grad_projected_val = torch::zeros(projected_val.sizes(), float_options);

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t state_total = batch_flat * query_nodes * k;
  const int64_t val_total = batch_flat * query_nodes * k * out_dim;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      edges.scalar_type(),
      "query_topk_reduce_backward_cuda",
      [&] {
    query_topk_reduce_backward_state_kernel<scalar_t>
        <<<(state_total + threads - 1) / threads, threads, 0, stream>>>(
            edges.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            projected_state.data_ptr<scalar_t>(),
            grad_delta_state.data_ptr<scalar_t>(),
            grad_edges.data_ptr<float>(),
            grad_projected_state.data_ptr<float>(),
            batch_flat,
            query_nodes,
            source_nodes,
            k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    query_topk_reduce_backward_val_kernel<scalar_t>
        <<<(val_total + threads - 1) / threads, threads, 0, stream>>>(
            edges.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            projected_val.data_ptr<scalar_t>(),
            grad_delta_val.data_ptr<scalar_t>(),
            grad_edges.data_ptr<float>(),
            grad_projected_val.data_ptr<float>(),
            batch_flat,
            query_nodes,
            source_nodes,
            out_dim,
            k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {grad_edges, grad_projected_state, grad_projected_val};
}

torch::Tensor jakal_net_softsign_backward_cuda(
    const torch::Tensor& scores,
    const torch::Tensor& grad_edges) {
  require_cuda_contiguous(scores, "scores");
  require_cuda_contiguous(grad_edges, "grad_edges");
  if (scores.sizes() != grad_edges.sizes()) {
    throw std::runtime_error("scores and grad_edges must share shape.");
  }
  auto grad_scores = torch::empty(scores.sizes(), scores.options().dtype(torch::kFloat32));
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto total = scores.numel();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      scores.scalar_type(),
      "softsign_backward_cuda",
      [&] {
    softsign_backward_kernel<scalar_t>
        <<<(total + threads - 1) / threads, threads, 0, stream>>>(
            scores.data_ptr<scalar_t>(),
            grad_edges.data_ptr<float>(),
            grad_scores.data_ptr<float>(),
            total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return grad_scores;
}

torch::Tensor jakal_net_softmax_backward_cuda(
    const torch::Tensor& routes,
    const torch::Tensor& grad_routes) {
  require_cuda_contiguous(routes, "routes");
  require_cuda_contiguous(grad_routes, "grad_routes");
  if (routes.sizes() != grad_routes.sizes() || routes.dim() != 3) {
    throw std::runtime_error("routes and grad_routes must be shaped [batch, query_nodes, topk].");
  }
  auto grad_scores = torch::empty(routes.sizes(), routes.options().dtype(torch::kFloat32));
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto rows = routes.size(0) * routes.size(1);
  const auto k = routes.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      routes.scalar_type(),
      "softmax_backward_cuda",
      [&] {
    softmax_backward_kernel<scalar_t>
        <<<(rows + threads - 1) / threads, threads, 0, stream>>>(
            routes.data_ptr<scalar_t>(),
            grad_routes.data_ptr<float>(),
            grad_scores.data_ptr<float>(),
            rows,
            k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return grad_scores;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_diagonal_pairwise_topk_backward_cuda(
    const torch::Tensor& query_val,
    const torch::Tensor& source_val,
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    const torch::Tensor& grad_scores,
    double temperature) {
  require_cuda_contiguous(query_val, "query_val");
  require_cuda_contiguous(source_val, "source_val");
  require_cuda_contiguous(weight, "weight");
  require_cuda_contiguous(indices, "indices");
  require_cuda_contiguous(grad_scores, "grad_scores");
  if (temperature <= 0.0) {
    throw std::runtime_error("temperature must be positive.");
  }

  const auto batch_flat = query_val.size(0);
  const auto query_nodes = query_val.size(1);
  const auto source_nodes = source_val.size(1);
  const auto dim = query_val.size(2);
  const auto k = indices.size(2);
  auto float_options = query_val.options().dtype(torch::kFloat32);
  auto grad_query = torch::zeros(query_val.sizes(), float_options);
  auto grad_source = torch::zeros(source_val.sizes(), float_options);
  auto grad_weight = torch::zeros(weight.sizes(), weight.options().dtype(torch::kFloat32));
  auto grad_bias = torch::zeros({}, weight.options().dtype(torch::kFloat32));

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = batch_flat * query_nodes * k * dim;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      query_val.scalar_type(),
      "diagonal_pairwise_topk_backward_cuda",
      [&] {
    diagonal_pairwise_topk_backward_kernel<scalar_t>
        <<<(total + threads - 1) / threads, threads, 0, stream>>>(
            query_val.data_ptr<scalar_t>(),
            source_val.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            grad_scores.data_ptr<float>(),
            grad_query.data_ptr<float>(),
            grad_source.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_bias.data_ptr<float>(),
            static_cast<float>(1.0 / temperature),
            batch_flat,
            query_nodes,
            source_nodes,
            dim,
            k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_query, grad_source, grad_weight, grad_bias};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_low_rank_pairwise_topk_backward_cuda(
    const torch::Tensor& query_val,
    const torch::Tensor& source_val,
    const torch::Tensor& source_weight,
    const torch::Tensor& target_weight,
    const torch::Tensor& core_weight,
    const torch::Tensor& projected_query,
    const torch::Tensor& projected_source,
    const torch::Tensor& indices,
    const torch::Tensor& grad_scores,
    double temperature) {
  require_cuda_contiguous(query_val, "query_val");
  require_cuda_contiguous(source_val, "source_val");
  require_cuda_contiguous(source_weight, "source_weight");
  require_cuda_contiguous(target_weight, "target_weight");
  require_cuda_contiguous(core_weight, "core_weight");
  require_cuda_contiguous(projected_query, "projected_query");
  require_cuda_contiguous(projected_source, "projected_source");
  require_cuda_contiguous(indices, "indices");
  require_cuda_contiguous(grad_scores, "grad_scores");
  if (temperature <= 0.0) {
    throw std::runtime_error("temperature must be positive.");
  }

  const auto batch_flat = query_val.size(0);
  const auto query_nodes = query_val.size(1);
  const auto source_nodes = source_val.size(1);
  const auto dim = query_val.size(2);
  const auto rank_dim = core_weight.size(0);
  const auto k = indices.size(2);
  auto float_options = query_val.options().dtype(torch::kFloat32);
  auto grad_query = torch::zeros(query_val.sizes(), float_options);
  auto grad_source = torch::zeros(source_val.sizes(), float_options);
  auto grad_source_weight =
      torch::zeros(source_weight.sizes(), source_weight.options().dtype(torch::kFloat32));
  auto grad_target_weight =
      torch::zeros(target_weight.sizes(), target_weight.options().dtype(torch::kFloat32));
  auto grad_core_weight =
      torch::zeros(core_weight.sizes(), core_weight.options().dtype(torch::kFloat32));
  auto grad_bias = torch::zeros({}, core_weight.options().dtype(torch::kFloat32));

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = batch_flat * query_nodes * k * rank_dim * dim;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      query_val.scalar_type(),
      "low_rank_pairwise_topk_backward_cuda",
      [&] {
    low_rank_pairwise_topk_backward_kernel<scalar_t>
        <<<(total + threads - 1) / threads, threads, 0, stream>>>(
            query_val.data_ptr<scalar_t>(),
            source_val.data_ptr<scalar_t>(),
            source_weight.data_ptr<scalar_t>(),
            target_weight.data_ptr<scalar_t>(),
            core_weight.data_ptr<scalar_t>(),
            projected_query.data_ptr<scalar_t>(),
            projected_source.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            grad_scores.data_ptr<float>(),
            grad_query.data_ptr<float>(),
            grad_source.data_ptr<float>(),
            grad_source_weight.data_ptr<float>(),
            grad_target_weight.data_ptr<float>(),
            grad_core_weight.data_ptr<float>(),
            grad_bias.data_ptr<float>(),
            static_cast<float>(1.0 / temperature),
            batch_flat,
            query_nodes,
            source_nodes,
            dim,
            rank_dim,
            k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return {grad_query, grad_source, grad_source_weight, grad_target_weight, grad_core_weight, grad_bias};
}

namespace {

struct ScanCudaLayerState {
  torch::Tensor state;
  torch::Tensor val;
};

inline c10::optional<torch::Tensor> scan_cuda_optional_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return c10::nullopt;
  }
  return tensor;
}

torch::Tensor scan_cuda_linear3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  auto output = torch::matmul(input, weight.to(input.scalar_type()).transpose(0, 1));
  if (bias.defined() && bias.numel() != 0) {
    output = output + bias.to(output.scalar_type()).view({1, 1, -1});
  }
  return output;
}

torch::Tensor scan_cuda_value_to_state3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  if (!weight.defined() || weight.numel() == 0) {
    return torch::linalg_vector_norm(input, 2, std::vector<int64_t>{-1}, false);
  }
  return scan_cuda_linear3d(input, weight, bias).squeeze(-1);
}

torch::Tensor scan_cuda_linear2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  auto output = torch::matmul(input, weight.to(input.scalar_type()).transpose(0, 1));
  if (bias.defined() && bias.numel() != 0) {
    output = output + bias.to(output.scalar_type());
  }
  return output;
}

torch::Tensor scan_cuda_layer_norm_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  if (!weight.defined() || weight.numel() == 0) {
    return input;
  }
  auto optional_bias = scan_cuda_optional_tensor(bias);
  return torch::layer_norm(
      input,
      {input.size(-1)},
      weight.to(input.scalar_type()),
      optional_bias.has_value() ? c10::optional<torch::Tensor>(optional_bias.value().to(input.scalar_type())) : c10::nullopt,
      1e-5,
      false);
}

torch::Tensor scan_cuda_signed_softmax_state(const torch::Tensor& state) {
  auto clean_state = torch::nan_to_num(state);
  clean_state = torch::layer_norm(
      clean_state,
      {clean_state.size(-1)},
      c10::nullopt,
      c10::nullopt,
      1e-5,
      false);
  auto magnitude = torch::softmax(clean_state.abs(), -1);
  const auto state_mass = static_cast<double>(state.size(-1));
  return torch::sign(clean_state) * magnitude * state_mass;
}

torch::Tensor scan_cuda_signed_abs_softmax_scores(const torch::Tensor& scores) {
  auto clean_scores = torch::nan_to_num(scores);
  return torch::nan_to_num(torch::sign(clean_scores) * torch::softmax(clean_scores.abs(), -1));
}

torch::Tensor scan_cuda_entmax15_scores(
    const torch::Tensor& scores,
    const c10::optional<torch::Tensor>& mask = c10::nullopt) {
  auto clean_scores = torch::nan_to_num(scores);
  if (mask.has_value()) {
    auto bool_mask = mask.value().to(torch::kBool);
    clean_scores = clean_scores.masked_fill(~bool_mask, -1e4);
  }
  auto shifted = clean_scores - std::get<0>(clean_scores.max(-1, true));
  shifted = shifted / 2;
  auto sorted = std::get<0>(shifted.sort(-1, true));
  auto cumulative = sorted.cumsum(-1);
  auto cumulative_sq = (sorted * sorted).cumsum(-1);
  auto rho = torch::arange(
      1,
      sorted.size(-1) + 1,
      sorted.options().dtype(sorted.scalar_type()));
  std::vector<int64_t> rho_shape(sorted.dim(), 1);
  rho_shape.back() = sorted.size(-1);
  rho = rho.view(rho_shape);
  auto mean = cumulative / rho;
  auto mean_sq = cumulative_sq / rho;
  auto support_var = rho * (mean_sq - mean * mean);
  auto delta = torch::clamp((1 - support_var) / rho, 0);
  auto tau = mean - torch::sqrt(delta);
  auto support = (tau <= sorted).sum(-1, true).clamp_min(1);
  auto tau_star = tau.gather(-1, support - 1);
  auto probs = torch::pow(torch::clamp(shifted - tau_star, 0), 2);
  if (mask.has_value()) {
    probs = probs * mask.value().to(probs.scalar_type());
  }
  auto denom = probs.sum(-1, true);
  return torch::where(
      denom > 0,
      probs / denom.clamp_min(std::numeric_limits<double>::epsilon()),
      torch::zeros_like(probs));
}

torch::Tensor scan_cuda_signed_entmax15_scores(
    const torch::Tensor& scores,
    const c10::optional<torch::Tensor>& mask = c10::nullopt) {
  auto clean_scores = torch::nan_to_num(scores);
  auto probs = scan_cuda_entmax15_scores(clean_scores.abs(), mask);
  auto signed_routes = torch::sign(clean_scores) * probs;
  if (mask.has_value()) {
    signed_routes = signed_routes * mask.value().to(signed_routes.scalar_type());
  }
  return torch::nan_to_num(signed_routes);
}

torch::Tensor scan_cuda_softsign(const torch::Tensor& tensor) {
  return tensor / (1.0 + tensor.abs());
}

torch::Tensor scan_cuda_compress_scores(
    const std::string& compress_name,
    const torch::Tensor& scores,
    const c10::optional<torch::Tensor>& mask = c10::nullopt) {
  if (compress_name == "softmax") {
    auto clean_scores = torch::nan_to_num(scores);
    if (mask.has_value()) {
      auto bool_mask = mask.value().to(torch::kBool);
      auto masked_scores = clean_scores.masked_fill(~bool_mask, -1e4);
      auto routes = torch::softmax(masked_scores, -1);
      return torch::where(bool_mask, routes, torch::zeros_like(routes));
    }
    return torch::softmax(clean_scores, -1);
  }
  if (compress_name == "signed_abs_softmax") {
    auto routes = scan_cuda_signed_abs_softmax_scores(scores);
    if (mask.has_value()) {
      routes = routes * mask.value().to(routes.scalar_type());
    }
    return routes;
  }
  if (compress_name == "signed_entmax15") {
    return scan_cuda_signed_entmax15_scores(scores, mask);
  }
  if (compress_name == "softsign") {
    auto edges = scan_cuda_softsign(scores);
    if (mask.has_value()) {
      edges = edges * mask.value().to(edges.scalar_type());
    }
    return edges;
  }
  throw std::runtime_error("Unsupported compress_name: " + compress_name);
}

ScanCudaLayerState scan_cuda_layer_with_val_norm(
    const ScanCudaLayerState& layer,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  return {layer.state, scan_cuda_layer_norm_last_dim(layer.val, weight, bias)};
}

ScanCudaLayerState scan_cuda_apply_delta_to_layer(
    const ScanCudaLayerState& layer,
    const torch::Tensor& delta_state,
    const torch::Tensor& delta_val,
    const torch::Tensor& val_norm_weight,
    const torch::Tensor& val_norm_bias,
    bool use_fused_forward) {
  if (use_fused_forward &&
      layer.state.is_cuda() &&
      delta_state.is_cuda() &&
      layer.val.is_cuda() &&
      delta_val.is_cuda() &&
      layer.state.is_contiguous() &&
      delta_state.is_contiguous() &&
      layer.val.is_contiguous() &&
      delta_val.is_contiguous() &&
      layer.state.dim() == 2 &&
      layer.val.dim() == 3 &&
      layer.state.scalar_type() == delta_state.scalar_type() &&
      layer.val.scalar_type() == delta_val.scalar_type() &&
      (!val_norm_weight.defined() || val_norm_weight.numel() == 0 ||
       val_norm_weight.scalar_type() == layer.val.scalar_type()) &&
      (!val_norm_bias.defined() || val_norm_bias.numel() == 0 ||
       val_norm_bias.scalar_type() == layer.val.scalar_type())) {
    auto updated_state = torch::empty_like(layer.state);
    auto updated_val = torch::empty_like(layer.val);
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto state_shmem = 2 * threads * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        layer.state.scalar_type(),
        "scan_signed_softmax_state_kernel",
        [&] {
          scan_signed_softmax_state_kernel<scalar_t>
              <<<layer.state.size(0), threads, state_shmem, stream>>>(
                  layer.state.data_ptr<scalar_t>(),
                  delta_state.data_ptr<scalar_t>(),
                  updated_state.data_ptr<scalar_t>(),
                  layer.state.size(0),
                  layer.state.size(1));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    const auto val_rows = layer.val.size(0) * layer.val.size(1);
    const auto val_dim = layer.val.size(2);
    const auto val_shmem = 2 * threads * sizeof(float);
    const bool has_weight = val_norm_weight.defined() && val_norm_weight.numel() != 0;
    const bool has_bias = val_norm_bias.defined() && val_norm_bias.numel() != 0;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        layer.val.scalar_type(),
        "scan_val_add_layer_norm_kernel",
        [&] {
          scan_val_add_layer_norm_kernel<scalar_t>
              <<<val_rows, threads, val_shmem, stream>>>(
                  layer.val.data_ptr<scalar_t>(),
                  delta_val.data_ptr<scalar_t>(),
                  has_weight ? val_norm_weight.data_ptr<scalar_t>() : nullptr,
                  has_bias ? val_norm_bias.data_ptr<scalar_t>() : nullptr,
                  updated_val.data_ptr<scalar_t>(),
                  val_rows,
                  val_dim,
                  has_weight,
                  has_bias);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return {updated_state, updated_val};
  }
  auto updated_state = scan_cuda_signed_softmax_state(layer.state + delta_state);
  auto updated_val = scan_cuda_layer_norm_last_dim(layer.val + delta_val, val_norm_weight, val_norm_bias);
  return {updated_state, updated_val};
}

torch::Tensor scan_cuda_full_topk_indices(const torch::Tensor& scores) {
  const auto batch = scores.size(0);
  const auto left = scores.size(1);
  const auto right = scores.size(2);
  return torch::arange(right, scores.options().dtype(torch::kLong))
      .view({1, 1, right})
      .expand({batch, left, right});
}

torch::Tensor scan_cuda_select_head_tensor(
    const torch::Tensor& tensor,
    int64_t head_index,
    int64_t num_heads) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return tensor;
  }
  if (tensor.dim() > 0 && tensor.size(0) == num_heads) {
    return tensor.select(0, head_index).contiguous();
  }
  return tensor;
}

std::tuple<torch::Tensor, torch::Tensor> scan_cuda_low_rank_transition_pairwise_topk(
    const torch::Tensor& sender_strength,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& src_val,
    const torch::Tensor& dst_val,
    const torch::Tensor& source_weight,
    const torch::Tensor& target_weight,
    const torch::Tensor& core_weight,
    const torch::Tensor& bias,
    int64_t topk,
    const std::string& compress_name,
    bool allow_fastpath) {
  const bool diagonal =
      (!source_weight.defined() || source_weight.numel() == 0) &&
      (!target_weight.defined() || target_weight.numel() == 0);
  const bool multihead =
      diagonal ? core_weight.dim() >= 2
               : (source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2);
  const bool use_fused_topk =
      compress_name == "softmax" ||
      compress_name == "signed_abs_softmax" ||
      compress_name == "signed_entmax15";
  if (use_fused_topk &&
      allow_fastpath &&
      !diagonal &&
      topk > 0 && topk <= 64) {
    torch::Tensor weighted_projected_source;
    torch::Tensor projected_target;
    if (multihead) {
      auto cast_source = source_weight.to(src_val.scalar_type());
      auto cast_target = target_weight.to(dst_val.scalar_type());
      auto cast_core = core_weight.to(src_val.scalar_type());
      auto projected_source = torch::einsum("bid,hrd->bhir", {src_val, cast_source}).contiguous();
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[1] = cast_core.size(0);
      core_shape[3] = cast_core.size(1);
      weighted_projected_source = (projected_source * cast_core.view(core_shape)).contiguous();
      projected_target = torch::einsum("bkd,hrd->bhkr", {dst_val, cast_target}).contiguous();
      if (bias.defined() && bias.numel() != 0) {
        auto head_bias = bias.to(weighted_projected_source.scalar_type()).reshape({1, bias.numel(), 1, 1});
        auto bias_column = head_bias.expand({weighted_projected_source.size(0), weighted_projected_source.size(1), weighted_projected_source.size(2), 1});
        auto target_ones = torch::ones(
            {projected_target.size(0), projected_target.size(1), projected_target.size(2), 1},
            projected_target.options());
        weighted_projected_source = torch::cat({weighted_projected_source, bias_column}, -1).contiguous();
        projected_target = torch::cat({projected_target, target_ones}, -1).contiguous();
      }
    } else {
      auto projected_source = torch::matmul(src_val, source_weight.to(src_val.scalar_type()).transpose(0, 1)).contiguous();
      weighted_projected_source = (projected_source * core_weight.to(projected_source.scalar_type()).view({1, 1, -1})).contiguous();
      projected_target = torch::matmul(dst_val, target_weight.to(dst_val.scalar_type()).transpose(0, 1)).contiguous();
    }
    auto weighted_projected_state =
        (sender_strength.to(torch::kFloat32) * projected_state.to(torch::kFloat32)).contiguous();
    auto weighted_projected_val =
        (sender_strength.to(torch::kFloat32).unsqueeze(-1) * projected_val.to(torch::kFloat32)).contiguous();
    auto fused = jakal_net_low_rank_pairwise_topk_forward_cuda(
        weighted_projected_source,
        projected_target,
        weighted_projected_state,
        weighted_projected_val,
        topk,
        (!multihead && bias.defined() && bias.numel() != 0) ? bias.item<double>() : 0.0,
        compress_name == "signed_abs_softmax"
            ? kCompressSignedAbsSoftmax
            : (compress_name == "signed_entmax15" ? kCompressSignedEntmax15 : kCompressSoftmax));
    return {
        std::get<0>(fused).to(projected_state.scalar_type()),
        std::get<1>(fused).to(projected_val.scalar_type()),
    };
  }

  torch::Tensor logits;
  if (diagonal && multihead) {
    auto cast_core = core_weight.to(src_val.scalar_type());
    auto weighted_source = src_val.unsqueeze(1) * cast_core.view({1, cast_core.size(0), 1, cast_core.size(1)});
    logits = torch::einsum("bhid,bkd->bhik", {weighted_source, dst_val});
    if (bias.defined() && bias.numel() != 0) {
      std::vector<int64_t> bias_shape(logits.dim(), 1);
      bias_shape[logits.dim() - 3] = bias.size(0);
      logits = logits + bias.to(logits.scalar_type()).view(bias_shape);
    }
    logits = std::get<0>(logits.max(1));
  } else if (diagonal) {
    auto weighted_source = src_val * core_weight.to(src_val.scalar_type()).view({1, 1, -1});
    logits = torch::matmul(weighted_source, dst_val.transpose(1, 2));
    if (bias.defined() && bias.numel() != 0) {
      logits = logits + bias.to(logits.scalar_type());
    }
  } else if (multihead) {
    auto cast_source = source_weight.to(src_val.scalar_type());
    auto cast_target = target_weight.to(dst_val.scalar_type());
    auto cast_core = core_weight.to(src_val.scalar_type());
    auto projected_source = torch::einsum("bid,hrd->bhir", {src_val, cast_source});
    auto projected_target = torch::einsum("bkd,hrd->bhkr", {dst_val, cast_target});
    std::vector<int64_t> core_shape(projected_source.dim(), 1);
    core_shape[projected_source.dim() - 3] = cast_core.size(0);
    core_shape[projected_source.dim() - 1] = cast_core.size(1);
    auto weighted_projected_source = projected_source * cast_core.view(core_shape);
    logits = torch::einsum("bhir,bhkr->bhik", {weighted_projected_source, projected_target});
    if (bias.defined() && bias.numel() != 0) {
      std::vector<int64_t> bias_shape(logits.dim(), 1);
      bias_shape[logits.dim() - 3] = bias.size(0);
      logits = logits + bias.to(logits.scalar_type()).view(bias_shape);
    }
    logits = std::get<0>(logits.max(1));
  } else {
    auto projected_source = torch::matmul(src_val, source_weight.to(src_val.scalar_type()).transpose(0, 1));
    auto weighted_projected_source = projected_source * core_weight.to(projected_source.scalar_type()).view({1, 1, -1});
    auto projected_target = torch::matmul(dst_val, target_weight.to(dst_val.scalar_type()).transpose(0, 1));
    logits = torch::matmul(weighted_projected_source, projected_target.transpose(1, 2));
    if (bias.defined() && bias.numel() != 0) {
      logits = logits + bias.to(logits.scalar_type());
    }
  }
  const auto dst_nodes = dst_val.size(1);
  const auto k = std::min<int64_t>(std::max<int64_t>(1, topk), dst_nodes);
  if (k == dst_nodes) {
    auto routes = scan_cuda_compress_scores(compress_name, logits);
    auto weighted_routes = routes * sender_strength.unsqueeze(-1);
    auto routes_t = weighted_routes.transpose(1, 2).to(torch::kFloat32);
    auto delta_state = torch::bmm(
        routes_t,
        projected_state.to(torch::kFloat32).unsqueeze(-1))
        .squeeze(-1);
    auto delta_val = torch::bmm(routes_t, projected_val.to(torch::kFloat32));
    return {
        delta_state.to(projected_state.scalar_type()),
        delta_val.to(projected_val.scalar_type()),
    };
  }

  auto topk_result = logits.topk(k, -1, true, true);
  auto selected_scores = std::get<0>(topk_result);
  auto selected_indices = std::get<1>(topk_result);
  auto routes = scan_cuda_compress_scores(compress_name, selected_scores);
  auto weighted_routes = routes * sender_strength.unsqueeze(-1);
  auto delta_state = torch::zeros(
      {projected_state.size(0), dst_nodes},
      projected_state.options().dtype(torch::kFloat32));
  auto delta_val = torch::zeros(
      {projected_val.size(0), dst_nodes, projected_val.size(2)},
      projected_val.options().dtype(torch::kFloat32));
  auto flat_indices = selected_indices.reshape({selected_indices.size(0), -1});
  auto state_contrib =
      (weighted_routes.to(torch::kFloat32) * projected_state.to(torch::kFloat32).unsqueeze(-1))
          .reshape({projected_state.size(0), -1});
  delta_state.scatter_add_(1, flat_indices, state_contrib);
  auto val_contrib =
      (weighted_routes.to(torch::kFloat32).unsqueeze(-1) *
       projected_val.to(torch::kFloat32).unsqueeze(-2))
          .reshape({projected_val.size(0), -1, projected_val.size(2)});
  auto scatter_index =
      flat_indices.unsqueeze(-1).expand({flat_indices.size(0), flat_indices.size(1), projected_val.size(2)});
  delta_val.scatter_add_(1, scatter_index, val_contrib);
  return {
      delta_state.to(projected_state.scalar_type()),
      delta_val.to(projected_val.scalar_type()),
  };
}

std::tuple<torch::Tensor, torch::Tensor> scan_cuda_low_rank_propagation_topk(
    const torch::Tensor& layer_state,
    const torch::Tensor& layer_val,
    const torch::Tensor& source_weight,
    const torch::Tensor& target_weight,
    const torch::Tensor& core_weight,
    const torch::Tensor& bias,
    int64_t topk,
    const std::string& compress_name,
    bool allow_fastpath) {
  const bool diagonal =
      (!source_weight.defined() || source_weight.numel() == 0) &&
      (!target_weight.defined() || target_weight.numel() == 0);
  const bool multihead =
      diagonal ? core_weight.dim() >= 2
               : (source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2);
  const bool use_fused_topk =
      compress_name == "softsign" ||
      compress_name == "signed_abs_softmax";
  if (use_fused_topk &&
      allow_fastpath &&
      !diagonal &&
      topk > 0 && topk <= 64) {
    torch::Tensor weighted_projected_source;
    torch::Tensor projected_target;
    if (multihead) {
      auto cast_source = source_weight.to(layer_val.scalar_type());
      auto cast_target = target_weight.to(layer_val.scalar_type());
      auto cast_core = core_weight.to(layer_val.scalar_type());
      auto projected_source = torch::einsum("bid,hrd->bhir", {layer_val, cast_source}).contiguous();
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[1] = cast_core.size(0);
      core_shape[3] = cast_core.size(1);
      weighted_projected_source = (projected_source * cast_core.view(core_shape)).contiguous();
      projected_target = torch::einsum("bid,hrd->bhir", {layer_val, cast_target}).contiguous();
      if (bias.defined() && bias.numel() != 0) {
        auto head_bias = bias.to(weighted_projected_source.scalar_type()).reshape({1, bias.numel(), 1, 1});
        auto bias_column = head_bias.expand({weighted_projected_source.size(0), weighted_projected_source.size(1), weighted_projected_source.size(2), 1});
        auto target_ones = torch::ones(
            {projected_target.size(0), projected_target.size(1), projected_target.size(2), 1},
            projected_target.options());
        weighted_projected_source = torch::cat({weighted_projected_source, bias_column}, -1).contiguous();
        projected_target = torch::cat({projected_target, target_ones}, -1).contiguous();
      }
    } else {
      auto projected_target_single = torch::matmul(layer_val, target_weight.to(layer_val.scalar_type()).transpose(0, 1)).contiguous();
      auto projected_source_single = torch::matmul(layer_val, source_weight.to(layer_val.scalar_type()).transpose(0, 1)).contiguous();
      weighted_projected_source = (projected_source_single * core_weight.to(projected_source_single.scalar_type()).view({1, 1, -1})).contiguous();
      projected_target = projected_target_single;
    }
    auto weighted_projected_state = layer_state.to(torch::kFloat32).contiguous();
    auto weighted_projected_val = layer_val.to(torch::kFloat32).contiguous();
    const auto score_bias =
        (!multihead && bias.defined() && bias.numel() != 0) ? bias.item<double>() : 0.0;
    auto fused = jakal_net_low_rank_propagation_topk_forward_cuda(
        weighted_projected_source,
        projected_target,
        weighted_projected_state,
        weighted_projected_val,
        topk,
        multihead ? 0.0 : score_bias,
        compress_name == "signed_abs_softmax" ? kCompressSignedAbsSoftmax : 0);
    return {
        std::get<0>(fused).to(layer_state.scalar_type()),
        std::get<1>(fused).to(layer_val.scalar_type()),
    };
  }

  auto dense_score_block = [&](const torch::Tensor& target_block, const torch::Tensor& source_block) {
    torch::Tensor block_scores;
    if (diagonal && multihead) {
      auto cast_core = core_weight.to(target_block.scalar_type());
      auto weighted_target = target_block.unsqueeze(1) * cast_core.view({1, cast_core.size(0), 1, cast_core.size(1)});
      block_scores = torch::einsum("bhid,bjd->bhij", {weighted_target, source_block});
      if (bias.defined() && bias.numel() != 0) {
        std::vector<int64_t> bias_shape(block_scores.dim(), 1);
        bias_shape[block_scores.dim() - 3] = bias.size(0);
        block_scores = block_scores + bias.to(block_scores.scalar_type()).view(bias_shape);
      }
      block_scores = std::get<0>(block_scores.max(1));
    } else if (diagonal) {
      auto weighted_target = target_block * core_weight.to(target_block.scalar_type()).view({1, 1, -1});
      block_scores = torch::matmul(weighted_target, source_block.transpose(1, 2));
      if (bias.defined() && bias.numel() != 0) {
        block_scores = block_scores + bias.to(block_scores.scalar_type());
      }
    } else if (multihead) {
      auto cast_source = source_weight.to(source_block.scalar_type());
      auto cast_target = target_weight.to(target_block.scalar_type());
      auto cast_core = core_weight.to(source_block.scalar_type());
      auto projected_target = torch::einsum("bid,hrd->bhir", {target_block, cast_target});
      auto projected_source = torch::einsum("bid,hrd->bhir", {source_block, cast_source});
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[projected_source.dim() - 3] = cast_core.size(0);
      core_shape[projected_source.dim() - 1] = cast_core.size(1);
      auto weighted_projected_source = projected_source * cast_core.view(core_shape);
      block_scores = torch::einsum("bhir,bhkr->bhik", {projected_target, weighted_projected_source});
      if (bias.defined() && bias.numel() != 0) {
        std::vector<int64_t> bias_shape(block_scores.dim(), 1);
        bias_shape[block_scores.dim() - 3] = bias.size(0);
        block_scores = block_scores + bias.to(block_scores.scalar_type()).view(bias_shape);
      }
      block_scores = std::get<0>(block_scores.max(1));
    } else {
      auto projected_target = torch::matmul(target_block, target_weight.to(target_block.scalar_type()).transpose(0, 1));
      auto projected_source = torch::matmul(source_block, source_weight.to(source_block.scalar_type()).transpose(0, 1));
      auto weighted_projected_source = projected_source * core_weight.to(projected_source.scalar_type()).view({1, 1, -1});
      block_scores = torch::matmul(projected_target, weighted_projected_source.transpose(1, 2));
      if (bias.defined() && bias.numel() != 0) {
        block_scores = block_scores + bias.to(block_scores.scalar_type());
      }
    }
    return block_scores;
  };

  if (topk <= 0 &&
      (compress_name == "signed_abs_softmax" ||
       compress_name == "softmax" ||
       compress_name == "softsign")) {
    const auto nodes = layer_val.size(1);
    const auto block_size = scan_cuda_dense_block_size();
    const auto state_acc_dtype = scan_cuda_accumulator_dtype(layer_state.scalar_type());
    const auto val_acc_dtype = scan_cuda_accumulator_dtype(layer_val.scalar_type());
    auto delta_state = torch::zeros(
        {layer_state.size(0), nodes},
        layer_state.options().dtype(state_acc_dtype));
    auto delta_val = torch::zeros(
        {layer_val.size(0), nodes, layer_val.size(2)},
        layer_val.options().dtype(val_acc_dtype));
    for (int64_t target_start = 0; target_start < nodes; target_start += block_size) {
      const auto target_end = std::min<int64_t>(target_start + block_size, nodes);
      auto target_block = layer_val.slice(1, target_start, target_end);
      auto target_state_acc = torch::zeros(
          {layer_state.size(0), target_end - target_start},
          layer_state.options().dtype(state_acc_dtype));
      auto target_val_acc = torch::zeros(
          {layer_val.size(0), target_end - target_start, layer_val.size(2)},
          layer_val.options().dtype(val_acc_dtype));
      if (compress_name == "softsign") {
        for (int64_t source_start = 0; source_start < nodes; source_start += block_size) {
          const auto source_end = std::min<int64_t>(source_start + block_size, nodes);
          auto edges = scan_cuda_compress_scores(
              compress_name,
              dense_score_block(target_block, layer_val.slice(1, source_start, source_end)));
          target_state_acc = target_state_acc + torch::bmm(
              edges.to(state_acc_dtype),
              layer_state.slice(1, source_start, source_end).to(state_acc_dtype).unsqueeze(-1)).squeeze(-1);
          target_val_acc = target_val_acc + torch::bmm(
              edges.to(val_acc_dtype),
              layer_val.slice(1, source_start, source_end).to(val_acc_dtype));
        }
      } else {
        auto running_max = torch::full(
            {layer_state.size(0), target_end - target_start},
            -std::numeric_limits<float>::infinity(),
            layer_state.options().dtype(torch::kFloat32));
        auto running_sum = torch::zeros_like(running_max);
        for (int64_t source_start = 0; source_start < nodes; source_start += block_size) {
          const auto source_end = std::min<int64_t>(source_start + block_size, nodes);
          auto scores_block = torch::nan_to_num(
              dense_score_block(target_block, layer_val.slice(1, source_start, source_end))).to(torch::kFloat32);
          auto stats_block = compress_name == "signed_abs_softmax" ? scores_block.abs() : scores_block;
          auto block_max = std::get<0>(stats_block.max(-1));
          auto next_max = torch::maximum(running_max, block_max);
          running_sum = running_sum * torch::exp(running_max - next_max) +
              torch::exp(stats_block - next_max.unsqueeze(-1)).sum(-1);
          running_max = next_max;
        }
        for (int64_t source_start = 0; source_start < nodes; source_start += block_size) {
          const auto source_end = std::min<int64_t>(source_start + block_size, nodes);
          auto scores_block = torch::nan_to_num(
              dense_score_block(target_block, layer_val.slice(1, source_start, source_end))).to(torch::kFloat32);
          auto stats_block = compress_name == "signed_abs_softmax" ? scores_block.abs() : scores_block;
          auto edges = torch::exp(stats_block - running_max.unsqueeze(-1)) / running_sum.unsqueeze(-1);
          if (compress_name == "signed_abs_softmax") {
            edges = edges * torch::sign(scores_block);
          }
          target_state_acc = target_state_acc + torch::bmm(
              edges.to(state_acc_dtype),
              layer_state.slice(1, source_start, source_end).to(state_acc_dtype).unsqueeze(-1)).squeeze(-1);
          target_val_acc = target_val_acc + torch::bmm(
              edges.to(val_acc_dtype),
              layer_val.slice(1, source_start, source_end).to(val_acc_dtype));
        }
      }
      delta_state.slice(1, target_start, target_end).copy_(target_state_acc);
      delta_val.slice(1, target_start, target_end).copy_(target_val_acc);
    }
    return {
        delta_state.to(layer_state.scalar_type()),
        delta_val.to(layer_val.scalar_type()),
    };
  }

  torch::Tensor scores;
  if (diagonal && multihead) {
    auto cast_core = core_weight.to(layer_val.scalar_type());
    auto weighted_source = layer_val.unsqueeze(1) * cast_core.view({1, cast_core.size(0), 1, cast_core.size(1)});
    scores = torch::einsum("bhid,bjd->bhij", {weighted_source, layer_val});
    if (bias.defined() && bias.numel() != 0) {
      std::vector<int64_t> bias_shape(scores.dim(), 1);
      bias_shape[scores.dim() - 3] = bias.size(0);
      scores = scores + bias.to(scores.scalar_type()).view(bias_shape);
    }
    scores = std::get<0>(scores.max(1));
  } else if (diagonal) {
    auto weighted_source = layer_val * core_weight.to(layer_val.scalar_type()).view({1, 1, -1});
    scores = torch::matmul(weighted_source, layer_val.transpose(1, 2));
    if (bias.defined() && bias.numel() != 0) {
      scores = scores + bias.to(scores.scalar_type());
    }
  } else if (multihead) {
    auto cast_source = source_weight.to(layer_val.scalar_type());
    auto cast_target = target_weight.to(layer_val.scalar_type());
    auto cast_core = core_weight.to(layer_val.scalar_type());
    auto projected_target = torch::einsum("bid,hrd->bhir", {layer_val, cast_target});
    auto projected_source = torch::einsum("bid,hrd->bhir", {layer_val, cast_source});
    std::vector<int64_t> core_shape(projected_source.dim(), 1);
    core_shape[projected_source.dim() - 3] = cast_core.size(0);
    core_shape[projected_source.dim() - 1] = cast_core.size(1);
    auto weighted_projected_source = projected_source * cast_core.view(core_shape);
    scores = torch::einsum("bhir,bhkr->bhik", {projected_target, weighted_projected_source});
    if (bias.defined() && bias.numel() != 0) {
      std::vector<int64_t> bias_shape(scores.dim(), 1);
      bias_shape[scores.dim() - 3] = bias.size(0);
      scores = scores + bias.to(scores.scalar_type()).view(bias_shape);
    }
    scores = std::get<0>(scores.max(1));
  } else {
    auto projected_target = torch::matmul(layer_val, target_weight.to(layer_val.scalar_type()).transpose(0, 1));
    auto projected_source = torch::matmul(layer_val, source_weight.to(layer_val.scalar_type()).transpose(0, 1));
    auto weighted_projected_source = projected_source * core_weight.to(projected_source.scalar_type()).view({1, 1, -1});
    scores = torch::matmul(projected_target, weighted_projected_source.transpose(1, 2));
    if (bias.defined() && bias.numel() != 0) {
      scores = scores + bias.to(scores.scalar_type());
    }
  }
  const auto nodes = layer_val.size(1);
  const auto k = topk <= 0 ? nodes : std::min<int64_t>(std::max<int64_t>(1, topk), nodes);
  if (k == nodes) {
    auto edges = scan_cuda_compress_scores(compress_name, scores);
    auto delta_state = torch::bmm(
        edges.to(torch::kFloat32),
        layer_state.to(torch::kFloat32).unsqueeze(-1)).squeeze(-1);
    auto delta_val = torch::bmm(
        edges.to(torch::kFloat32),
        layer_val.to(torch::kFloat32));
    return {
        delta_state.to(layer_state.scalar_type()),
        delta_val.to(layer_val.scalar_type()),
    };
  }

  auto topk_result = scores.topk(k, -1, true, true);
  auto selected_scores = std::get<0>(topk_result);
  auto selected_indices = std::get<1>(topk_result);
  auto expanded_state = layer_state.unsqueeze(1).expand({layer_state.size(0), nodes, layer_state.size(1)});
  auto selected_state = torch::gather(expanded_state, 2, selected_indices);
  auto expanded_val = layer_val.unsqueeze(1).expand({layer_val.size(0), nodes, layer_val.size(1), layer_val.size(2)});
  auto selected_val = torch::gather(
      expanded_val,
      2,
      selected_indices.unsqueeze(-1).expand({selected_indices.size(0), selected_indices.size(1), selected_indices.size(2), layer_val.size(2)}));
  auto edges = scan_cuda_compress_scores(compress_name, selected_scores);
  auto delta_state =
      (edges.to(torch::kFloat32) * selected_state.to(torch::kFloat32)).sum(-1);
  auto delta_val =
      (edges.to(torch::kFloat32).unsqueeze(-1) * selected_val.to(torch::kFloat32)).sum(-2);
  return {
      delta_state.to(layer_state.scalar_type()),
      delta_val.to(layer_val.scalar_type()),
  };
}

torch::Tensor scan_cuda_read_memory_vector(
    const std::vector<ScanCudaLayerState>& memory_state,
    const std::vector<torch::Tensor>& val_norm_weights,
    const std::vector<torch::Tensor>& val_norm_biases,
    const torch::Tensor& read_template_val,
    const std::vector<torch::Tensor>& read_projection_weights,
    const std::vector<torch::Tensor>& read_gates) {
  torch::Tensor read_sum;
  bool has_term = false;
  for (size_t index = 0; index < memory_state.size(); ++index) {
    auto sender_strength = torch::softplus(memory_state[index].state).unsqueeze(-1);
    auto read_summary = (sender_strength * memory_state[index].val).sum(1);
    read_summary = read_summary + read_template_val.to(read_summary.scalar_type()).unsqueeze(0);
    auto projected = scan_cuda_linear2d(read_summary, read_projection_weights[index], torch::Tensor());
    auto gate = torch::sigmoid(read_gates[index].to(read_summary.scalar_type()));
    auto term = gate * projected;
    if (!has_term) {
      read_sum = term;
      has_term = true;
    } else {
      read_sum = read_sum + term;
    }
  }
  if (!has_term) {
    throw std::runtime_error("scan_cuda_read_memory_vector requires at least one memory level.");
  }
  return read_sum;
}

ScanCudaLayerState scan_cuda_apply_ffn_to_layer(
    const ScanCudaLayerState& layer,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& in_weight,
    const torch::Tensor& in_bias,
    const torch::Tensor& out_weight,
    const torch::Tensor& out_bias,
    bool use_fused_forward) {
  if (!in_weight.defined() || in_weight.numel() == 0) {
    return layer;
  }
  auto normalized = scan_cuda_layer_norm_last_dim(layer.val, norm_weight, norm_bias);
  if (use_fused_forward &&
      normalized.is_cuda() &&
      normalized.is_contiguous() &&
      layer.val.is_contiguous()) {
    auto hidden = scan_cuda_linear3d(normalized, in_weight, torch::Tensor()).contiguous();
    auto hidden_bias =
        (in_bias.defined() && in_bias.numel() != 0 && in_bias.scalar_type() != hidden.scalar_type())
            ? in_bias.to(hidden.scalar_type()).contiguous()
            : in_bias;
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto hidden_total = hidden.numel();
    const bool has_hidden_bias = hidden_bias.defined() && hidden_bias.numel() != 0;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        hidden.scalar_type(),
        "scan_bias_gelu_kernel",
        [&] {
          scan_bias_gelu_kernel<scalar_t>
              <<<(hidden_total + threads - 1) / threads, threads, 0, stream>>>(
                  hidden.data_ptr<scalar_t>(),
                  has_hidden_bias ? hidden_bias.data_ptr<scalar_t>() : nullptr,
                  hidden_total,
                  hidden.size(-1),
                  has_hidden_bias);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    auto delta = scan_cuda_linear3d(hidden, out_weight, torch::Tensor())
                     .to(layer.val.scalar_type())
                     .contiguous();
    auto output_bias =
        (out_bias.defined() && out_bias.numel() != 0 && out_bias.scalar_type() != delta.scalar_type())
            ? out_bias.to(delta.scalar_type()).contiguous()
            : out_bias;
    const auto delta_total = delta.numel();
    const bool has_output_bias = output_bias.defined() && output_bias.numel() != 0;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        delta.scalar_type(),
        "scan_residual_bias_add_kernel",
        [&] {
          scan_residual_bias_add_kernel<scalar_t>
              <<<(delta_total + threads - 1) / threads, threads, 0, stream>>>(
                  layer.val.data_ptr<scalar_t>(),
                  delta.data_ptr<scalar_t>(),
                  has_output_bias ? output_bias.data_ptr<scalar_t>() : nullptr,
                  delta_total,
                  delta.size(-1),
                  has_output_bias);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return {layer.state, delta};
  }
  auto hidden = scan_cuda_linear3d(normalized, in_weight, in_bias);
  auto activated = 0.5 * hidden * (1 + torch::erf(hidden * 0.70710678118654752440));
  auto delta = scan_cuda_linear3d(activated, out_weight, out_bias);
  return {layer.state, layer.val + delta};
}

void scan_cuda_require_vector_size(
    const std::vector<torch::Tensor>& tensors,
    size_t expected,
    const std::string& name) {
  if (tensors.size() != expected) {
    throw std::runtime_error(
        name + " has unexpected length: got " + std::to_string(tensors.size()) +
        ", expected " + std::to_string(expected) + ".");
  }
}

void scan_cuda_require_int_vector_size(
    const std::vector<int64_t>& values,
    size_t expected,
    const std::string& name) {
  if (values.size() != expected) {
    throw std::runtime_error(
        name + " has unexpected length: got " + std::to_string(values.size()) +
        ", expected " + std::to_string(expected) + ".");
  }
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
scan_cuda_forward_impl(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const torch::Tensor& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const torch::Tensor& prediction_input_norm_bias,
    const torch::Tensor& read_template_val,
    const std::vector<torch::Tensor>& read_projection_weights,
    const std::vector<torch::Tensor>& read_gates,
    const std::vector<torch::Tensor>& write_source_weights,
    const std::vector<torch::Tensor>& write_target_weights,
    const std::vector<torch::Tensor>& write_core_weights,
    const std::vector<torch::Tensor>& write_biases,
    const std::vector<int64_t>& write_topks,
    const std::string& transition_compress_name,
    const std::vector<torch::Tensor>& propagation_source_weights,
    const std::vector<torch::Tensor>& propagation_target_weights,
    const std::vector<torch::Tensor>& propagation_core_weights,
    const std::vector<torch::Tensor>& propagation_biases,
    const std::vector<int64_t>& propagation_topks,
    const std::string& propagation_compress_name,
    const std::vector<torch::Tensor>& val_norm_weights,
    const std::vector<torch::Tensor>& val_norm_biases,
    const std::vector<torch::Tensor>& level_transition_source_weights,
    const std::vector<torch::Tensor>& level_transition_target_weights,
    const std::vector<torch::Tensor>& level_transition_core_weights,
    const std::vector<torch::Tensor>& level_transition_biases,
    const std::vector<int64_t>& level_transition_topks,
    const std::vector<torch::Tensor>& level_norm_weights,
    const std::vector<torch::Tensor>& level_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_norm_weights,
    const std::vector<torch::Tensor>& level_ffn_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_in_weights,
    const std::vector<torch::Tensor>& level_ffn_in_biases,
    const std::vector<torch::Tensor>& level_ffn_out_weights,
    const std::vector<torch::Tensor>& level_ffn_out_biases,
    const std::vector<torch::Tensor>& skip_source_weights,
    const std::vector<torch::Tensor>& skip_target_weights,
    const std::vector<torch::Tensor>& skip_core_weights,
    const std::vector<torch::Tensor>& skip_biases,
    const std::vector<torch::Tensor>& skip_gates,
    const std::vector<int64_t>& skip_topks,
    bool collect_trace) {
  if (!aligned_s.is_cuda()) {
    throw std::runtime_error("scan_cuda_forward_impl requires CUDA inputs.");
  }
  if (aligned_s.dim() != 3) {
    throw std::runtime_error("aligned_s must have shape [batch, seq_len, dim].");
  }
  if (flat_memory.size() % 2 != 0) {
    throw std::runtime_error("flat_memory must contain alternating state/val tensors.");
  }
  const auto num_levels = flat_memory.size() / 2;
  const auto expected_skip_count = num_levels > 1 ? num_levels - 1 : 0;
  scan_cuda_require_vector_size(read_projection_weights, num_levels, "read_projection_weights");
  scan_cuda_require_vector_size(read_gates, num_levels, "read_gates");
  scan_cuda_require_vector_size(write_source_weights, num_levels, "write_source_weights");
  scan_cuda_require_vector_size(write_target_weights, num_levels, "write_target_weights");
  scan_cuda_require_vector_size(write_core_weights, num_levels, "write_core_weights");
  scan_cuda_require_vector_size(write_biases, num_levels, "write_biases");
  scan_cuda_require_int_vector_size(write_topks, num_levels, "write_topks");
  scan_cuda_require_vector_size(propagation_source_weights, num_levels, "propagation_source_weights");
  scan_cuda_require_vector_size(propagation_target_weights, num_levels, "propagation_target_weights");
  scan_cuda_require_vector_size(propagation_core_weights, num_levels, "propagation_core_weights");
  scan_cuda_require_vector_size(propagation_biases, num_levels, "propagation_biases");
  scan_cuda_require_int_vector_size(propagation_topks, num_levels, "propagation_topks");
  scan_cuda_require_vector_size(val_norm_weights, num_levels, "val_norm_weights");
  scan_cuda_require_vector_size(val_norm_biases, num_levels, "val_norm_biases");
  scan_cuda_require_vector_size(level_transition_source_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_source_weights");
  scan_cuda_require_vector_size(level_transition_target_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_target_weights");
  scan_cuda_require_vector_size(level_transition_core_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_core_weights");
  scan_cuda_require_vector_size(level_transition_biases, num_levels > 0 ? num_levels - 1 : 0, "level_transition_biases");
  scan_cuda_require_int_vector_size(level_transition_topks, num_levels > 0 ? num_levels - 1 : 0, "level_transition_topks");
  scan_cuda_require_vector_size(level_norm_weights, num_levels, "level_norm_weights");
  scan_cuda_require_vector_size(level_norm_biases, num_levels, "level_norm_biases");
  scan_cuda_require_vector_size(level_ffn_norm_weights, num_levels, "level_ffn_norm_weights");
  scan_cuda_require_vector_size(level_ffn_norm_biases, num_levels, "level_ffn_norm_biases");
  scan_cuda_require_vector_size(level_ffn_in_weights, num_levels, "level_ffn_in_weights");
  scan_cuda_require_vector_size(level_ffn_in_biases, num_levels, "level_ffn_in_biases");
  scan_cuda_require_vector_size(level_ffn_out_weights, num_levels, "level_ffn_out_weights");
  scan_cuda_require_vector_size(level_ffn_out_biases, num_levels, "level_ffn_out_biases");
  scan_cuda_require_vector_size(skip_source_weights, expected_skip_count, "skip_source_weights");
  scan_cuda_require_vector_size(skip_target_weights, expected_skip_count, "skip_target_weights");
  scan_cuda_require_vector_size(skip_core_weights, expected_skip_count, "skip_core_weights");
  scan_cuda_require_vector_size(skip_biases, expected_skip_count, "skip_biases");
  scan_cuda_require_vector_size(skip_gates, expected_skip_count, "skip_gates");
  scan_cuda_require_int_vector_size(skip_topks, expected_skip_count, "skip_topks");

  std::vector<ScanCudaLayerState> current_memory;
  current_memory.reserve(num_levels);
  for (size_t index = 0; index < num_levels; ++index) {
    current_memory.push_back({flat_memory[index * 2], flat_memory[index * 2 + 1]});
  }

  std::vector<std::vector<torch::Tensor>> trace_states(num_levels);
  std::vector<std::vector<torch::Tensor>> trace_vals(num_levels);
  auto projected_s = scan_cuda_linear3d(aligned_s, s_prediction_weight, torch::Tensor()).contiguous();
  auto query_val = torch::empty_like(projected_s);

  for (int64_t time_index = 0; time_index < aligned_s.size(1); ++time_index) {
    if (collect_trace) {
      for (size_t trace_index = 0; trace_index < num_levels; ++trace_index) {
        trace_states[trace_index].push_back(current_memory[trace_index].state);
        trace_vals[trace_index].push_back(current_memory[trace_index].val);
      }
    }
    auto token_val = aligned_s.slice(1, time_index, time_index + 1);
    auto token_state = scan_cuda_value_to_state3d(token_val, value_to_state_weight, value_to_state_bias);
    ScanCudaLayerState token_layer{token_state, token_val};

    std::vector<ScanCudaLayerState> next_memory;
    next_memory.reserve(num_levels);

    auto first_write_delta = scan_cuda_low_rank_transition_pairwise_topk(
        torch::softplus(token_layer.state),
        token_layer.state,
        token_layer.val,
        token_layer.val,
        current_memory[0].val,
        write_source_weights[0],
        write_target_weights[0],
        write_core_weights[0],
        write_biases[0],
        write_topks[0],
        transition_compress_name,
        true);
    auto level = scan_cuda_apply_delta_to_layer(
        current_memory[0],
        std::get<0>(first_write_delta),
        std::get<1>(first_write_delta),
        val_norm_weights[0],
        val_norm_biases[0],
        true);
    level = scan_cuda_apply_ffn_to_layer(
        level,
        level_ffn_norm_weights[0],
        level_ffn_norm_biases[0],
        level_ffn_in_weights[0],
        level_ffn_in_biases[0],
        level_ffn_out_weights[0],
        level_ffn_out_biases[0],
        true);
    if (propagation_topks[0] >= 0) {
      auto first_prop_delta = scan_cuda_low_rank_propagation_topk(
          level.state,
          level.val,
          propagation_source_weights[0],
          propagation_target_weights[0],
          propagation_core_weights[0],
          propagation_biases[0],
          propagation_topks[0],
          propagation_compress_name,
          true);
      level = scan_cuda_apply_delta_to_layer(
          level,
          std::get<0>(first_prop_delta),
          std::get<1>(first_prop_delta),
          val_norm_weights[0],
          val_norm_biases[0],
          true);
      level = scan_cuda_apply_ffn_to_layer(
          level,
          level_ffn_norm_weights[0],
          level_ffn_norm_biases[0],
          level_ffn_in_weights[0],
          level_ffn_in_biases[0],
          level_ffn_out_weights[0],
          level_ffn_out_biases[0],
          true);
    }
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < num_levels; ++level_index) {
      auto current_level = current_memory[level_index];
      auto parent_delta = scan_cuda_low_rank_transition_pairwise_topk(
          torch::softplus(next_memory[level_index - 1].state),
          next_memory[level_index - 1].state,
          next_memory[level_index - 1].val,
          next_memory[level_index - 1].val,
          current_level.val,
          level_transition_source_weights[level_index - 1],
          level_transition_target_weights[level_index - 1],
          level_transition_core_weights[level_index - 1],
          level_transition_biases[level_index - 1],
          level_transition_topks[level_index - 1],
          transition_compress_name,
          true);
      auto updated_level = scan_cuda_apply_delta_to_layer(
          current_level,
          std::get<0>(parent_delta),
          std::get<1>(parent_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index],
          true);
      updated_level = scan_cuda_apply_ffn_to_layer(
          updated_level,
          level_ffn_norm_weights[level_index],
          level_ffn_norm_biases[level_index],
          level_ffn_in_weights[level_index],
          level_ffn_in_biases[level_index],
          level_ffn_out_weights[level_index],
          level_ffn_out_biases[level_index],
          true);

      if (level_index == 1 && expected_skip_count > 0) {
        auto skip_gate = torch::sigmoid(skip_gates[0].to(token_val.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(token_layer.state),
            token_layer.state,
            token_layer.val,
            token_layer.val,
            current_level.val,
            skip_source_weights[0],
            skip_target_weights[0],
            skip_core_weights[0],
            skip_biases[0],
            skip_topks[0],
            transition_compress_name,
            true);
        updated_level = scan_cuda_apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index],
            true);
        updated_level = scan_cuda_apply_ffn_to_layer(
            updated_level,
            level_ffn_norm_weights[level_index],
            level_ffn_norm_biases[level_index],
            level_ffn_in_weights[level_index],
            level_ffn_in_biases[level_index],
            level_ffn_out_weights[level_index],
            level_ffn_out_biases[level_index],
            true);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto skip_gate = torch::sigmoid(skip_gates[skip_index].to(next_memory[level_index - 2].val.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(next_memory[level_index - 2].state),
            next_memory[level_index - 2].state,
            next_memory[level_index - 2].val,
            next_memory[level_index - 2].val,
            current_level.val,
            skip_source_weights[skip_index],
            skip_target_weights[skip_index],
            skip_core_weights[skip_index],
            skip_biases[skip_index],
            skip_topks[skip_index],
            transition_compress_name,
            true);
        updated_level = scan_cuda_apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index],
            true);
        updated_level = scan_cuda_apply_ffn_to_layer(
            updated_level,
            level_ffn_norm_weights[level_index],
            level_ffn_norm_biases[level_index],
            level_ffn_in_weights[level_index],
            level_ffn_in_biases[level_index],
            level_ffn_out_weights[level_index],
            level_ffn_out_biases[level_index],
            true);
      }

      if (propagation_topks[level_index] >= 0) {
        auto propagation_delta = scan_cuda_low_rank_propagation_topk(
            updated_level.state,
            updated_level.val,
            propagation_source_weights[level_index],
            propagation_target_weights[level_index],
            propagation_core_weights[level_index],
            propagation_biases[level_index],
            propagation_topks[level_index],
            propagation_compress_name,
            true);
        updated_level = scan_cuda_apply_delta_to_layer(
            updated_level,
            std::get<0>(propagation_delta),
            std::get<1>(propagation_delta),
            val_norm_weights[level_index],
            val_norm_biases[level_index],
            true);
        updated_level = scan_cuda_apply_ffn_to_layer(
            updated_level,
            level_ffn_norm_weights[level_index],
            level_ffn_norm_biases[level_index],
            level_ffn_in_weights[level_index],
            level_ffn_in_biases[level_index],
            level_ffn_out_weights[level_index],
            level_ffn_out_biases[level_index],
            true);
      }
      next_memory.push_back(updated_level);
    }

    current_memory = next_memory;
    auto read_vector = scan_cuda_read_memory_vector(
        current_memory,
        val_norm_weights,
        val_norm_biases,
        read_template_val,
        read_projection_weights,
        read_gates);
    auto query_input = projected_s.select(1, time_index) + read_vector;
    query_val.select(1, time_index).copy_(
        scan_cuda_layer_norm_last_dim(
            query_input,
            prediction_input_norm_weight,
            prediction_input_norm_bias));
  }
  std::vector<torch::Tensor> next_memory_flat;
  next_memory_flat.reserve(current_memory.size() * 2);
  for (const auto& layer : current_memory) {
    next_memory_flat.push_back(layer.state);
    next_memory_flat.push_back(layer.val);
  }

  std::vector<torch::Tensor> trace_flat;
  if (collect_trace) {
    trace_flat.reserve(num_levels * 2);
    for (size_t trace_index = 0; trace_index < num_levels; ++trace_index) {
      trace_flat.push_back(torch::stack(trace_states[trace_index], 0));
      trace_flat.push_back(torch::stack(trace_vals[trace_index], 0));
    }
  }
  return {query_val, next_memory_flat, trace_flat};
}

}  // namespace

std::tuple<torch::Tensor, std::vector<torch::Tensor>>
jakal_net_causal_memory_scan_fused_forward_cuda(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const torch::Tensor& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const torch::Tensor& prediction_input_norm_bias,
    const torch::Tensor& read_template_val,
    const std::vector<torch::Tensor>& read_projection_weights,
    const std::vector<torch::Tensor>& read_gates,
    const std::vector<torch::Tensor>& write_source_weights,
    const std::vector<torch::Tensor>& write_target_weights,
    const std::vector<torch::Tensor>& write_core_weights,
    const std::vector<torch::Tensor>& write_biases,
    const std::vector<int64_t>& write_topks,
    const std::string& transition_compress_name,
    const std::vector<torch::Tensor>& propagation_source_weights,
    const std::vector<torch::Tensor>& propagation_target_weights,
    const std::vector<torch::Tensor>& propagation_core_weights,
    const std::vector<torch::Tensor>& propagation_biases,
    const std::vector<int64_t>& propagation_topks,
    const std::string& propagation_compress_name,
    const std::vector<torch::Tensor>& val_norm_weights,
    const std::vector<torch::Tensor>& val_norm_biases,
    const std::vector<torch::Tensor>& level_transition_source_weights,
    const std::vector<torch::Tensor>& level_transition_target_weights,
    const std::vector<torch::Tensor>& level_transition_core_weights,
    const std::vector<torch::Tensor>& level_transition_biases,
    const std::vector<int64_t>& level_transition_topks,
    const std::vector<torch::Tensor>& level_norm_weights,
    const std::vector<torch::Tensor>& level_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_norm_weights,
    const std::vector<torch::Tensor>& level_ffn_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_in_weights,
    const std::vector<torch::Tensor>& level_ffn_in_biases,
    const std::vector<torch::Tensor>& level_ffn_out_weights,
    const std::vector<torch::Tensor>& level_ffn_out_biases,
    const std::vector<torch::Tensor>& skip_source_weights,
    const std::vector<torch::Tensor>& skip_target_weights,
    const std::vector<torch::Tensor>& skip_core_weights,
    const std::vector<torch::Tensor>& skip_biases,
    const std::vector<torch::Tensor>& skip_gates,
    const std::vector<int64_t>& skip_topks) {
  auto result = scan_cuda_forward_impl(
      aligned_s,
      flat_memory,
      value_to_state_weight,
      value_to_state_bias,
      s_prediction_weight,
      prediction_input_norm_weight,
      prediction_input_norm_bias,
      read_template_val,
      read_projection_weights,
      read_gates,
      write_source_weights,
      write_target_weights,
      write_core_weights,
      write_biases,
      write_topks,
      transition_compress_name,
      propagation_source_weights,
      propagation_target_weights,
      propagation_core_weights,
      propagation_biases,
      propagation_topks,
      propagation_compress_name,
      val_norm_weights,
      val_norm_biases,
      level_transition_source_weights,
      level_transition_target_weights,
      level_transition_core_weights,
      level_transition_biases,
      level_transition_topks,
      level_norm_weights,
      level_norm_biases,
      level_ffn_norm_weights,
      level_ffn_norm_biases,
      level_ffn_in_weights,
      level_ffn_in_biases,
      level_ffn_out_weights,
      level_ffn_out_biases,
      skip_source_weights,
      skip_target_weights,
      skip_core_weights,
      skip_biases,
      skip_gates,
      skip_topks,
      false);
  return {std::get<0>(result), std::get<1>(result)};
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
jakal_net_causal_memory_scan_fused_trace_cuda(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const torch::Tensor& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const torch::Tensor& prediction_input_norm_bias,
    const torch::Tensor& read_template_val,
    const std::vector<torch::Tensor>& read_projection_weights,
    const std::vector<torch::Tensor>& read_gates,
    const std::vector<torch::Tensor>& write_source_weights,
    const std::vector<torch::Tensor>& write_target_weights,
    const std::vector<torch::Tensor>& write_core_weights,
    const std::vector<torch::Tensor>& write_biases,
    const std::vector<int64_t>& write_topks,
    const std::string& transition_compress_name,
    const std::vector<torch::Tensor>& propagation_source_weights,
    const std::vector<torch::Tensor>& propagation_target_weights,
    const std::vector<torch::Tensor>& propagation_core_weights,
    const std::vector<torch::Tensor>& propagation_biases,
    const std::vector<int64_t>& propagation_topks,
    const std::string& propagation_compress_name,
    const std::vector<torch::Tensor>& val_norm_weights,
    const std::vector<torch::Tensor>& val_norm_biases,
    const std::vector<torch::Tensor>& level_transition_source_weights,
    const std::vector<torch::Tensor>& level_transition_target_weights,
    const std::vector<torch::Tensor>& level_transition_core_weights,
    const std::vector<torch::Tensor>& level_transition_biases,
    const std::vector<int64_t>& level_transition_topks,
    const std::vector<torch::Tensor>& level_norm_weights,
    const std::vector<torch::Tensor>& level_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_norm_weights,
    const std::vector<torch::Tensor>& level_ffn_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_in_weights,
    const std::vector<torch::Tensor>& level_ffn_in_biases,
    const std::vector<torch::Tensor>& level_ffn_out_weights,
    const std::vector<torch::Tensor>& level_ffn_out_biases,
    const std::vector<torch::Tensor>& skip_source_weights,
    const std::vector<torch::Tensor>& skip_target_weights,
    const std::vector<torch::Tensor>& skip_core_weights,
    const std::vector<torch::Tensor>& skip_biases,
    const std::vector<torch::Tensor>& skip_gates,
    const std::vector<int64_t>& skip_topks) {
  return scan_cuda_forward_impl(
      aligned_s,
      flat_memory,
      value_to_state_weight,
      value_to_state_bias,
      s_prediction_weight,
      prediction_input_norm_weight,
      prediction_input_norm_bias,
      read_template_val,
      read_projection_weights,
      read_gates,
      write_source_weights,
      write_target_weights,
      write_core_weights,
      write_biases,
      write_topks,
      transition_compress_name,
      propagation_source_weights,
      propagation_target_weights,
      propagation_core_weights,
      propagation_biases,
      propagation_topks,
      propagation_compress_name,
      val_norm_weights,
      val_norm_biases,
      level_transition_source_weights,
      level_transition_target_weights,
      level_transition_core_weights,
      level_transition_biases,
      level_transition_topks,
      level_norm_weights,
      level_norm_biases,
      level_ffn_norm_weights,
      level_ffn_norm_biases,
      level_ffn_in_weights,
      level_ffn_in_biases,
      level_ffn_out_weights,
      level_ffn_out_biases,
      skip_source_weights,
      skip_target_weights,
      skip_core_weights,
      skip_biases,
      skip_gates,
      skip_topks,
      true);
}

std::vector<torch::Tensor> jakal_net_causal_memory_scan_fused_backward_cuda(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const torch::Tensor& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const torch::Tensor& prediction_input_norm_bias,
    const torch::Tensor& read_template_val,
    const std::vector<torch::Tensor>& read_projection_weights,
    const std::vector<torch::Tensor>& read_gates,
    const std::vector<torch::Tensor>& write_source_weights,
    const std::vector<torch::Tensor>& write_target_weights,
    const std::vector<torch::Tensor>& write_core_weights,
    const std::vector<torch::Tensor>& write_biases,
    const std::vector<int64_t>& write_topks,
    const std::string& transition_compress_name,
    const std::vector<torch::Tensor>& propagation_source_weights,
    const std::vector<torch::Tensor>& propagation_target_weights,
    const std::vector<torch::Tensor>& propagation_core_weights,
    const std::vector<torch::Tensor>& propagation_biases,
    const std::vector<int64_t>& propagation_topks,
    const std::string& propagation_compress_name,
    const std::vector<torch::Tensor>& val_norm_weights,
    const std::vector<torch::Tensor>& val_norm_biases,
    const std::vector<torch::Tensor>& level_transition_source_weights,
    const std::vector<torch::Tensor>& level_transition_target_weights,
    const std::vector<torch::Tensor>& level_transition_core_weights,
    const std::vector<torch::Tensor>& level_transition_biases,
    const std::vector<int64_t>& level_transition_topks,
    const std::vector<torch::Tensor>& level_norm_weights,
    const std::vector<torch::Tensor>& level_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_norm_weights,
    const std::vector<torch::Tensor>& level_ffn_norm_biases,
    const std::vector<torch::Tensor>& level_ffn_in_weights,
    const std::vector<torch::Tensor>& level_ffn_in_biases,
    const std::vector<torch::Tensor>& level_ffn_out_weights,
    const std::vector<torch::Tensor>& level_ffn_out_biases,
    const std::vector<torch::Tensor>& skip_source_weights,
    const std::vector<torch::Tensor>& skip_target_weights,
    const std::vector<torch::Tensor>& skip_core_weights,
    const std::vector<torch::Tensor>& skip_biases,
    const std::vector<torch::Tensor>& skip_gates,
    const std::vector<int64_t>& skip_topks,
    const std::vector<torch::Tensor>& trace_tensors,
    const torch::Tensor& grad_query_val,
    const std::vector<torch::Tensor>& grad_next_memory) {
  c10::AutoGradMode enable_grad(true);
  std::vector<torch::Tensor> local_trace_tensors = trace_tensors;
  if (local_trace_tensors.empty()) {
    auto trace_result = scan_cuda_forward_impl(
        aligned_s,
        flat_memory,
        value_to_state_weight,
        value_to_state_bias,
        s_prediction_weight,
        prediction_input_norm_weight,
        prediction_input_norm_bias,
        read_template_val,
        read_projection_weights,
        read_gates,
        write_source_weights,
        write_target_weights,
        write_core_weights,
        write_biases,
        write_topks,
        transition_compress_name,
        propagation_source_weights,
        propagation_target_weights,
        propagation_core_weights,
        propagation_biases,
        propagation_topks,
        propagation_compress_name,
        val_norm_weights,
        val_norm_biases,
        level_transition_source_weights,
        level_transition_target_weights,
        level_transition_core_weights,
        level_transition_biases,
        level_transition_topks,
        level_norm_weights,
        level_norm_biases,
        level_ffn_norm_weights,
        level_ffn_norm_biases,
        level_ffn_in_weights,
        level_ffn_in_biases,
        level_ffn_out_weights,
        level_ffn_out_biases,
        skip_source_weights,
        skip_target_weights,
        skip_core_weights,
        skip_biases,
        skip_gates,
        skip_topks,
        true);
    local_trace_tensors = std::get<2>(trace_result);
  }

  auto make_leaf = [](const torch::Tensor& tensor) {
    auto leaf = tensor.detach();
    leaf.set_requires_grad(tensor.requires_grad());
    return leaf;
  };
  auto accumulate = [](torch::Tensor& dst, const torch::Tensor& grad) {
    if (!grad.defined()) {
      return;
    }
    if (!dst.defined()) {
      dst = grad;
    } else {
      dst = dst + grad;
    }
  };

  auto value_to_state_weight_leaf = make_leaf(value_to_state_weight);
  auto value_to_state_bias_leaf = make_leaf(value_to_state_bias);
  auto s_prediction_weight_leaf = make_leaf(s_prediction_weight);
  auto prediction_input_norm_weight_leaf = make_leaf(prediction_input_norm_weight);
  auto prediction_input_norm_bias_leaf = make_leaf(prediction_input_norm_bias);
  auto read_template_val_leaf = make_leaf(read_template_val);
  std::vector<torch::Tensor> read_projection_weights_leaves;
  std::vector<torch::Tensor> read_gates_leaves;
  std::vector<torch::Tensor> write_source_weights_leaves;
  std::vector<torch::Tensor> write_target_weights_leaves;
  std::vector<torch::Tensor> write_core_weights_leaves;
  std::vector<torch::Tensor> write_biases_leaves;
  std::vector<torch::Tensor> propagation_source_weights_leaves;
  std::vector<torch::Tensor> propagation_target_weights_leaves;
  std::vector<torch::Tensor> propagation_core_weights_leaves;
  std::vector<torch::Tensor> propagation_biases_leaves;
  std::vector<torch::Tensor> val_norm_weights_leaves;
  std::vector<torch::Tensor> val_norm_biases_leaves;
  std::vector<torch::Tensor> level_transition_source_weights_leaves;
  std::vector<torch::Tensor> level_transition_target_weights_leaves;
  std::vector<torch::Tensor> level_transition_core_weights_leaves;
  std::vector<torch::Tensor> level_transition_biases_leaves;
  std::vector<torch::Tensor> level_norm_weights_leaves;
  std::vector<torch::Tensor> level_norm_biases_leaves;
  std::vector<torch::Tensor> level_ffn_norm_weights_leaves;
  std::vector<torch::Tensor> level_ffn_norm_biases_leaves;
  std::vector<torch::Tensor> level_ffn_in_weights_leaves;
  std::vector<torch::Tensor> level_ffn_in_biases_leaves;
  std::vector<torch::Tensor> level_ffn_out_weights_leaves;
  std::vector<torch::Tensor> level_ffn_out_biases_leaves;
  std::vector<torch::Tensor> skip_source_weights_leaves;
  std::vector<torch::Tensor> skip_target_weights_leaves;
  std::vector<torch::Tensor> skip_core_weights_leaves;
  std::vector<torch::Tensor> skip_biases_leaves;
  std::vector<torch::Tensor> skip_gates_leaves;
  for (const auto& tensor : read_projection_weights) read_projection_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : read_gates) read_gates_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : write_source_weights) write_source_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : write_target_weights) write_target_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : write_core_weights) write_core_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : write_biases) write_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : propagation_source_weights) propagation_source_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : propagation_target_weights) propagation_target_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : propagation_core_weights) propagation_core_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : propagation_biases) propagation_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : val_norm_weights) val_norm_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : val_norm_biases) val_norm_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_transition_source_weights) level_transition_source_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_transition_target_weights) level_transition_target_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_transition_core_weights) level_transition_core_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_transition_biases) level_transition_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_norm_weights) level_norm_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_norm_biases) level_norm_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_ffn_norm_weights) level_ffn_norm_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_ffn_norm_biases) level_ffn_norm_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_ffn_in_weights) level_ffn_in_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_ffn_in_biases) level_ffn_in_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_ffn_out_weights) level_ffn_out_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : level_ffn_out_biases) level_ffn_out_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : skip_source_weights) skip_source_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : skip_target_weights) skip_target_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : skip_core_weights) skip_core_weights_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : skip_biases) skip_biases_leaves.push_back(make_leaf(tensor));
  for (const auto& tensor : skip_gates) skip_gates_leaves.push_back(make_leaf(tensor));

  std::vector<torch::Tensor> carry_memory = grad_next_memory;
  std::vector<torch::Tensor> shared_grad_accum;
  auto aligned_s_grad = aligned_s.requires_grad() ? torch::zeros_like(aligned_s) : torch::Tensor();

  for (int64_t time_index = aligned_s.size(1) - 1; time_index >= 0; --time_index) {
    auto token_val_leaf = make_leaf(aligned_s.slice(1, time_index, time_index + 1));
    std::vector<ScanCudaLayerState> current_memory;
    std::vector<torch::Tensor> current_memory_leaves;
    current_memory.reserve(flat_memory.size() / 2);
    current_memory_leaves.reserve(flat_memory.size());
    for (size_t level_index = 0; level_index < flat_memory.size() / 2; ++level_index) {
      auto state_leaf = local_trace_tensors[level_index * 2].select(0, time_index).detach();
      state_leaf.set_requires_grad(flat_memory[level_index * 2].requires_grad());
      auto val_leaf = local_trace_tensors[level_index * 2 + 1].select(0, time_index).detach();
      val_leaf.set_requires_grad(flat_memory[level_index * 2 + 1].requires_grad());
      current_memory.push_back({state_leaf, val_leaf});
      current_memory_leaves.push_back(state_leaf);
      current_memory_leaves.push_back(val_leaf);
    }

    auto token_state = scan_cuda_value_to_state3d(
        token_val_leaf,
        value_to_state_weight_leaf,
        value_to_state_bias_leaf);
    auto projected_s_t = scan_cuda_linear3d(token_val_leaf, s_prediction_weight_leaf, torch::Tensor()).squeeze(1);

    std::vector<ScanCudaLayerState> next_memory;
    next_memory.reserve(current_memory.size());
    auto first_write_delta = scan_cuda_low_rank_transition_pairwise_topk(
        torch::softplus(token_state),
        token_state,
        token_val_leaf,
        token_val_leaf,
        current_memory[0].val,
        write_source_weights_leaves[0],
        write_target_weights_leaves[0],
        write_core_weights_leaves[0],
        write_biases_leaves[0],
        write_topks[0],
        transition_compress_name,
        false);
    auto level = scan_cuda_apply_delta_to_layer(
        current_memory[0],
        std::get<0>(first_write_delta),
        std::get<1>(first_write_delta),
        val_norm_weights_leaves[0],
        val_norm_biases_leaves[0],
        false);
    level = scan_cuda_apply_ffn_to_layer(
        level,
        level_ffn_norm_weights_leaves[0],
        level_ffn_norm_biases_leaves[0],
        level_ffn_in_weights_leaves[0],
        level_ffn_in_biases_leaves[0],
        level_ffn_out_weights_leaves[0],
        level_ffn_out_biases_leaves[0],
        false);
    if (propagation_topks[0] >= 0) {
      auto first_prop_delta = scan_cuda_low_rank_propagation_topk(
          level.state,
          level.val,
          propagation_source_weights_leaves[0],
          propagation_target_weights_leaves[0],
          propagation_core_weights_leaves[0],
          propagation_biases_leaves[0],
          propagation_topks[0],
          propagation_compress_name,
          false);
      level = scan_cuda_apply_delta_to_layer(
          level,
          std::get<0>(first_prop_delta),
          std::get<1>(first_prop_delta),
          val_norm_weights_leaves[0],
          val_norm_biases_leaves[0],
          false);
      level = scan_cuda_apply_ffn_to_layer(
          level,
          level_ffn_norm_weights_leaves[0],
          level_ffn_norm_biases_leaves[0],
          level_ffn_in_weights_leaves[0],
          level_ffn_in_biases_leaves[0],
          level_ffn_out_weights_leaves[0],
          level_ffn_out_biases_leaves[0],
          false);
    }
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < current_memory.size(); ++level_index) {
      auto current_level = current_memory[level_index];
      auto parent_delta = scan_cuda_low_rank_transition_pairwise_topk(
          torch::softplus(next_memory[level_index - 1].state),
          next_memory[level_index - 1].state,
          next_memory[level_index - 1].val,
          next_memory[level_index - 1].val,
          current_level.val,
          level_transition_source_weights_leaves[level_index - 1],
          level_transition_target_weights_leaves[level_index - 1],
          level_transition_core_weights_leaves[level_index - 1],
          level_transition_biases_leaves[level_index - 1],
          level_transition_topks[level_index - 1],
          transition_compress_name,
          false);
      auto updated_level = scan_cuda_apply_delta_to_layer(
          current_level,
          std::get<0>(parent_delta),
          std::get<1>(parent_delta),
          val_norm_weights_leaves[level_index],
          val_norm_biases_leaves[level_index],
          false);
      updated_level = scan_cuda_apply_ffn_to_layer(
          updated_level,
          level_ffn_norm_weights_leaves[level_index],
          level_ffn_norm_biases_leaves[level_index],
          level_ffn_in_weights_leaves[level_index],
          level_ffn_in_biases_leaves[level_index],
          level_ffn_out_weights_leaves[level_index],
          level_ffn_out_biases_leaves[level_index],
          false);

      if (level_index == 1 && !skip_source_weights_leaves.empty()) {
        auto skip_gate = torch::sigmoid(skip_gates_leaves[0].to(token_val_leaf.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(token_state),
            token_state,
            token_val_leaf,
            token_val_leaf,
            current_level.val,
            skip_source_weights_leaves[0],
            skip_target_weights_leaves[0],
            skip_core_weights_leaves[0],
            skip_biases_leaves[0],
            skip_topks[0],
            transition_compress_name,
            false);
        updated_level = scan_cuda_apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights_leaves[level_index],
            val_norm_biases_leaves[level_index],
            false);
        updated_level = scan_cuda_apply_ffn_to_layer(
            updated_level,
            level_ffn_norm_weights_leaves[level_index],
            level_ffn_norm_biases_leaves[level_index],
            level_ffn_in_weights_leaves[level_index],
            level_ffn_in_biases_leaves[level_index],
            level_ffn_out_weights_leaves[level_index],
            level_ffn_out_biases_leaves[level_index],
            false);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto skip_gate = torch::sigmoid(skip_gates_leaves[skip_index].to(next_memory[level_index - 2].val.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(next_memory[level_index - 2].state),
            next_memory[level_index - 2].state,
            next_memory[level_index - 2].val,
            next_memory[level_index - 2].val,
            current_level.val,
            skip_source_weights_leaves[skip_index],
            skip_target_weights_leaves[skip_index],
            skip_core_weights_leaves[skip_index],
            skip_biases_leaves[skip_index],
            skip_topks[skip_index],
            transition_compress_name,
            false);
        updated_level = scan_cuda_apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights_leaves[level_index],
            val_norm_biases_leaves[level_index],
            false);
        updated_level = scan_cuda_apply_ffn_to_layer(
            updated_level,
            level_ffn_norm_weights_leaves[level_index],
            level_ffn_norm_biases_leaves[level_index],
            level_ffn_in_weights_leaves[level_index],
            level_ffn_in_biases_leaves[level_index],
            level_ffn_out_weights_leaves[level_index],
            level_ffn_out_biases_leaves[level_index],
            false);
      }

      if (propagation_topks[level_index] >= 0) {
        auto propagation_delta = scan_cuda_low_rank_propagation_topk(
            updated_level.state,
            updated_level.val,
            propagation_source_weights_leaves[level_index],
            propagation_target_weights_leaves[level_index],
            propagation_core_weights_leaves[level_index],
            propagation_biases_leaves[level_index],
            propagation_topks[level_index],
            propagation_compress_name,
            false);
        updated_level = scan_cuda_apply_delta_to_layer(
            updated_level,
            std::get<0>(propagation_delta),
            std::get<1>(propagation_delta),
            val_norm_weights_leaves[level_index],
            val_norm_biases_leaves[level_index],
            false);
        updated_level = scan_cuda_apply_ffn_to_layer(
            updated_level,
            level_ffn_norm_weights_leaves[level_index],
            level_ffn_norm_biases_leaves[level_index],
            level_ffn_in_weights_leaves[level_index],
            level_ffn_in_biases_leaves[level_index],
            level_ffn_out_weights_leaves[level_index],
            level_ffn_out_biases_leaves[level_index],
            false);
      }
      next_memory.push_back(updated_level);
    }

    auto read_vector = scan_cuda_read_memory_vector(
        next_memory,
        val_norm_weights_leaves,
        val_norm_biases_leaves,
        read_template_val_leaf,
        read_projection_weights_leaves,
        read_gates_leaves);
    auto query_input = projected_s_t + read_vector;
    auto query_step = scan_cuda_layer_norm_last_dim(
        query_input,
        prediction_input_norm_weight_leaf,
        prediction_input_norm_bias_leaf);

    torch::autograd::variable_list outputs;
    outputs.push_back(query_step);
    for (const auto& layer : next_memory) {
      outputs.push_back(layer.state);
      outputs.push_back(layer.val);
    }
    torch::autograd::variable_list local_grad_outputs;
    local_grad_outputs.push_back(
        grad_query_val.defined()
            ? grad_query_val.select(1, time_index)
            : torch::zeros_like(query_step));
    for (size_t memory_index = 0; memory_index < carry_memory.size(); ++memory_index) {
      local_grad_outputs.push_back(
          carry_memory[memory_index].defined()
              ? carry_memory[memory_index]
              : torch::zeros_like(outputs[memory_index + 1]));
    }

    std::vector<torch::Tensor> all_inputs;
    all_inputs.push_back(token_val_leaf);
    for (const auto& tensor : current_memory_leaves) all_inputs.push_back(tensor);
    all_inputs.push_back(value_to_state_weight_leaf);
    all_inputs.push_back(value_to_state_bias_leaf);
    all_inputs.push_back(s_prediction_weight_leaf);
    all_inputs.push_back(prediction_input_norm_weight_leaf);
    all_inputs.push_back(prediction_input_norm_bias_leaf);
    all_inputs.push_back(read_template_val_leaf);
    for (const auto& tensor : read_projection_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : read_gates_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : write_source_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : write_target_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : write_core_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : write_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : propagation_source_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : propagation_target_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : propagation_core_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : propagation_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : val_norm_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : val_norm_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_transition_source_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_transition_target_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_transition_core_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_transition_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_norm_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_norm_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_ffn_norm_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_ffn_norm_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_ffn_in_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_ffn_in_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_ffn_out_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : level_ffn_out_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : skip_source_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : skip_target_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : skip_core_weights_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : skip_biases_leaves) all_inputs.push_back(tensor);
    for (const auto& tensor : skip_gates_leaves) all_inputs.push_back(tensor);

    torch::autograd::variable_list grad_inputs;
    std::vector<int64_t> grad_input_map(all_inputs.size(), -1);
    for (size_t index = 0; index < all_inputs.size(); ++index) {
      if (all_inputs[index].defined() && all_inputs[index].requires_grad()) {
        grad_input_map[index] = static_cast<int64_t>(grad_inputs.size());
        grad_inputs.push_back(all_inputs[index]);
      }
    }
    auto grads = torch::autograd::grad(
        outputs,
        grad_inputs,
        local_grad_outputs,
        std::nullopt,
        false,
        true);

    std::vector<torch::Tensor> all_grads;
    all_grads.reserve(all_inputs.size());
    for (size_t index = 0; index < all_inputs.size(); ++index) {
      if (grad_input_map[index] < 0) {
        all_grads.push_back(torch::Tensor());
      } else {
        all_grads.push_back(grads[static_cast<size_t>(grad_input_map[index])]);
      }
    }
    if (aligned_s_grad.defined() && all_grads[0].defined()) {
      aligned_s_grad.slice(1, time_index, time_index + 1).add_(all_grads[0]);
    }
    std::vector<torch::Tensor> next_carry;
    next_carry.reserve(current_memory_leaves.size());
    for (size_t memory_index = 0; memory_index < current_memory_leaves.size(); ++memory_index) {
      auto& grad = all_grads[1 + memory_index];
      next_carry.push_back(grad.defined() ? grad : torch::zeros_like(current_memory_leaves[memory_index]));
    }
    carry_memory = next_carry;

    size_t shared_offset = 1 + current_memory_leaves.size();
    if (shared_grad_accum.empty()) {
      shared_grad_accum.resize(all_inputs.size() - shared_offset);
    }
    for (size_t shared_index = 0; shared_index < shared_grad_accum.size(); ++shared_index) {
      accumulate(shared_grad_accum[shared_index], all_grads[shared_offset + shared_index]);
    }
  }

  std::vector<torch::Tensor> result;
  result.reserve(1 + flat_memory.size() + shared_grad_accum.size());
  result.push_back(aligned_s_grad);
  for (const auto& grad : carry_memory) result.push_back(grad);
  for (const auto& grad : shared_grad_accum) result.push_back(grad);
  return result;
}
