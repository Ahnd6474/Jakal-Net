#include "jakal_net_native_cuda.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/autograd/autograd.h>

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

bool jakal_net_low_rank_propagation_window_entmax15_forward_cuda_available() {
  return true;
}

namespace {

constexpr int kPairwiseTopkForwardThreads = 128;
constexpr int kPairwiseTopkForwardMaxK = 32;
constexpr float kNegativeInfinity = -INFINITY;
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
    const torch::Tensor& projected_target,
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
    throw std::runtime_error("topk must be in the supported fused range [1, 32].");
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
    throw std::runtime_error("Route projection tensors must share dtype.");
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
    const torch::Tensor& projected_target,
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
    throw std::runtime_error("topk must be in the supported fused range [1, 32].");
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
    throw std::runtime_error("Route projection tensors must share dtype.");
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
jakal_net_low_rank_propagation_window_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
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
    throw std::runtime_error("Route projection tensors must share dtype.");
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
jakal_net_low_rank_propagation_window_entmax15_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
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
    throw std::runtime_error("Route projection tensors must share dtype.");
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
  auto output = torch::matmul(input, weight.transpose(0, 1));
  if (bias.defined() && bias.numel() != 0) {
    output = output + bias.to(output.scalar_type()).view({1, 1, -1});
  }
  return output;
}

torch::Tensor scan_cuda_linear2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  auto output = torch::matmul(input, weight.transpose(0, 1));
  if (bias.defined() && bias.numel() != 0) {
    output = output + bias.to(output.scalar_type());
  }
  return output;
}

torch::Tensor scan_cuda_layer_norm_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
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
    const torch::Tensor& val_norm_bias) {
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
  const bool multihead =
      source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2;
  const bool use_fused_topk =
      compress_name == "softmax" ||
      compress_name == "signed_abs_softmax" ||
      compress_name == "signed_entmax15";
  if (use_fused_topk &&
      allow_fastpath &&
      topk > 0 && topk <= 32) {
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
  if (multihead) {
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
  torch::Tensor selected_scores;
  torch::Tensor selected_indices;
  if (k == dst_nodes) {
    selected_scores = logits;
    selected_indices = scan_cuda_full_topk_indices(logits);
  } else {
    auto topk_result = logits.topk(k, -1, true, true);
    selected_scores = std::get<0>(topk_result);
    selected_indices = std::get<1>(topk_result);
  }
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
  const bool multihead =
      source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2;
  const bool use_fused_topk =
      compress_name == "softsign" ||
      compress_name == "signed_abs_softmax";
  if (use_fused_topk &&
      allow_fastpath &&
      topk > 0 && topk <= 32) {
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
    auto weighted_projected_state = (layer_state.to(torch::kFloat32) * layer_state.to(torch::kFloat32)).contiguous();
    auto weighted_projected_val =
        (layer_state.to(torch::kFloat32).unsqueeze(-1) * layer_val.to(torch::kFloat32)).contiguous();
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

  torch::Tensor scores;
  if (multihead) {
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
  const auto k = std::min<int64_t>(std::max<int64_t>(1, topk), nodes);
  torch::Tensor selected_scores;
  torch::Tensor selected_indices;
  if (k == nodes) {
    selected_scores = scores;
    selected_indices = scan_cuda_full_topk_indices(scores);
  } else {
    auto topk_result = scores.topk(k, -1, true, true);
    selected_scores = std::get<0>(topk_result);
    selected_indices = std::get<1>(topk_result);
  }
  auto expanded_state = layer_state.unsqueeze(1).expand({layer_state.size(0), nodes, layer_state.size(1)});
  auto selected_state = torch::gather(expanded_state, 2, selected_indices);
  auto expanded_val = layer_val.unsqueeze(1).expand({layer_val.size(0), nodes, layer_val.size(1), layer_val.size(2)});
  auto selected_val = torch::gather(
      expanded_val,
      2,
      selected_indices.unsqueeze(-1).expand({selected_indices.size(0), selected_indices.size(1), selected_indices.size(2), layer_val.size(2)}));
  auto edges = scan_cuda_compress_scores(compress_name, selected_scores);
  auto weighted_edges = edges * selected_state;
  auto delta_state =
      (weighted_edges.to(torch::kFloat32) * selected_state.to(torch::kFloat32)).sum(-1);
  auto delta_val =
      (weighted_edges.to(torch::kFloat32).unsqueeze(-1) * selected_val.to(torch::kFloat32)).sum(-2);
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
  std::vector<torch::Tensor> read_terms;
  read_terms.reserve(memory_state.size());
  for (size_t index = 0; index < memory_state.size(); ++index) {
    auto read_layer = scan_cuda_layer_with_val_norm(memory_state[index], val_norm_weights[index], val_norm_biases[index]);
    auto sender_strength = torch::softplus(read_layer.state).unsqueeze(-1);
    auto read_summary = (sender_strength * read_layer.val).sum(1);
    read_summary = read_summary + read_template_val.to(read_summary.scalar_type()).unsqueeze(0);
    auto projected = scan_cuda_linear2d(read_summary, read_projection_weights[index], torch::Tensor());
    auto gate = torch::sigmoid(read_gates[index].to(read_summary.scalar_type()));
    read_terms.push_back(gate * projected);
  }
  return torch::stack(read_terms, 0).sum(0);
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
  auto projected_s = scan_cuda_linear3d(aligned_s, s_prediction_weight, torch::Tensor());
  std::vector<torch::Tensor> query_steps;
  query_steps.reserve(aligned_s.size(1));

  for (int64_t time_index = 0; time_index < aligned_s.size(1); ++time_index) {
    if (collect_trace) {
      for (size_t trace_index = 0; trace_index < num_levels; ++trace_index) {
        trace_states[trace_index].push_back(current_memory[trace_index].state);
        trace_vals[trace_index].push_back(current_memory[trace_index].val);
      }
    }
    auto token_val = aligned_s.slice(1, time_index, time_index + 1);
    auto token_state = scan_cuda_linear3d(token_val, value_to_state_weight, value_to_state_bias).squeeze(-1);
    ScanCudaLayerState token_layer{token_state, token_val};

    std::vector<ScanCudaLayerState> next_memory;
    next_memory.reserve(num_levels);

    auto first_level_normed = scan_cuda_layer_with_val_norm(current_memory[0], val_norm_weights[0], val_norm_biases[0]);
    auto first_write_delta = scan_cuda_low_rank_transition_pairwise_topk(
        torch::softplus(token_layer.state),
        token_layer.state,
        token_layer.val,
        token_layer.val,
        first_level_normed.val,
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
        val_norm_biases[0]);
    auto level_for_propagation = scan_cuda_layer_with_val_norm(level, val_norm_weights[0], val_norm_biases[0]);
    auto first_prop_delta = scan_cuda_low_rank_propagation_topk(
        level_for_propagation.state,
        level_for_propagation.val,
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
        val_norm_biases[0]);
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < num_levels; ++level_index) {
      auto current_level = current_memory[level_index];
      auto normalized_level = scan_cuda_layer_with_val_norm(
          current_level,
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      auto normalized_parent = scan_cuda_layer_with_val_norm(
          next_memory[level_index - 1],
          level_norm_weights[level_index - 1],
          level_norm_biases[level_index - 1]);
      auto parent_delta = scan_cuda_low_rank_transition_pairwise_topk(
          torch::softplus(normalized_parent.state),
          normalized_parent.state,
          normalized_parent.val,
          normalized_parent.val,
          normalized_level.val,
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
          val_norm_biases[level_index]);

      if (level_index == 1 && expected_skip_count > 0) {
        auto skip_gate = torch::sigmoid(skip_gates[0].to(token_val.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(token_layer.state),
            token_layer.state,
            token_layer.val,
            token_layer.val,
            normalized_level.val,
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
            val_norm_biases[level_index]);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto normalized_skip_source = scan_cuda_layer_with_val_norm(
            next_memory[level_index - 2],
            level_norm_weights[level_index - 2],
            level_norm_biases[level_index - 2]);
        auto skip_gate = torch::sigmoid(skip_gates[skip_index].to(normalized_skip_source.val.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(normalized_skip_source.state),
            normalized_skip_source.state,
            normalized_skip_source.val,
            normalized_skip_source.val,
            normalized_level.val,
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
            val_norm_biases[level_index]);
      }

      auto updated_level_for_prop = scan_cuda_layer_with_val_norm(
          updated_level,
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      auto propagation_delta = scan_cuda_low_rank_propagation_topk(
          updated_level_for_prop.state,
          updated_level_for_prop.val,
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
          val_norm_biases[level_index]);
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
    query_steps.push_back(
        scan_cuda_layer_norm_last_dim(
            query_input,
            prediction_input_norm_weight,
            prediction_input_norm_bias)
            .unsqueeze(1));
  }

  torch::Tensor query_val;
  if (query_steps.empty()) {
    query_val = aligned_s.new_empty({aligned_s.size(0), 0, aligned_s.size(2)});
  } else {
    query_val = torch::cat(query_steps, 1);
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

    auto token_state = scan_cuda_linear3d(
        token_val_leaf,
        value_to_state_weight_leaf,
        value_to_state_bias_leaf).squeeze(-1);
    auto projected_s_t = scan_cuda_linear3d(token_val_leaf, s_prediction_weight_leaf, torch::Tensor()).squeeze(1);

    std::vector<ScanCudaLayerState> next_memory;
    next_memory.reserve(current_memory.size());
    auto first_level_normed = scan_cuda_layer_with_val_norm(current_memory[0], val_norm_weights_leaves[0], val_norm_biases_leaves[0]);
    auto first_write_delta = scan_cuda_low_rank_transition_pairwise_topk(
        torch::softplus(token_state),
        token_state,
        token_val_leaf,
        token_val_leaf,
        first_level_normed.val,
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
        val_norm_biases_leaves[0]);
    auto level_for_propagation = scan_cuda_layer_with_val_norm(level, val_norm_weights_leaves[0], val_norm_biases_leaves[0]);
    auto first_prop_delta = scan_cuda_low_rank_propagation_topk(
        level_for_propagation.state,
        level_for_propagation.val,
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
        val_norm_biases_leaves[0]);
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < current_memory.size(); ++level_index) {
      auto current_level = current_memory[level_index];
      auto normalized_level = scan_cuda_layer_with_val_norm(current_level, val_norm_weights_leaves[level_index], val_norm_biases_leaves[level_index]);
      auto normalized_parent = scan_cuda_layer_with_val_norm(next_memory[level_index - 1], level_norm_weights_leaves[level_index - 1], level_norm_biases_leaves[level_index - 1]);
      auto parent_delta = scan_cuda_low_rank_transition_pairwise_topk(
          torch::softplus(normalized_parent.state),
          normalized_parent.state,
          normalized_parent.val,
          normalized_parent.val,
          normalized_level.val,
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
          val_norm_biases_leaves[level_index]);

      if (level_index == 1 && !skip_source_weights_leaves.empty()) {
        auto skip_gate = torch::sigmoid(skip_gates_leaves[0].to(token_val_leaf.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(token_state),
            token_state,
            token_val_leaf,
            token_val_leaf,
            normalized_level.val,
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
            val_norm_biases_leaves[level_index]);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto normalized_skip_source = scan_cuda_layer_with_val_norm(next_memory[level_index - 2], level_norm_weights_leaves[level_index - 2], level_norm_biases_leaves[level_index - 2]);
        auto skip_gate = torch::sigmoid(skip_gates_leaves[skip_index].to(normalized_skip_source.val.scalar_type()));
        auto skip_delta = scan_cuda_low_rank_transition_pairwise_topk(
            torch::softplus(normalized_skip_source.state),
            normalized_skip_source.state,
            normalized_skip_source.val,
            normalized_skip_source.val,
            normalized_level.val,
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
            val_norm_biases_leaves[level_index]);
      }

      auto updated_level_for_prop = scan_cuda_layer_with_val_norm(updated_level, val_norm_weights_leaves[level_index], val_norm_biases_leaves[level_index]);
      auto propagation_delta = scan_cuda_low_rank_propagation_topk(
          updated_level_for_prop.state,
          updated_level_for_prop.val,
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
          val_norm_biases_leaves[level_index]);
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
