#include "jakal_net_native_cuda.h"

#include <cmath>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

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

namespace {

constexpr int kPairwiseTopkForwardThreads = 128;
constexpr int kPairwiseTopkForwardMaxK = 32;
constexpr float kNegativeInfinity = -INFINITY;

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
    int64_t k) {
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
    insert_descending_topk(score, dst, local_scores, local_indices, k);
  }

  extern __shared__ unsigned char shared_storage[];
  auto* shared_scores = reinterpret_cast<float*>(shared_storage);
  auto* shared_indices = reinterpret_cast<int64_t*>(shared_scores + blockDim.x * k);
  auto* shared_routes = reinterpret_cast<float*>(shared_indices + blockDim.x * k);
  auto* selected_indices = reinterpret_cast<int32_t*>(shared_routes + k);

  const int64_t thread_base = static_cast<int64_t>(tid) * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    shared_scores[thread_base + rank] = local_scores[rank];
    shared_indices[thread_base + rank] = local_indices[rank];
  }
  __syncthreads();

  if (tid == 0) {
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
            shared_indices[base + rank],
            merged_scores,
            merged_indices,
            k);
      }
    }

    float max_score = merged_scores[0];
    float denom = 0.0f;
    for (int64_t rank = 0; rank < k; ++rank) {
      const float route = expf(merged_scores[rank] - max_score);
      shared_routes[rank] = route;
      denom += route;
    }
    denom = denom > 0.0f ? denom : 1.0f;
    const int64_t output_base = (batch * src_nodes + src) * k;
    for (int64_t rank = 0; rank < k; ++rank) {
      shared_routes[rank] /= denom;
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
    float score_bias) {
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
  auto* shared_indices = reinterpret_cast<int64_t*>(shared_scores + blockDim.x * k);
  auto* shared_edges = reinterpret_cast<float*>(shared_indices + blockDim.x * k);
  auto* selected_indices = reinterpret_cast<int32_t*>(shared_edges + k);

  const int64_t thread_base = static_cast<int64_t>(tid) * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    shared_scores[thread_base + rank] = local_scores[rank];
    shared_indices[thread_base + rank] = local_indices[rank];
  }
  __syncthreads();

  if (tid == 0) {
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
            shared_indices[base + rank],
            merged_scores,
            merged_indices,
            k);
      }
    }

    float state_acc = 0.0f;
    const int64_t state_base = batch * source_nodes;
    for (int64_t rank = 0; rank < k; ++rank) {
      const float edge = softsignf_device(merged_scores[rank] + score_bias);
      const int64_t source = merged_indices[rank];
      shared_edges[rank] = edge;
      selected_indices[rank] = static_cast<int32_t>(source);
      state_acc += edge * projected_state[state_base + source];
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
    int64_t topk) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(weighted_projected_state, "weighted_projected_state");
  require_cuda_contiguous(weighted_projected_val, "weighted_projected_val");
  if (topk <= 0 || topk > kPairwiseTopkForwardMaxK) {
    throw std::runtime_error("topk must be in the supported fused range [1, 32].");
  }
  if (weighted_projected_source.dim() != 3 || projected_target.dim() != 3) {
    throw std::runtime_error(
        "weighted_projected_source and projected_target must be shaped [batch, nodes, rank].");
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
  if (weighted_projected_source.size(1) != weighted_projected_state.size(1) ||
      weighted_projected_source.size(1) != weighted_projected_val.size(1)) {
    throw std::runtime_error("weighted projected tensors must share src_nodes.");
  }
  if (weighted_projected_source.size(2) != projected_target.size(2)) {
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
  const auto src_nodes = weighted_projected_source.size(1);
  const auto dst_nodes = projected_target.size(1);
  const auto rank_dim = weighted_projected_source.size(2);
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
      threads * k * (sizeof(float) + sizeof(int64_t)) + k * (sizeof(float) + sizeof(int32_t));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_pairwise_topk_forward_cuda",
      [&] {
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
                k);
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
    double score_bias) {
  require_cuda_contiguous(weighted_projected_source, "weighted_projected_source");
  require_cuda_contiguous(projected_target, "projected_target");
  require_cuda_contiguous(projected_state, "projected_state");
  require_cuda_contiguous(projected_val, "projected_val");
  if (topk <= 0 || topk > kPairwiseTopkForwardMaxK) {
    throw std::runtime_error("topk must be in the supported fused range [1, 32].");
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
  const auto k = std::min<int64_t>(topk, source_nodes);
  if (k <= 0) {
    throw std::runtime_error("source_nodes must be positive.");
  }

  auto delta_state = torch::empty({batch_flat, target_nodes}, projected_state.options());
  auto delta_val = torch::empty({batch_flat, target_nodes, out_dim}, projected_val.options());

  const auto blocks = batch_flat * target_nodes;
  constexpr int threads = kPairwiseTopkForwardThreads;
  const auto shmem =
      threads * k * (sizeof(float) + sizeof(int64_t)) + k * (sizeof(float) + sizeof(int32_t));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weighted_projected_source.scalar_type(),
      "low_rank_propagation_topk_forward_cuda",
      [&] {
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
                static_cast<float>(score_bias));
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
