#include "jakal_net_native_cuda.h"

#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

bool jakal_net_compiled_with_cuda_source() {
  return true;
}

bool jakal_net_query_topk_reduce_cuda_available() {
  return true;
}

namespace {

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
__global__ void query_topk_reduce_backward_state_kernel(
    const scalar_t* __restrict__ edges,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ projected_state,
    const scalar_t* __restrict__ grad_delta_state,
    scalar_t* __restrict__ grad_edges,
    scalar_t* __restrict__ grad_projected_state,
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
  const scalar_t grad_state = grad_delta_state[query_linear];
  const scalar_t edge = edges[linear];
  const scalar_t selected_state = projected_state[batch * source_nodes + source_index];

  grad_edges[linear] = grad_state * selected_state;
  atomicAdd(&grad_projected_state[batch * source_nodes + source_index], grad_state * edge);
}

template <typename scalar_t>
__global__ void query_topk_reduce_backward_val_kernel(
    const scalar_t* __restrict__ edges,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ projected_val,
    const scalar_t* __restrict__ grad_delta_val,
    scalar_t* __restrict__ grad_edges,
    scalar_t* __restrict__ grad_projected_val,
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
  const scalar_t grad_val = grad_delta_val[query_linear * out_dim + out];
  const scalar_t edge = edges[rank_linear];
  const int64_t source_offset = batch * source_nodes * out_dim + source_index * out_dim + out;

  atomicAdd(&grad_edges[rank_linear], grad_val * projected_val[source_offset]);
  atomicAdd(&grad_projected_val[source_offset], grad_val * edge);
}

template <typename scalar_t>
__global__ void softsign_backward_kernel(
    const scalar_t* __restrict__ scores,
    const scalar_t* __restrict__ grad_edges,
    scalar_t* __restrict__ grad_scores,
    int64_t total) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const scalar_t score = scores[linear];
  const scalar_t denom = static_cast<scalar_t>(1) + (score < 0 ? -score : score);
  grad_scores[linear] = grad_edges[linear] / (denom * denom);
}

template <typename scalar_t>
__global__ void softmax_backward_kernel(
    const scalar_t* __restrict__ routes,
    const scalar_t* __restrict__ grad_routes,
    scalar_t* __restrict__ grad_scores,
    int64_t rows,
    int64_t k) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  scalar_t dot = static_cast<scalar_t>(0);
  const int64_t base = row * k;
  for (int64_t rank = 0; rank < k; ++rank) {
    dot += routes[base + rank] * grad_routes[base + rank];
  }
  for (int64_t rank = 0; rank < k; ++rank) {
    const int64_t offset = base + rank;
    grad_scores[offset] = routes[offset] * (grad_routes[offset] - dot);
  }
}

template <typename scalar_t>
__global__ void diagonal_pairwise_topk_backward_kernel(
    const scalar_t* __restrict__ query_val,
    const scalar_t* __restrict__ source_val,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ grad_scores,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_source,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    scalar_t inv_temperature,
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
  const scalar_t g = grad_scores[rank_linear] * inv_temperature;
  const scalar_t q = query_val[(batch * query_nodes + query) * dim + d];
  const scalar_t s = source_val[(batch * source_nodes + source_index) * dim + d];
  const scalar_t w = weight[d];

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
    const scalar_t* __restrict__ grad_scores,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_source,
    scalar_t* __restrict__ grad_source_weight,
    scalar_t* __restrict__ grad_target_weight,
    scalar_t* __restrict__ grad_core_weight,
    scalar_t* __restrict__ grad_bias,
    scalar_t inv_temperature,
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
  const scalar_t g = grad_scores[topk_linear] * inv_temperature;
  const scalar_t projected_q = projected_query[(batch * query_nodes + query) * rank_dim + r];
  const scalar_t projected_s = projected_source[(batch * source_nodes + source_index) * rank_dim + r];
  const scalar_t core = core_weight[r];
  const scalar_t q = query_val[(batch * query_nodes + query) * dim + d];
  const scalar_t s = source_val[(batch * source_nodes + source_index) * dim + d];

  const scalar_t grad_projected_s = g * core * projected_q;
  const scalar_t grad_projected_q = g * core * projected_s;

  atomicAdd(&grad_source[(batch * source_nodes + source_index) * dim + d],
            grad_projected_s * source_weight[r * dim + d]);
  atomicAdd(&grad_query[(batch * query_nodes + query) * dim + d],
            grad_projected_q * target_weight[r * dim + d]);
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
  auto grad_edges = torch::empty_like(edges);
  auto grad_projected_state = torch::zeros_like(projected_state);
  auto grad_projected_val = torch::zeros_like(projected_val);

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t state_total = batch_flat * query_nodes * k;
  const int64_t val_total = batch_flat * query_nodes * k * out_dim;

  AT_DISPATCH_FLOATING_TYPES(edges.scalar_type(), "query_topk_reduce_backward_cuda", [&] {
    query_topk_reduce_backward_state_kernel<scalar_t>
        <<<(state_total + threads - 1) / threads, threads, 0, stream>>>(
            edges.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            projected_state.data_ptr<scalar_t>(),
            grad_delta_state.data_ptr<scalar_t>(),
            grad_edges.data_ptr<scalar_t>(),
            grad_projected_state.data_ptr<scalar_t>(),
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
            grad_edges.data_ptr<scalar_t>(),
            grad_projected_val.data_ptr<scalar_t>(),
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
  auto grad_scores = torch::empty_like(scores);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto total = scores.numel();
  AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "softsign_backward_cuda", [&] {
    softsign_backward_kernel<scalar_t>
        <<<(total + threads - 1) / threads, threads, 0, stream>>>(
            scores.data_ptr<scalar_t>(),
            grad_edges.data_ptr<scalar_t>(),
            grad_scores.data_ptr<scalar_t>(),
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
  auto grad_scores = torch::empty_like(routes);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto rows = routes.size(0) * routes.size(1);
  const auto k = routes.size(2);
  AT_DISPATCH_FLOATING_TYPES(routes.scalar_type(), "softmax_backward_cuda", [&] {
    softmax_backward_kernel<scalar_t>
        <<<(rows + threads - 1) / threads, threads, 0, stream>>>(
            routes.data_ptr<scalar_t>(),
            grad_routes.data_ptr<scalar_t>(),
            grad_scores.data_ptr<scalar_t>(),
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
  auto grad_query = torch::zeros_like(query_val);
  auto grad_source = torch::zeros_like(source_val);
  auto grad_weight = torch::zeros_like(weight);
  auto grad_bias = torch::zeros({}, weight.options());

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = batch_flat * query_nodes * k * dim;
  AT_DISPATCH_FLOATING_TYPES(query_val.scalar_type(), "diagonal_pairwise_topk_backward_cuda", [&] {
    diagonal_pairwise_topk_backward_kernel<scalar_t>
        <<<(total + threads - 1) / threads, threads, 0, stream>>>(
            query_val.data_ptr<scalar_t>(),
            source_val.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            grad_scores.data_ptr<scalar_t>(),
            grad_query.data_ptr<scalar_t>(),
            grad_source.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            static_cast<scalar_t>(1.0 / temperature),
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
  auto grad_query = torch::zeros_like(query_val);
  auto grad_source = torch::zeros_like(source_val);
  auto grad_source_weight = torch::zeros_like(source_weight);
  auto grad_target_weight = torch::zeros_like(target_weight);
  auto grad_core_weight = torch::zeros_like(core_weight);
  auto grad_bias = torch::zeros({}, core_weight.options());

  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = batch_flat * query_nodes * k * rank_dim * dim;
  AT_DISPATCH_FLOATING_TYPES(query_val.scalar_type(), "low_rank_pairwise_topk_backward_cuda", [&] {
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
            grad_scores.data_ptr<scalar_t>(),
            grad_query.data_ptr<scalar_t>(),
            grad_source.data_ptr<scalar_t>(),
            grad_source_weight.data_ptr<scalar_t>(),
            grad_target_weight.data_ptr<scalar_t>(),
            grad_core_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            static_cast<scalar_t>(1.0 / temperature),
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
