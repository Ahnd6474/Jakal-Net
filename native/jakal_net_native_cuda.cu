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
