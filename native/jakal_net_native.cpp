#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#if __has_include(<torch/cuda.h>)
#include <torch/cuda.h>
#define JAKAL_NET_HAS_TORCH_CUDA_HEADER 1
#else
#define JAKAL_NET_HAS_TORCH_CUDA_HEADER 0
#endif

#include "jakal_net_native_cuda.h"

#ifndef WITH_CUDA
bool jakal_net_compiled_with_cuda_source() {
  return false;
}
#endif

namespace {

int64_t product(const c10::IntArrayRef sizes, int64_t start, int64_t end) {
  int64_t value = 1;
  for (int64_t index = start; index < end; ++index) {
    value *= sizes[index];
  }
  return value;
}

std::vector<int64_t> batch_shape(const torch::Tensor& tensor, int64_t trailing_dims) {
  const auto dims = tensor.dim();
  std::vector<int64_t> result;
  result.reserve(std::max<int64_t>(0, dims - trailing_dims));
  for (int64_t index = 0; index < dims - trailing_dims; ++index) {
    result.push_back(tensor.size(index));
  }
  return result;
}

torch::Tensor flatten_state(const torch::Tensor& state) {
  const auto flat_batch = product(state.sizes(), 0, state.dim() - 1);
  return state.reshape({flat_batch, state.size(-1)});
}

torch::Tensor flatten_val(const torch::Tensor& val) {
  const auto flat_batch = product(val.sizes(), 0, val.dim() - 2);
  return val.reshape({flat_batch, val.size(-2), val.size(-1)});
}

torch::Tensor reshape_state(
    const torch::Tensor& flat_state,
    const std::vector<int64_t>& batch_sizes,
    int64_t nodes) {
  std::vector<int64_t> shape = batch_sizes;
  shape.push_back(nodes);
  return flat_state.reshape(shape);
}

torch::Tensor reshape_val(
    const torch::Tensor& flat_val,
    const std::vector<int64_t>& batch_sizes,
    int64_t nodes,
    int64_t dim) {
  std::vector<int64_t> shape = batch_sizes;
  shape.push_back(nodes);
  shape.push_back(dim);
  return flat_val.reshape(shape);
}

bool supports_cuda_runtime() {
#if JAKAL_NET_HAS_TORCH_CUDA_HEADER
  return torch::cuda::is_available();
#else
  return false;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> query_topk_reduce_cuda_wrapper(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val) {
#ifdef WITH_CUDA
  return jakal_net_query_topk_reduce_cuda(edges, indices, projected_state, projected_val);
#else
  throw std::runtime_error("query_topk_reduce_cuda requires a CUDA-enabled build.");
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_topk_reduce_backward_cuda_wrapper(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val) {
#ifdef WITH_CUDA
  return jakal_net_query_topk_reduce_backward_cuda(
      edges, indices, projected_state, projected_val, grad_delta_state, grad_delta_val);
#else
  throw std::runtime_error("query_topk_reduce_backward_cuda requires a CUDA-enabled build.");
#endif
}

torch::Tensor softsign_backward_cuda_wrapper(
    const torch::Tensor& scores,
    const torch::Tensor& grad_edges) {
#ifdef WITH_CUDA
  return jakal_net_softsign_backward_cuda(scores, grad_edges);
#else
  throw std::runtime_error("softsign_backward_cuda requires a CUDA-enabled build.");
#endif
}

torch::Tensor softmax_backward_cuda_wrapper(
    const torch::Tensor& routes,
    const torch::Tensor& grad_routes) {
#ifdef WITH_CUDA
  return jakal_net_softmax_backward_cuda(routes, grad_routes);
#else
  throw std::runtime_error("softmax_backward_cuda requires a CUDA-enabled build.");
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
diagonal_pairwise_topk_backward_cuda_wrapper(
    const torch::Tensor& query_val,
    const torch::Tensor& source_val,
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    const torch::Tensor& grad_scores,
    double temperature) {
#ifdef WITH_CUDA
  return jakal_net_diagonal_pairwise_topk_backward_cuda(
      query_val, source_val, weight, indices, grad_scores, temperature);
#else
  throw std::runtime_error(
      "diagonal_pairwise_topk_backward_cuda requires a CUDA-enabled build.");
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
low_rank_pairwise_topk_backward_cuda_wrapper(
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
#ifdef WITH_CUDA
  return jakal_net_low_rank_pairwise_topk_backward_cuda(
      query_val,
      source_val,
      source_weight,
      target_weight,
      core_weight,
      projected_query,
      projected_source,
      indices,
      grad_scores,
      temperature);
#else
  throw std::runtime_error(
      "low_rank_pairwise_topk_backward_cuda requires a CUDA-enabled build.");
#endif
}

void require_supported_device(const torch::Tensor& tensor, const std::string& name) {
  if (!tensor.device().is_cpu() && !tensor.device().is_cuda()) {
    throw std::runtime_error(
        name + " must be a CPU or CUDA tensor in the native backend.");
  }
}

void require_same_batch_shape(
    const torch::Tensor& left,
    int64_t left_trailing_dims,
    const std::string& left_name,
    const torch::Tensor& right,
    int64_t right_trailing_dims,
    const std::string& right_name) {
  auto left_batch = batch_shape(left, left_trailing_dims);
  auto right_batch = batch_shape(right, right_trailing_dims);
  if (left_batch != right_batch) {
    throw std::runtime_error(
        left_name + " and " + right_name + " must share batch dimensions.");
  }
}

void require_query_source_shapes(
    const torch::Tensor& query_val,
    const torch::Tensor& source_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val) {
  require_same_batch_shape(query_val, 2, "query_val", source_val, 2, "source_val");
  require_same_batch_shape(source_val, 2, "source_val", projected_state, 1, "projected_state");
  require_same_batch_shape(source_val, 2, "source_val", projected_val, 2, "projected_val");
  if (projected_state.size(-1) != source_val.size(-2)) {
    throw std::runtime_error("projected_state must end with source_nodes.");
  }
  if (projected_val.size(-2) != source_val.size(-2)) {
    throw std::runtime_error("projected_val must end with [source_nodes, out_dim].");
  }
}

void require_known_edge_compress(const std::string& edge_compress_name) {
  if (edge_compress_name != "softsign") {
    throw std::runtime_error(
        "Only edge_compress_name='softsign' is supported by the CPU native backend.");
  }
}

torch::ScalarType accumulator_dtype(torch::ScalarType input_dtype) {
  if (input_dtype == torch::kFloat16 || input_dtype == torch::kBFloat16) {
    return torch::kFloat32;
  }
  return input_dtype;
}

torch::Tensor allocate_accumulator(
    const std::vector<int64_t>& shape,
    const torch::Tensor& reference,
    torch::ScalarType dtype) {
  return torch::zeros(shape, reference.options().dtype(dtype));
}

torch::Tensor softsign(const torch::Tensor& tensor) {
  return tensor / (1 + tensor.abs());
}

torch::Tensor linear3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  auto output = torch::matmul(input, weight.transpose(0, 1));
  if (bias.has_value()) {
    output = output + bias.value().view({1, 1, -1});
  }
  return output;
}

torch::Tensor linear4d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  auto output = torch::matmul(input, weight.transpose(0, 1));
  if (bias.has_value()) {
    output = output + bias.value().view({1, 1, 1, -1});
  }
  return output;
}

c10::optional<torch::Tensor> slice_optional_tensor(
    const c10::optional<torch::Tensor>& tensor,
    int64_t start,
    int64_t end) {
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return tensor.value().slice(0, start, end).contiguous();
}

torch::Tensor pairwise_scores(
    const std::string& pairwise_kind,
    const torch::Tensor& target_val,
    const torch::Tensor& source_val,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias) {
  if (pairwise_kind == "diagonal_bilinear") {
    auto target_proj = target_val * weight.view({1, 1, -1});
    auto scores = torch::bmm(target_proj, source_val.transpose(1, 2));
    if (bias.has_value()) {
      scores = scores + bias.value();
    }
    return scores;
  }

  if (pairwise_kind == "bilinear") {
    auto target_proj = torch::matmul(target_val, weight);
    auto scores = torch::bmm(target_proj, source_val.transpose(1, 2));
    if (bias.has_value()) {
      scores = scores + bias.value();
    }
    return scores;
  }

  if (pairwise_kind == "hadamard_mlp") {
    if (!in_weight.has_value() || !out_weight.has_value()) {
      throw std::runtime_error("Hadamard MLP pairwise kernel is missing MLP weights.");
    }
    auto hidden = torch::einsum(
        "bid,hd,bjd->bijh",
        {target_val, in_weight.value(), source_val});
    if (in_bias.has_value()) {
      hidden = hidden + in_bias.value().view({1, 1, 1, -1});
    }
    hidden = hidden * torch::sigmoid(hidden);
    auto scores = torch::matmul(hidden, out_weight.value().transpose(0, 1)).squeeze(-1);
    if (out_bias.has_value()) {
      scores = scores + out_bias.value();
    } else if (bias.has_value()) {
      scores = scores + bias.value();
    }
    return scores;
  }

  throw std::runtime_error("Unsupported pairwise kernel kind: " + pairwise_kind);
}

torch::Tensor prepare_route_context(
    const std::string& route_kind,
    const torch::Tensor& src_val,
    const torch::Tensor& in_weight,
    const c10::optional<torch::Tensor>& in_bias) {
  if (route_kind == "linear") {
    return src_val;
  }
  if (route_kind == "mlp") {
    auto hidden = linear3d(src_val, in_weight, in_bias);
    return hidden * torch::sigmoid(hidden);
  }
  throw std::runtime_error("Unsupported route kernel kind: " + route_kind);
}

torch::Tensor route_block_logits(
    const std::string& route_kind,
    const torch::Tensor& route_context,
    const torch::Tensor& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    int64_t start,
    int64_t end) {
  if (route_kind == "linear") {
    auto weight_slice = in_weight.slice(0, start, end).contiguous();
    auto bias_slice = slice_optional_tensor(in_bias, start, end);
    return linear3d(route_context, weight_slice, bias_slice);
  }
  if (route_kind == "mlp") {
    if (!out_weight.has_value()) {
      throw std::runtime_error("MLP route kernel is missing output weight.");
    }
    auto weight_slice = out_weight.value().slice(0, start, end).contiguous();
    auto bias_slice = slice_optional_tensor(out_bias, start, end);
    return linear3d(route_context, weight_slice, bias_slice);
  }
  throw std::runtime_error("Unsupported route kernel kind: " + route_kind);
}

torch::Tensor pairwise_route_block_logits(
    const std::string& route_kind,
    const torch::Tensor& src_val,
    const torch::Tensor& dst_val,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& source_bias,
    const c10::optional<torch::Tensor>& target_weight,
    const c10::optional<torch::Tensor>& target_bias,
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& hidden_weight,
    const c10::optional<torch::Tensor>& hidden_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    double temperature) {
  if (temperature <= 0.0) {
    throw std::runtime_error("route temperature must be positive.");
  }

  torch::Tensor scores;
  if (route_kind == "diagonal_bilinear_route") {
    auto weighted_source = src_val * core_weight.view({1, 1, -1});
    scores = torch::bmm(weighted_source, dst_val.transpose(1, 2));
  } else if (route_kind == "low_rank_bilinear_route") {
    if (!source_weight.has_value() || !target_weight.has_value()) {
      throw std::runtime_error("Low-rank route kernel is missing projection weights.");
    }
    auto projected_source =
        torch::matmul(src_val, source_weight.value().transpose(0, 1)) *
        core_weight.view({1, 1, -1});
    auto projected_target = torch::matmul(dst_val, target_weight.value().transpose(0, 1));
    scores = torch::bmm(projected_source, projected_target.transpose(1, 2));
  } else if (route_kind == "full_bilinear_route") {
    if (!source_weight.has_value() || !target_weight.has_value()) {
      throw std::runtime_error("Full bilinear route kernel is missing projection weights.");
    }
    auto projected_source = torch::matmul(src_val, source_weight.value().transpose(0, 1));
    auto projected_target = torch::matmul(dst_val, target_weight.value().transpose(0, 1));
    auto weighted_source = torch::matmul(projected_source, core_weight);
    scores = torch::bmm(weighted_source, projected_target.transpose(1, 2));
  } else if (route_kind == "source_target_hadamard_mlp_route") {
    if (!source_weight.has_value() || !target_weight.has_value() ||
        !hidden_weight.has_value() || !out_weight.has_value()) {
      throw std::runtime_error("Hadamard MLP route kernel is missing projection weights.");
    }
    auto projected_source = linear3d(src_val, source_weight.value(), source_bias);
    auto projected_target = linear3d(dst_val, target_weight.value(), target_bias);
    const auto width = projected_source.size(2);
    auto hidden_slices = hidden_weight.value().split(width, 1);
    if (hidden_slices.size() != 3) {
      throw std::runtime_error(
          "Hadamard MLP route hidden_weight must split into source/target/hadamard slices.");
    }
    auto source_linear = linear3d(projected_source, hidden_slices[0], c10::nullopt);
    auto target_linear = linear3d(projected_target, hidden_slices[1], hidden_bias);
    auto hidden = torch::einsum(
        "bid,hd,bkd->bikh",
        {projected_source, hidden_slices[2], projected_target});
    hidden = hidden + source_linear.unsqueeze(2);
    hidden = hidden + target_linear.unsqueeze(1);
    hidden = hidden * torch::sigmoid(hidden);
    scores = torch::matmul(hidden, out_weight.value().transpose(0, 1)).squeeze(-1);
    if (out_bias.has_value()) {
      scores = scores + out_bias.value();
    }
  } else {
    throw std::runtime_error("Unsupported pairwise route kernel kind: " + route_kind);
  }

  if (bias.has_value()) {
    scores = scores + bias.value();
  }
  if (temperature != 1.0) {
    scores = scores / temperature;
  }
  return scores;
}

torch::Tensor causal_window_mask(
    int64_t target_start,
    int64_t target_end,
    int64_t source_start,
    int64_t source_end,
    int64_t window,
    torch::Device device) {
  auto options = torch::TensorOptions().dtype(torch::kLong).device(device);
  auto target_index = torch::arange(target_start, target_end, options).unsqueeze(-1);
  auto source_index = torch::arange(source_start, source_end, options).unsqueeze(0);
  return source_index.le(target_index) & source_index.ge(target_index - window);
}

std::tuple<torch::Tensor, torch::Tensor> gather_by_indices(
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& indices) {
  const auto batch_flat = projected_state.size(0);
  const auto target_nodes = indices.size(1);
  const auto source_nodes = projected_state.size(1);
  const auto out_dim = projected_val.size(2);

  auto expanded_state =
      projected_state.unsqueeze(1).expand({batch_flat, target_nodes, source_nodes});
  auto gathered_state = expanded_state.gather(2, indices);

  auto expanded_val =
      projected_val.unsqueeze(1).expand({batch_flat, target_nodes, source_nodes, out_dim});
  auto gather_index =
      indices.unsqueeze(-1).expand({batch_flat, target_nodes, indices.size(2), out_dim});
  auto gathered_val = expanded_val.gather(2, gather_index);
  return {gathered_state, gathered_val};
}

std::vector<std::string> supported_ops() {
  auto ops = std::vector<std::string>{
      "propagation_dense",
      "propagation_window",
      "propagation_topk",
      "propagation_query_dense",
      "transition_dense",
      "transition_pairwise_dense",
      "transition_pairwise_topk",
      "transition_topk",
      "transition_query_topk",
      "transition_query_topk_select",
  };
  if (jakal_net_compiled_with_cuda_source()) {
    ops.insert(
        ops.end(),
        {
            "query_topk_reduce_cuda",
            "query_topk_reduce_backward_cuda",
            "softsign_backward_cuda",
            "softmax_backward_cuda",
            "diagonal_pairwise_topk_backward_cuda",
            "low_rank_pairwise_topk_backward_cuda",
        });
  }
  return ops;
}

std::vector<std::string> supported_devices() {
  auto devices = std::vector<std::string>{"cpu"};
  if (supports_cuda_runtime()) {
    devices.push_back("cuda");
  }
  return devices;
}

std::string backend_name() {
  if (supports_cuda_runtime()) {
    if (jakal_net_compiled_with_cuda_source()) {
      return "aten_cpp_cuda";
    }
    return "aten_cpp_dispatch_cuda";
  }
  return "aten_cpp_cpu";
}

bool can_use_full_dense_logits(
    const torch::Tensor& reference,
    int64_t batch_flat,
    int64_t left_nodes,
    int64_t right_nodes) {
  constexpr int64_t kDefaultMaxDenseScoreElements = 64LL * 1024LL * 1024LL;
  int64_t max_dense_score_elements = kDefaultMaxDenseScoreElements;
  if (const char* raw_limit = std::getenv("JAKAL_NET_NATIVE_DENSE_FULL_MAX_ELEMENTS")) {
    try {
      max_dense_score_elements = std::max<int64_t>(0, std::stoll(raw_limit));
    } catch (const std::exception&) {
      max_dense_score_elements = kDefaultMaxDenseScoreElements;
    }
  }
  if (!reference.is_cuda()) {
    return false;
  }
  if (max_dense_score_elements <= 0) {
    return false;
  }
  if (batch_flat <= 0 || left_nodes <= 0 || right_nodes <= 0) {
    return false;
  }
  if (left_nodes > max_dense_score_elements / right_nodes) {
    return false;
  }
  const auto per_batch = left_nodes * right_nodes;
  return batch_flat <= max_dense_score_elements / per_batch;
}

std::tuple<torch::Tensor, torch::Tensor> propagation_dense(
    const std::string& pairwise_kind,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    const std::string& edge_compress_name,
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t target_block_size,
    int64_t source_block_size) {
  require_known_edge_compress(edge_compress_name);
  require_supported_device(layer_val, "layer_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");

  auto batch_sizes = batch_shape(layer_val, 2);
  auto flat_val = flatten_val(layer_val).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_val.size(0);
  const auto num_nodes = flat_val.size(1);
  const auto out_dim = flat_projected_val.size(2);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());

  if (pairwise_kind != "hadamard_mlp" &&
      can_use_full_dense_logits(flat_val, batch_flat, num_nodes, num_nodes)) {
    auto scores = pairwise_scores(
        pairwise_kind, flat_val, flat_val, weight, bias, in_weight, in_bias, out_weight, out_bias);
    auto edges = softsign(scores);
    auto delta_state =
        torch::bmm(
            edges.to(state_acc_dtype),
            flat_projected_state.to(state_acc_dtype).unsqueeze(-1))
            .squeeze(-1);
    auto delta_val = torch::bmm(edges.to(val_acc_dtype), flat_projected_val.to(val_acc_dtype));
    return {
        reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, num_nodes),
        reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, num_nodes, out_dim),
    };
  }

  std::vector<torch::Tensor> state_blocks;
  std::vector<torch::Tensor> val_blocks;

  const auto target_step = target_block_size <= 0 ? num_nodes : std::min(target_block_size, num_nodes);
  const auto source_step = source_block_size <= 0 ? num_nodes : std::min(source_block_size, num_nodes);

  for (int64_t target_start = 0; target_start < num_nodes; target_start += target_step) {
    const auto target_end = std::min(target_start + target_step, num_nodes);
    auto target_val = flat_val.slice(1, target_start, target_end);
    auto target_state_acc = allocate_accumulator(
        {batch_flat, target_end - target_start}, flat_projected_state, state_acc_dtype);
    auto target_val_acc = allocate_accumulator(
        {batch_flat, target_end - target_start, out_dim}, flat_projected_val, val_acc_dtype);

    for (int64_t source_start = 0; source_start < num_nodes; source_start += source_step) {
      const auto source_end = std::min(source_start + source_step, num_nodes);
      auto source_val = flat_val.slice(1, source_start, source_end);
      auto scores = pairwise_scores(
          pairwise_kind, target_val, source_val, weight, bias, in_weight, in_bias, out_weight, out_bias);
      auto edges = softsign(scores);
      auto state_edges = edges.to(state_acc_dtype);
      auto val_edges = edges.to(val_acc_dtype);
      auto source_state =
          flat_projected_state.slice(1, source_start, source_end).to(state_acc_dtype);
      auto source_proj_val =
          flat_projected_val.slice(1, source_start, source_end).to(val_acc_dtype);

      target_state_acc = target_state_acc +
                         torch::bmm(state_edges, source_state.unsqueeze(-1)).squeeze(-1);
      target_val_acc = target_val_acc + torch::bmm(val_edges, source_proj_val);
    }

    state_blocks.push_back(target_state_acc.to(projected_state.scalar_type()));
    val_blocks.push_back(target_val_acc.to(projected_val.scalar_type()));
  }

  return {
      reshape_state(torch::cat(state_blocks, 1), batch_sizes, num_nodes),
      reshape_val(torch::cat(val_blocks, 1), batch_sizes, num_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> propagation_query_dense(
    const std::string& pairwise_kind,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    const std::string& edge_compress_name,
    const torch::Tensor& query_val,
    const torch::Tensor& source_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t query_block_size,
    int64_t source_block_size) {
  require_known_edge_compress(edge_compress_name);
  require_supported_device(query_val, "query_val");
  require_supported_device(source_val, "source_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");
  require_query_source_shapes(query_val, source_val, projected_state, projected_val);

  auto batch_sizes = batch_shape(query_val, 2);
  auto flat_query_val = flatten_val(query_val).contiguous();
  auto flat_source_val = flatten_val(source_val).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_query_val.size(0);
  const auto query_nodes = flat_query_val.size(1);
  const auto source_nodes = flat_source_val.size(1);
  const auto out_dim = flat_projected_val.size(2);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());

  if (pairwise_kind != "hadamard_mlp" &&
      can_use_full_dense_logits(flat_query_val, batch_flat, query_nodes, source_nodes)) {
    auto scores = pairwise_scores(
        pairwise_kind,
        flat_query_val,
        flat_source_val,
        weight,
        bias,
        in_weight,
        in_bias,
        out_weight,
        out_bias);
    auto edges = softsign(scores);
    auto delta_state =
        torch::bmm(
            edges.to(state_acc_dtype),
            flat_projected_state.to(state_acc_dtype).unsqueeze(-1))
            .squeeze(-1);
    auto delta_val = torch::bmm(edges.to(val_acc_dtype), flat_projected_val.to(val_acc_dtype));
    return {
        reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, query_nodes),
        reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, query_nodes, out_dim),
    };
  }

  std::vector<torch::Tensor> state_blocks;
  std::vector<torch::Tensor> val_blocks;
  const auto query_step =
      query_block_size <= 0 ? query_nodes : std::min(query_block_size, query_nodes);
  const auto source_step =
      source_block_size <= 0 ? source_nodes : std::min(source_block_size, source_nodes);

  for (int64_t query_start = 0; query_start < query_nodes; query_start += query_step) {
    const auto query_end = std::min(query_start + query_step, query_nodes);
    auto query_block = flat_query_val.slice(1, query_start, query_end);
    auto query_state_acc = allocate_accumulator(
        {batch_flat, query_end - query_start}, flat_projected_state, state_acc_dtype);
    auto query_val_acc = allocate_accumulator(
        {batch_flat, query_end - query_start, out_dim}, flat_projected_val, val_acc_dtype);

    for (int64_t source_start = 0; source_start < source_nodes; source_start += source_step) {
      const auto source_end = std::min(source_start + source_step, source_nodes);
      auto source_block = flat_source_val.slice(1, source_start, source_end);
      auto scores = pairwise_scores(
          pairwise_kind,
          query_block,
          source_block,
          weight,
          bias,
          in_weight,
          in_bias,
          out_weight,
          out_bias);
      auto edges = softsign(scores);
      query_state_acc = query_state_acc +
                        torch::bmm(
                            edges.to(state_acc_dtype),
                            flat_projected_state.slice(1, source_start, source_end)
                                .to(state_acc_dtype)
                                .unsqueeze(-1))
                            .squeeze(-1);
      query_val_acc = query_val_acc +
                      torch::bmm(
                          edges.to(val_acc_dtype),
                          flat_projected_val.slice(1, source_start, source_end)
                              .to(val_acc_dtype));
    }
    state_blocks.push_back(query_state_acc.to(projected_state.scalar_type()));
    val_blocks.push_back(query_val_acc.to(projected_val.scalar_type()));
  }

  return {
      reshape_state(torch::cat(state_blocks, 1), batch_sizes, query_nodes),
      reshape_val(torch::cat(val_blocks, 1), batch_sizes, query_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> propagation_window(
    const std::string& pairwise_kind,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    const std::string& edge_compress_name,
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    int64_t target_block_size,
    int64_t source_block_size) {
  require_known_edge_compress(edge_compress_name);
  require_supported_device(layer_val, "layer_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");
  if (window < 0) {
    throw std::runtime_error("window must be non-negative.");
  }

  auto batch_sizes = batch_shape(layer_val, 2);
  auto flat_val = flatten_val(layer_val).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_val.size(0);
  const auto num_nodes = flat_val.size(1);
  const auto out_dim = flat_projected_val.size(2);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());
  std::vector<torch::Tensor> state_blocks;
  std::vector<torch::Tensor> val_blocks;

  const auto target_step = target_block_size <= 0 ? num_nodes : std::min(target_block_size, num_nodes);
  const auto source_step = source_block_size <= 0 ? num_nodes : std::min(source_block_size, num_nodes);

  for (int64_t target_start = 0; target_start < num_nodes; target_start += target_step) {
    const auto target_end = std::min(target_start + target_step, num_nodes);
    auto target_val = flat_val.slice(1, target_start, target_end);
    const auto source_floor = std::max<int64_t>(0, target_start - window);
    const auto source_ceiling = target_end;
    auto target_state_acc = allocate_accumulator(
        {batch_flat, target_end - target_start}, flat_projected_state, state_acc_dtype);
    auto target_val_acc = allocate_accumulator(
        {batch_flat, target_end - target_start, out_dim}, flat_projected_val, val_acc_dtype);

    for (int64_t source_start = source_floor; source_start < source_ceiling; source_start += source_step) {
      const auto source_end = std::min(source_start + source_step, source_ceiling);
      auto source_val = flat_val.slice(1, source_start, source_end);
      auto scores = pairwise_scores(
          pairwise_kind, target_val, source_val, weight, bias, in_weight, in_bias, out_weight, out_bias);
      auto mask = causal_window_mask(
                      target_start, target_end, source_start, source_end, window, layer_val.device())
                      .unsqueeze(0)
                      .to(scores.scalar_type());
      auto edges = softsign(scores) * mask;
      auto state_edges = edges.to(state_acc_dtype);
      auto val_edges = edges.to(val_acc_dtype);
      auto source_state =
          flat_projected_state.slice(1, source_start, source_end).to(state_acc_dtype);
      auto source_proj_val =
          flat_projected_val.slice(1, source_start, source_end).to(val_acc_dtype);

      target_state_acc = target_state_acc +
                         torch::bmm(state_edges, source_state.unsqueeze(-1)).squeeze(-1);
      target_val_acc = target_val_acc + torch::bmm(val_edges, source_proj_val);
    }

    state_blocks.push_back(target_state_acc.to(projected_state.scalar_type()));
    val_blocks.push_back(target_val_acc.to(projected_val.scalar_type()));
  }

  return {
      reshape_state(torch::cat(state_blocks, 1), batch_sizes, num_nodes),
      reshape_val(torch::cat(val_blocks, 1), batch_sizes, num_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> propagation_topk(
    const std::string& pairwise_kind,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    const std::string& edge_compress_name,
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    int64_t target_block_size,
    int64_t source_block_size) {
  require_known_edge_compress(edge_compress_name);
  require_supported_device(layer_val, "layer_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");
  if (topk <= 0) {
    throw std::runtime_error("topk must be positive.");
  }

  auto batch_sizes = batch_shape(layer_val, 2);
  auto flat_val = flatten_val(layer_val).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_val.size(0);
  const auto num_nodes = flat_val.size(1);
  const auto out_dim = flat_projected_val.size(2);
  const auto k = std::min<int64_t>(topk, num_nodes);

  if (k == num_nodes) {
    return propagation_dense(
        pairwise_kind,
        weight,
        bias,
        in_weight,
        in_bias,
        out_weight,
        out_bias,
        edge_compress_name,
        layer_val,
        projected_state,
        projected_val,
        target_block_size,
        source_block_size);
  }

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());
  std::vector<torch::Tensor> state_blocks;
  std::vector<torch::Tensor> val_blocks;

  const auto target_step = target_block_size <= 0 ? num_nodes : std::min(target_block_size, num_nodes);
  const auto source_step = source_block_size <= 0 ? num_nodes : std::min(source_block_size, num_nodes);

  for (int64_t target_start = 0; target_start < num_nodes; target_start += target_step) {
    const auto target_end = std::min(target_start + target_step, num_nodes);
    const auto target_nodes = target_end - target_start;
    auto target_val = flat_val.slice(1, target_start, target_end);
    auto best_scores = torch::full(
        {batch_flat, target_nodes, k},
        -std::numeric_limits<float>::infinity(),
        flat_val.options());
    auto best_indices = torch::zeros(
        {batch_flat, target_nodes, k},
        flat_val.options().dtype(torch::kLong));

    for (int64_t source_start = 0; source_start < num_nodes; source_start += source_step) {
      const auto source_end = std::min(source_start + source_step, num_nodes);
      auto source_val = flat_val.slice(1, source_start, source_end);
      auto scores = pairwise_scores(
          pairwise_kind, target_val, source_val, weight, bias, in_weight, in_bias, out_weight, out_bias);
      auto source_indices = torch::arange(
                                source_start,
                                source_end,
                                flat_val.options().dtype(torch::kLong))
                                .view({1, 1, source_end - source_start})
                                .expand({batch_flat, target_nodes, source_end - source_start});
      auto candidate_scores = torch::cat({best_scores, scores}, -1);
      auto candidate_indices = torch::cat({best_indices, source_indices}, -1);
      auto topk_result = candidate_scores.topk(k, -1, true, true);
      best_scores = std::get<0>(topk_result);
      best_indices = candidate_indices.gather(-1, std::get<1>(topk_result));
    }

    auto gathered = gather_by_indices(flat_projected_state, flat_projected_val, best_indices);
    auto selected_state = std::get<0>(gathered);
    auto selected_val = std::get<1>(gathered);
    auto edges = softsign(best_scores);
    auto state_block =
        (edges.to(state_acc_dtype) * selected_state.to(state_acc_dtype)).sum(-1);
    auto val_block =
        (edges.to(val_acc_dtype).unsqueeze(-1) * selected_val.to(val_acc_dtype)).sum(-2);
    state_blocks.push_back(state_block.to(projected_state.scalar_type()));
    val_blocks.push_back(val_block.to(projected_val.scalar_type()));
  }

  return {
      reshape_state(torch::cat(state_blocks, 1), batch_sizes, num_nodes),
      reshape_val(torch::cat(val_blocks, 1), batch_sizes, num_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> transition_dense(
    const std::string& route_kind,
    const torch::Tensor& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t dst_nodes,
    int64_t src_block_size,
    int64_t dst_block_size) {
  require_supported_device(sender_strength, "sender_strength");
  require_supported_device(src_val, "src_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");

  auto batch_sizes = batch_shape(src_val, 2);
  auto flat_src_val = flatten_val(src_val).contiguous();
  auto flat_sender_strength = flatten_state(sender_strength).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_src_val.size(0);
  const auto src_nodes = flat_src_val.size(1);
  const auto out_dim = flat_projected_val.size(2);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());

  if (route_kind != "source_target_hadamard_mlp_route" &&
      can_use_full_dense_logits(flat_src_val, batch_flat, src_nodes, dst_nodes)) {
    auto route_context = prepare_route_context(route_kind, flat_src_val, in_weight, in_bias);
    auto logits = route_block_logits(
        route_kind, route_context, in_weight, in_bias, out_weight, out_bias, 0, dst_nodes);
    auto routes = torch::softmax(logits, -1);
    auto state_sender =
        (flat_sender_strength.to(state_acc_dtype) * flat_projected_state.to(state_acc_dtype));
    auto val_sender =
        (flat_sender_strength.to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.to(val_acc_dtype));
    auto transport = routes.transpose(1, 2).contiguous();
    auto delta_state =
        torch::bmm(transport.to(state_acc_dtype), state_sender.unsqueeze(-1)).squeeze(-1);
    auto delta_val = torch::bmm(transport.to(val_acc_dtype), val_sender);
    return {
        reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, dst_nodes),
        reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
    };
  }

  auto delta_state = allocate_accumulator({batch_flat, dst_nodes}, flat_projected_state, state_acc_dtype);
  auto delta_val = allocate_accumulator({batch_flat, dst_nodes, out_dim}, flat_projected_val, val_acc_dtype);

  const auto src_step = src_block_size <= 0 ? src_nodes : std::min(src_block_size, src_nodes);
  const auto dst_step = dst_block_size <= 0 ? dst_nodes : std::min(dst_block_size, dst_nodes);

  for (int64_t src_start = 0; src_start < src_nodes; src_start += src_step) {
    const auto src_end = std::min(src_start + src_step, src_nodes);
    const auto block_nodes = src_end - src_start;
    auto src_block = flat_src_val.slice(1, src_start, src_end);
    auto route_context = prepare_route_context(route_kind, src_block, in_weight, in_bias);

    torch::Tensor running_max;
    torch::Tensor running_sum;
    bool initialized = false;

    for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
      const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
      auto logits = route_block_logits(
          route_kind, route_context, in_weight, in_bias, out_weight, out_bias, dst_start, dst_end);
      auto block_max = std::get<0>(logits.max(-1));
      auto block_exp = torch::exp(logits - block_max.unsqueeze(-1));
      auto block_sum = block_exp.sum(-1);

      if (!initialized) {
        running_max = block_max;
        running_sum = block_sum;
        initialized = true;
      } else {
        auto next_max = torch::maximum(running_max, block_max);
        auto running_scale = torch::exp(running_max - next_max);
        auto block_scale = torch::exp(block_max - next_max);
        running_sum = running_sum * running_scale + block_sum * block_scale;
        running_max = next_max;
      }
    }

    auto state_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(state_acc_dtype) *
         flat_projected_state.slice(1, src_start, src_end).to(state_acc_dtype));
    auto val_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.slice(1, src_start, src_end).to(val_acc_dtype));

    for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
      const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
      auto logits = route_block_logits(
          route_kind, route_context, in_weight, in_bias, out_weight, out_bias, dst_start, dst_end);
      auto routes = torch::exp(logits - running_max.unsqueeze(-1)) / running_sum.unsqueeze(-1);
      auto transport = routes.transpose(1, 2).contiguous();
      delta_state.slice(1, dst_start, dst_end) =
          delta_state.slice(1, dst_start, dst_end) +
          torch::bmm(transport.to(state_acc_dtype), state_sender.unsqueeze(-1)).squeeze(-1);
      delta_val.slice(1, dst_start, dst_end) =
          delta_val.slice(1, dst_start, dst_end) +
          torch::bmm(transport.to(val_acc_dtype), val_sender);
    }
  }

  return {
      reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, dst_nodes),
      reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> transition_topk(
    const std::string& route_kind,
    const torch::Tensor& in_weight,
    const c10::optional<torch::Tensor>& in_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t dst_nodes,
    int64_t topk,
    int64_t src_block_size,
    int64_t dst_block_size) {
  require_supported_device(sender_strength, "sender_strength");
  require_supported_device(src_val, "src_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");
  if (topk <= 0) {
    throw std::runtime_error("topk must be positive.");
  }

  const auto k = std::min<int64_t>(topk, dst_nodes);
  if (k == dst_nodes) {
    return transition_dense(
        route_kind,
        in_weight,
        in_bias,
        out_weight,
        out_bias,
        sender_strength,
        src_val,
        projected_state,
        projected_val,
        dst_nodes,
        src_block_size,
        dst_block_size);
  }

  auto batch_sizes = batch_shape(src_val, 2);
  auto flat_src_val = flatten_val(src_val).contiguous();
  auto flat_sender_strength = flatten_state(sender_strength).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_src_val.size(0);
  const auto src_nodes = flat_src_val.size(1);
  const auto out_dim = flat_projected_val.size(2);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());
  auto delta_state = allocate_accumulator({batch_flat, dst_nodes}, flat_projected_state, state_acc_dtype);
  auto delta_val = allocate_accumulator({batch_flat, dst_nodes, out_dim}, flat_projected_val, val_acc_dtype);

  const auto src_step = src_block_size <= 0 ? src_nodes : std::min(src_block_size, src_nodes);
  const auto dst_step = dst_block_size <= 0 ? dst_nodes : std::min(dst_block_size, dst_nodes);

  for (int64_t src_start = 0; src_start < src_nodes; src_start += src_step) {
    const auto src_end = std::min(src_start + src_step, src_nodes);
    const auto block_nodes = src_end - src_start;
    auto src_block = flat_src_val.slice(1, src_start, src_end);
    auto route_context = prepare_route_context(route_kind, src_block, in_weight, in_bias);
    auto best_values = torch::full(
        {batch_flat, block_nodes, k},
        -std::numeric_limits<float>::infinity(),
        flat_src_val.options());
    auto best_indices = torch::zeros(
        {batch_flat, block_nodes, k},
        flat_src_val.options().dtype(torch::kLong));

    for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
      const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
      auto logits = route_block_logits(
          route_kind, route_context, in_weight, in_bias, out_weight, out_bias, dst_start, dst_end);
      auto block_indices = torch::arange(
                               dst_start,
                               dst_end,
                               flat_src_val.options().dtype(torch::kLong))
                               .view({1, 1, dst_end - dst_start})
                               .expand({batch_flat, block_nodes, dst_end - dst_start});
      auto candidate_values = torch::cat({best_values, logits}, -1);
      auto candidate_indices = torch::cat({best_indices, block_indices}, -1);
      auto topk_result = candidate_values.topk(k, -1, true, true);
      best_values = std::get<0>(topk_result);
      best_indices = candidate_indices.gather(-1, std::get<1>(topk_result));
    }

    auto routes = torch::softmax(best_values, -1);
    auto state_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(state_acc_dtype) *
         flat_projected_state.slice(1, src_start, src_end).to(state_acc_dtype))
            .unsqueeze(-1);
    auto state_contrib = (routes.to(state_acc_dtype) * state_sender).reshape({batch_flat, -1});
    delta_state.scatter_add_(1, best_indices.reshape({batch_flat, -1}), state_contrib);

    auto val_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.slice(1, src_start, src_end).to(val_acc_dtype))
            .unsqueeze(-2);
    auto val_contrib =
        (routes.to(val_acc_dtype).unsqueeze(-1) * val_sender).reshape({batch_flat, -1, out_dim});
    auto scatter_index =
        best_indices.unsqueeze(-1).expand({batch_flat, block_nodes, k, out_dim}).reshape(
            {batch_flat, -1, out_dim});
    delta_val.scatter_add_(1, scatter_index, val_contrib);
  }

  return {
      reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, dst_nodes),
      reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> transition_pairwise_dense(
    const std::string& route_kind,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& source_bias,
    const c10::optional<torch::Tensor>& target_weight,
    const c10::optional<torch::Tensor>& target_bias,
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& hidden_weight,
    const c10::optional<torch::Tensor>& hidden_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    double temperature,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& dst_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t src_block_size,
    int64_t dst_block_size) {
  require_supported_device(sender_strength, "sender_strength");
  require_supported_device(src_val, "src_val");
  require_supported_device(dst_val, "dst_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");

  auto batch_sizes = batch_shape(src_val, 2);
  auto flat_src_val = flatten_val(src_val).contiguous();
  auto flat_dst_val = flatten_val(dst_val).contiguous();
  auto flat_sender_strength = flatten_state(sender_strength).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_src_val.size(0);
  const auto src_nodes = flat_src_val.size(1);
  const auto dst_nodes = flat_dst_val.size(1);
  const auto out_dim = flat_projected_val.size(2);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());

  if (route_kind != "source_target_hadamard_mlp_route" &&
      can_use_full_dense_logits(flat_src_val, batch_flat, src_nodes, dst_nodes)) {
    auto logits = pairwise_route_block_logits(
        route_kind,
        flat_src_val,
        flat_dst_val,
        source_weight,
        source_bias,
        target_weight,
        target_bias,
        core_weight,
        bias,
        hidden_weight,
        hidden_bias,
        out_weight,
        out_bias,
        temperature);
    auto routes = torch::softmax(logits, -1);
    auto state_sender =
        (flat_sender_strength.to(state_acc_dtype) * flat_projected_state.to(state_acc_dtype));
    auto val_sender =
        (flat_sender_strength.to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.to(val_acc_dtype));
    auto transport = routes.transpose(1, 2).contiguous();
    auto delta_state =
        torch::bmm(transport.to(state_acc_dtype), state_sender.unsqueeze(-1)).squeeze(-1);
    auto delta_val = torch::bmm(transport.to(val_acc_dtype), val_sender);
    return {
        reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, dst_nodes),
        reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
    };
  }

  auto delta_state =
      allocate_accumulator({batch_flat, dst_nodes}, flat_projected_state, state_acc_dtype);
  auto delta_val =
      allocate_accumulator({batch_flat, dst_nodes, out_dim}, flat_projected_val, val_acc_dtype);

  const auto src_step = src_block_size <= 0 ? src_nodes : std::min(src_block_size, src_nodes);
  const auto dst_step = dst_block_size <= 0 ? dst_nodes : std::min(dst_block_size, dst_nodes);

  for (int64_t src_start = 0; src_start < src_nodes; src_start += src_step) {
    const auto src_end = std::min(src_start + src_step, src_nodes);
    auto src_block = flat_src_val.slice(1, src_start, src_end);

    torch::Tensor running_max;
    torch::Tensor running_sum;
    bool initialized = false;

    for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
      const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
      auto dst_block = flat_dst_val.slice(1, dst_start, dst_end);
      auto logits = pairwise_route_block_logits(
          route_kind,
          src_block,
          dst_block,
          source_weight,
          source_bias,
          target_weight,
          target_bias,
          core_weight,
          bias,
          hidden_weight,
          hidden_bias,
          out_weight,
          out_bias,
          temperature);
      auto block_max = std::get<0>(logits.max(-1));
      auto block_exp = torch::exp(logits - block_max.unsqueeze(-1));
      auto block_sum = block_exp.sum(-1);

      if (!initialized) {
        running_max = block_max;
        running_sum = block_sum;
        initialized = true;
      } else {
        auto next_max = torch::maximum(running_max, block_max);
        auto running_scale = torch::exp(running_max - next_max);
        auto block_scale = torch::exp(block_max - next_max);
        running_sum = running_sum * running_scale + block_sum * block_scale;
        running_max = next_max;
      }
    }

    auto state_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(state_acc_dtype) *
         flat_projected_state.slice(1, src_start, src_end).to(state_acc_dtype));
    auto val_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.slice(1, src_start, src_end).to(val_acc_dtype));

    for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
      const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
      auto dst_block = flat_dst_val.slice(1, dst_start, dst_end);
      auto logits = pairwise_route_block_logits(
          route_kind,
          src_block,
          dst_block,
          source_weight,
          source_bias,
          target_weight,
          target_bias,
          core_weight,
          bias,
          hidden_weight,
          hidden_bias,
          out_weight,
          out_bias,
          temperature);
      auto routes = torch::exp(logits - running_max.unsqueeze(-1)) / running_sum.unsqueeze(-1);
      auto transport = routes.transpose(1, 2).contiguous();
      delta_state.slice(1, dst_start, dst_end) =
          delta_state.slice(1, dst_start, dst_end) +
          torch::bmm(transport.to(state_acc_dtype), state_sender.unsqueeze(-1)).squeeze(-1);
      delta_val.slice(1, dst_start, dst_end) =
          delta_val.slice(1, dst_start, dst_end) +
          torch::bmm(transport.to(val_acc_dtype), val_sender);
    }
  }

  return {
      reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, dst_nodes),
      reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor> transition_pairwise_topk(
    const std::string& route_kind,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& source_bias,
    const c10::optional<torch::Tensor>& target_weight,
    const c10::optional<torch::Tensor>& target_bias,
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& hidden_weight,
    const c10::optional<torch::Tensor>& hidden_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    double temperature,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& dst_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    int64_t src_block_size,
    int64_t dst_block_size) {
  require_supported_device(sender_strength, "sender_strength");
  require_supported_device(src_val, "src_val");
  require_supported_device(dst_val, "dst_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");
  if (topk <= 0) {
    throw std::runtime_error("topk must be positive.");
  }

  auto batch_sizes = batch_shape(src_val, 2);
  auto flat_src_val = flatten_val(src_val).contiguous();
  auto flat_dst_val = flatten_val(dst_val).contiguous();
  auto flat_sender_strength = flatten_state(sender_strength).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_src_val.size(0);
  const auto src_nodes = flat_src_val.size(1);
  const auto dst_nodes = flat_dst_val.size(1);
  const auto out_dim = flat_projected_val.size(2);
  const auto k = std::min<int64_t>(topk, dst_nodes);

  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());
  auto delta_state = allocate_accumulator({batch_flat, dst_nodes}, flat_projected_state, state_acc_dtype);
  auto delta_val = allocate_accumulator({batch_flat, dst_nodes, out_dim}, flat_projected_val, val_acc_dtype);

  const auto src_step = src_block_size <= 0 ? src_nodes : std::min(src_block_size, src_nodes);
  const auto dst_step = dst_block_size <= 0 ? dst_nodes : std::min(dst_block_size, dst_nodes);

  for (int64_t src_start = 0; src_start < src_nodes; src_start += src_step) {
    const auto src_end = std::min(src_start + src_step, src_nodes);
    const auto block_nodes = src_end - src_start;
    auto src_block = flat_src_val.slice(1, src_start, src_end);
    auto best_values = torch::full(
        {batch_flat, block_nodes, k},
        -std::numeric_limits<float>::infinity(),
        flat_src_val.options());
    auto best_indices = torch::zeros(
        {batch_flat, block_nodes, k},
        flat_src_val.options().dtype(torch::kLong));

    for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
      const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
      auto dst_block = flat_dst_val.slice(1, dst_start, dst_end);
      auto logits = pairwise_route_block_logits(
          route_kind,
          src_block,
          dst_block,
          source_weight,
          source_bias,
          target_weight,
          target_bias,
          core_weight,
          bias,
          hidden_weight,
          hidden_bias,
          out_weight,
          out_bias,
          temperature);
      auto block_indices = torch::arange(
                               dst_start,
                               dst_end,
                               flat_src_val.options().dtype(torch::kLong))
                               .view({1, 1, dst_end - dst_start})
                               .expand({batch_flat, block_nodes, dst_end - dst_start});
      auto candidate_values = torch::cat({best_values, logits}, -1);
      auto candidate_indices = torch::cat({best_indices, block_indices}, -1);
      auto topk_result = candidate_values.topk(k, -1, true, true);
      best_values = std::get<0>(topk_result);
      best_indices = candidate_indices.gather(-1, std::get<1>(topk_result));
    }

    auto routes = torch::softmax(best_values, -1);
    auto state_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(state_acc_dtype) *
         flat_projected_state.slice(1, src_start, src_end).to(state_acc_dtype))
            .unsqueeze(-1);
    auto state_contrib = (routes.to(state_acc_dtype) * state_sender).reshape({batch_flat, -1});
    delta_state.scatter_add_(1, best_indices.reshape({batch_flat, -1}), state_contrib);

    auto val_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.slice(1, src_start, src_end).to(val_acc_dtype))
            .unsqueeze(-2);
    auto val_contrib =
        (routes.to(val_acc_dtype).unsqueeze(-1) * val_sender).reshape({batch_flat, -1, out_dim});
    auto scatter_index =
        best_indices.unsqueeze(-1).expand({batch_flat, block_nodes, k, out_dim}).reshape(
            {batch_flat, -1, out_dim});
    delta_val.scatter_add_(1, scatter_index, val_contrib);
  }

  return {
      reshape_state(delta_state.to(projected_state.scalar_type()), batch_sizes, dst_nodes),
      reshape_val(delta_val.to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> transition_query_topk_core(
    const std::string& route_kind,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& source_bias,
    const c10::optional<torch::Tensor>& target_weight,
    const c10::optional<torch::Tensor>& target_bias,
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& hidden_weight,
    const c10::optional<torch::Tensor>& hidden_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    double temperature,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& query_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    int64_t query_block_size,
    int64_t source_block_size,
    bool use_cuda_reduce) {
  require_supported_device(sender_strength, "sender_strength");
  require_supported_device(src_val, "src_val");
  require_supported_device(query_val, "query_val");
  require_supported_device(projected_state, "projected_state");
  require_supported_device(projected_val, "projected_val");
  require_query_source_shapes(query_val, src_val, projected_state, projected_val);
  if (topk <= 0) {
    throw std::runtime_error("topk must be positive.");
  }

  auto batch_sizes = batch_shape(query_val, 2);
  auto flat_src_val = flatten_val(src_val).contiguous();
  auto flat_query_val = flatten_val(query_val).contiguous();
  auto flat_sender_strength = flatten_state(sender_strength).contiguous();
  auto flat_projected_state = flatten_state(projected_state).contiguous();
  auto flat_projected_val = flatten_val(projected_val).contiguous();
  const auto batch_flat = flat_src_val.size(0);
  const auto source_nodes = flat_src_val.size(1);
  const auto query_nodes = flat_query_val.size(1);
  const auto out_dim = flat_projected_val.size(2);
  const auto k = std::min<int64_t>(topk, source_nodes);

  auto selected_scores = torch::full(
      {batch_flat, query_nodes, k},
      -std::numeric_limits<float>::infinity(),
      flat_query_val.options());
  auto selected_indices = torch::zeros(
      {batch_flat, query_nodes, k},
      flat_query_val.options().dtype(torch::kLong));

  const auto query_step =
      query_block_size <= 0 ? query_nodes : std::min(query_block_size, query_nodes);
  const auto source_step =
      source_block_size <= 0 ? source_nodes : std::min(source_block_size, source_nodes);

  for (int64_t query_start = 0; query_start < query_nodes; query_start += query_step) {
    const auto query_end = std::min(query_start + query_step, query_nodes);
    const auto block_nodes = query_end - query_start;
    auto query_block = flat_query_val.slice(1, query_start, query_end);
    auto best_values = torch::full(
        {batch_flat, block_nodes, k},
        -std::numeric_limits<float>::infinity(),
        flat_query_val.options());
    auto best_indices = torch::zeros(
        {batch_flat, block_nodes, k},
        flat_query_val.options().dtype(torch::kLong));

    for (int64_t source_start = 0; source_start < source_nodes; source_start += source_step) {
      const auto source_end = std::min(source_start + source_step, source_nodes);
      auto source_block = flat_src_val.slice(1, source_start, source_end);
      auto logits = pairwise_route_block_logits(
                        route_kind,
                        source_block,
                        query_block,
                        source_weight,
                        source_bias,
                        target_weight,
                        target_bias,
                        core_weight,
                        bias,
                        hidden_weight,
                        hidden_bias,
                        out_weight,
                        out_bias,
                        temperature)
                        .transpose(1, 2)
                        .contiguous();
      auto block_indices = torch::arange(
                               source_start,
                               source_end,
                               flat_query_val.options().dtype(torch::kLong))
                               .view({1, 1, source_end - source_start})
                               .expand({batch_flat, block_nodes, source_end - source_start});
      auto candidate_values = torch::cat({best_values, logits}, -1);
      auto candidate_indices = torch::cat({best_indices, block_indices}, -1);
      auto topk_result = candidate_values.topk(k, -1, true, true);
      best_values = std::get<0>(topk_result);
      best_indices = candidate_indices.gather(-1, std::get<1>(topk_result));
    }

    selected_scores.slice(1, query_start, query_end).copy_(best_values);
    selected_indices.slice(1, query_start, query_end).copy_(best_indices);
  }

  auto routes = torch::softmax(selected_scores, -1);

  torch::Tensor delta_state;
  torch::Tensor delta_val;
  if (
      use_cuda_reduce && query_val.is_cuda()
#ifdef WITH_CUDA
      && jakal_net_query_topk_reduce_cuda_available()
#endif
  ) {
    auto weighted_state = (flat_sender_strength * flat_projected_state).contiguous();
    auto weighted_val =
        (flat_sender_strength.unsqueeze(-1) * flat_projected_val).contiguous();
    auto reduced = query_topk_reduce_cuda_wrapper(
        routes.contiguous(),
        selected_indices.contiguous(),
        weighted_state,
        weighted_val);
    delta_state = std::get<0>(reduced);
    delta_val = std::get<1>(reduced);
  } else {
    const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
    const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());
    auto weighted_state =
        (flat_sender_strength.to(state_acc_dtype) *
         flat_projected_state.to(state_acc_dtype))
            .contiguous();
    auto weighted_val =
        (flat_sender_strength.to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.to(val_acc_dtype))
            .contiguous();
    auto gathered = gather_by_indices(weighted_state, weighted_val, selected_indices);
    delta_state =
        (routes.to(state_acc_dtype) * std::get<0>(gathered)).sum(-1).to(projected_state.scalar_type());
    delta_val =
        (routes.to(val_acc_dtype).unsqueeze(-1) * std::get<1>(gathered))
            .sum(-2)
            .to(projected_val.scalar_type());
  }

  std::vector<int64_t> selection_shape = batch_sizes;
  selection_shape.push_back(query_nodes);
  selection_shape.push_back(k);
  return {
      reshape_state(delta_state, batch_sizes, query_nodes),
      reshape_val(delta_val, batch_sizes, query_nodes, out_dim),
      selected_scores.reshape(selection_shape),
      selected_indices.reshape(selection_shape),
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transition_query_topk_select(
    const std::string& route_kind,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& target_weight,
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& bias,
    double temperature,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& query_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    int64_t query_block_size,
    int64_t source_block_size,
    bool use_cuda_reduce) {
  return transition_query_topk_core(
      route_kind,
      source_weight,
      c10::nullopt,
      target_weight,
      c10::nullopt,
      core_weight,
      bias,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      temperature,
      sender_strength,
      src_val,
      query_val,
      projected_state,
      projected_val,
      topk,
      query_block_size,
      source_block_size,
      use_cuda_reduce);
}

std::tuple<torch::Tensor, torch::Tensor> transition_query_topk(
    const std::string& route_kind,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& source_bias,
    const c10::optional<torch::Tensor>& target_weight,
    const c10::optional<torch::Tensor>& target_bias,
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& hidden_weight,
    const c10::optional<torch::Tensor>& hidden_bias,
    const c10::optional<torch::Tensor>& out_weight,
    const c10::optional<torch::Tensor>& out_bias,
    double temperature,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& query_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    int64_t query_block_size,
    int64_t source_block_size,
    bool use_cuda_reduce) {
  auto result = transition_query_topk_core(
      route_kind,
      source_weight,
      source_bias,
      target_weight,
      target_bias,
      core_weight,
      bias,
      hidden_weight,
      hidden_bias,
      out_weight,
      out_bias,
      temperature,
      sender_strength,
      src_val,
      query_val,
      projected_state,
      projected_val,
      topk,
      query_block_size,
      source_block_size,
      use_cuda_reduce);
  return {std::get<0>(result), std::get<1>(result)};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supported_ops", &supported_ops, "List supported native ops");
  m.def("supported_devices", &supported_devices, "List supported native devices");
  m.def("backend_name", &backend_name, "Return native backend name");
  m.def("propagation_dense", &propagation_dense, "Native dense propagation");
  m.def("propagation_query_dense", &propagation_query_dense, "Native dense query-conditioned propagation");
  m.def("propagation_window", &propagation_window, "Native window propagation");
  m.def("propagation_topk", &propagation_topk, "Native top-k propagation");
  m.def("query_topk_reduce_cuda", &query_topk_reduce_cuda_wrapper, "CUDA query top-k reduction");
  m.def(
      "query_topk_reduce_backward_cuda",
      &query_topk_reduce_backward_cuda_wrapper,
      "CUDA query top-k reduction backward");
  m.def("softsign_backward_cuda", &softsign_backward_cuda_wrapper, "CUDA softsign backward");
  m.def("softmax_backward_cuda", &softmax_backward_cuda_wrapper, "CUDA softmax backward");
  m.def("transition_dense", &transition_dense, "Native dense transition");
  m.def("transition_pairwise_dense", &transition_pairwise_dense, "Native pairwise dense transition");
  m.def("transition_pairwise_topk", &transition_pairwise_topk, "Native pairwise sparse transition");
  m.def("transition_query_topk", &transition_query_topk, "Native query-conditioned sparse transition");
  m.def(
      "transition_query_topk_select",
      &transition_query_topk_select,
      "Native query-conditioned sparse transition top-k selection");
  m.def("transition_topk", &transition_topk, "Native sparse transition");
  m.def(
      "diagonal_pairwise_topk_backward_cuda",
      &diagonal_pairwise_topk_backward_cuda_wrapper,
      "CUDA diagonal pairwise top-k backward");
  m.def(
      "low_rank_pairwise_topk_backward_cuda",
      &low_rank_pairwise_topk_backward_cuda_wrapper,
      "CUDA low-rank pairwise top-k backward");
}
