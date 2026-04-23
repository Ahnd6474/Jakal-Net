#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/csrc/autograd/autograd.h>
#include <c10/core/GradMode.h>

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

bool experimental_cuda_scan_fastpath_enabled() {
  if (const char* raw = std::getenv("JAKAL_NET_ENABLE_EXPERIMENTAL_CUDA_SCAN_FASTPATH")) {
    return std::string(raw) == "1" || std::string(raw) == "true" ||
           std::string(raw) == "TRUE" || std::string(raw) == "yes";
  }
  return false;
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
low_rank_pairwise_topk_forward_cuda_wrapper(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& weighted_projected_state,
    const torch::Tensor& weighted_projected_val,
    int64_t topk,
    double score_bias,
    int64_t compress_kind) {
#ifdef WITH_CUDA
  return jakal_net_low_rank_pairwise_topk_forward_cuda(
      weighted_projected_source,
      projected_target,
      weighted_projected_state,
      weighted_projected_val,
      topk,
      score_bias,
      compress_kind);
#else
  throw std::runtime_error(
      "low_rank_pairwise_topk_forward_cuda requires a CUDA-enabled build.");
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
low_rank_propagation_topk_forward_cuda_wrapper(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    double score_bias,
    int64_t compress_kind) {
#ifdef WITH_CUDA
  return jakal_net_low_rank_propagation_topk_forward_cuda(
      weighted_projected_source,
      projected_target,
      projected_state,
      projected_val,
      topk,
      score_bias,
      compress_kind);
#else
  throw std::runtime_error(
      "low_rank_propagation_topk_forward_cuda requires a CUDA-enabled build.");
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
low_rank_propagation_window_forward_cuda_wrapper(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias) {
#ifdef WITH_CUDA
  return jakal_net_low_rank_propagation_window_forward_cuda(
      weighted_projected_source,
      projected_target,
      projected_state,
      projected_val,
      window,
      score_bias);
#else
  throw std::runtime_error(
      "low_rank_propagation_window_forward_cuda requires a CUDA-enabled build.");
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
low_rank_propagation_window_entmax15_forward_cuda_wrapper(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias) {
#ifdef WITH_CUDA
  return jakal_net_low_rank_propagation_window_entmax15_forward_cuda(
      weighted_projected_source,
      projected_target,
      projected_state,
      projected_val,
      window,
      score_bias);
#else
  throw std::runtime_error(
      "low_rank_propagation_window_entmax15_forward_cuda requires a CUDA-enabled build.");
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
  if (edge_compress_name != "softsign" && edge_compress_name != "signed_entmax15") {
    throw std::runtime_error(
        "Only edge_compress_name in {'softsign', 'signed_entmax15'} is supported by the CPU native backend.");
  }
}

void require_known_route_compress(const std::string& route_compress_name) {
  if (route_compress_name != "softmax" && route_compress_name != "signed_entmax15") {
    throw std::runtime_error(
        "Only route_compress_name in {'softmax', 'signed_entmax15'} is supported by the CPU native backend.");
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

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

int64_t infer_multihead_count(
    const torch::Tensor& core_weight,
    const c10::optional<torch::Tensor>& source_weight,
    const c10::optional<torch::Tensor>& target_weight) {
  if (core_weight.dim() >= 2) {
    return core_weight.size(0);
  }
  if (source_weight.has_value() && source_weight.value().dim() >= 3) {
    return source_weight.value().size(0);
  }
  if (target_weight.has_value() && target_weight.value().dim() >= 3) {
    return target_weight.value().size(0);
  }
  throw std::runtime_error("Unable to infer multi-head count from packed weights.");
}

torch::Tensor select_head_tensor(
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

c10::optional<torch::Tensor> select_head_optional_tensor(
    const c10::optional<torch::Tensor>& tensor,
    int64_t head_index,
    int64_t num_heads) {
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return select_head_tensor(tensor.value(), head_index, num_heads);
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
  if (starts_with(pairwise_kind, "multihead_max_")) {
    const auto base_kind = pairwise_kind.substr(std::string("multihead_max_").size());
    if (base_kind == "low_rank_bilinear") {
      if (!in_weight.has_value() || !out_weight.has_value()) {
        throw std::runtime_error("multihead_max low_rank_bilinear is missing projection weights.");
      }
      auto cast_source = in_weight.value().to(source_val.scalar_type());
      auto cast_target = out_weight.value().to(target_val.scalar_type());
      auto cast_core = weight.to(source_val.scalar_type());
      auto projected_source = torch::einsum(
          "bid,hrd->bhir",
          std::vector<torch::Tensor>{source_val, cast_source});
      auto projected_target = torch::einsum(
          "bkd,hrd->bhkr",
          std::vector<torch::Tensor>{target_val, cast_target});
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[projected_source.dim() - 3] = cast_core.size(0);
      core_shape[projected_source.dim() - 1] = cast_core.size(1);
      auto weighted_source = projected_source * cast_core.view(core_shape);
      auto scores = torch::einsum(
          "bhir,bhkr->bhik",
          std::vector<torch::Tensor>{weighted_source, projected_target});
      if (bias.has_value()) {
        std::vector<int64_t> bias_shape(scores.dim(), 1);
        bias_shape[scores.dim() - 3] = bias.value().size(0);
        scores = scores + bias.value().to(scores.scalar_type()).view(bias_shape);
      }
      return std::get<0>(scores.max(1));
    }
    const auto num_heads = infer_multihead_count(weight, in_weight, out_weight);
    c10::optional<torch::Tensor> best_scores = c10::nullopt;
    for (int64_t head_index = 0; head_index < num_heads; ++head_index) {
      auto head_scores = pairwise_scores(
          base_kind,
          target_val,
          source_val,
          select_head_tensor(weight, head_index, num_heads),
          select_head_optional_tensor(bias, head_index, num_heads),
          select_head_optional_tensor(in_weight, head_index, num_heads),
          select_head_optional_tensor(in_bias, head_index, num_heads),
          select_head_optional_tensor(out_weight, head_index, num_heads),
          select_head_optional_tensor(out_bias, head_index, num_heads));
      if (best_scores.has_value()) {
        best_scores = torch::maximum(best_scores.value(), head_scores);
      } else {
        best_scores = head_scores;
      }
    }
    if (!best_scores.has_value()) {
      throw std::runtime_error("multihead_max pairwise kernel requires at least one head.");
    }
    return best_scores.value();
  }

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

  if (pairwise_kind == "low_rank_bilinear") {
    if (!in_weight.has_value() || !out_weight.has_value()) {
      throw std::runtime_error("Low-rank bilinear pairwise kernel is missing projection weights.");
    }
    auto projected_target = torch::matmul(target_val, out_weight.value().transpose(0, 1));
    auto projected_source = torch::matmul(source_val, in_weight.value().transpose(0, 1));
    auto weighted_source =
        projected_source * weight.view({1, 1, -1}).to(projected_source.scalar_type());
    auto scores = torch::bmm(projected_target, weighted_source.transpose(1, 2));
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
  if (starts_with(route_kind, "multihead_max_")) {
    const auto base_kind = route_kind.substr(std::string("multihead_max_").size());
    if (base_kind == "low_rank_bilinear_route") {
      auto cast_source = source_weight.value().to(src_val.scalar_type());
      auto cast_target = target_weight.value().to(dst_val.scalar_type());
      auto cast_core = core_weight.to(src_val.scalar_type());
      auto projected_source = torch::einsum(
          "bid,hrd->bhir",
          std::vector<torch::Tensor>{src_val, cast_source});
      auto projected_target = torch::einsum(
          "bkd,hrd->bhkr",
          std::vector<torch::Tensor>{dst_val, cast_target});
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[projected_source.dim() - 3] = cast_core.size(0);
      core_shape[projected_source.dim() - 1] = cast_core.size(1);
      auto weighted_source = projected_source * cast_core.view(core_shape);
      auto scores = torch::einsum(
          "bhir,bhkr->bhik",
          std::vector<torch::Tensor>{weighted_source, projected_target});
      if (bias.has_value()) {
        std::vector<int64_t> bias_shape(scores.dim(), 1);
        bias_shape[scores.dim() - 3] = bias.value().size(0);
        scores = scores + bias.value().to(scores.scalar_type()).view(bias_shape);
      }
      if (temperature != 1.0) {
        scores = scores / temperature;
      }
      return std::get<0>(scores.max(1));
    }
    const auto num_heads = infer_multihead_count(core_weight, source_weight, target_weight);
    c10::optional<torch::Tensor> best_scores = c10::nullopt;
    for (int64_t head_index = 0; head_index < num_heads; ++head_index) {
      auto head_scores = pairwise_route_block_logits(
          base_kind,
          src_val,
          dst_val,
          select_head_optional_tensor(source_weight, head_index, num_heads),
          select_head_optional_tensor(source_bias, head_index, num_heads),
          select_head_optional_tensor(target_weight, head_index, num_heads),
          select_head_optional_tensor(target_bias, head_index, num_heads),
          select_head_tensor(core_weight, head_index, num_heads),
          select_head_optional_tensor(bias, head_index, num_heads),
          select_head_optional_tensor(hidden_weight, head_index, num_heads),
          select_head_optional_tensor(hidden_bias, head_index, num_heads),
          select_head_optional_tensor(out_weight, head_index, num_heads),
          select_head_optional_tensor(out_bias, head_index, num_heads),
          temperature);
      if (best_scores.has_value()) {
        best_scores = torch::maximum(best_scores.value(), head_scores);
      } else {
        best_scores = head_scores;
      }
    }
    if (!best_scores.has_value()) {
      throw std::runtime_error("multihead_max route kernel requires at least one head.");
    }
    return best_scores.value();
  }

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

struct NativeLayerState {
  torch::Tensor state;
  torch::Tensor val;
};

c10::optional<torch::Tensor> packed_optional_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return c10::nullopt;
  }
  return tensor;
}

torch::Tensor cast_tensor_like(const torch::Tensor& tensor, const torch::Tensor& reference) {
  if (tensor.scalar_type() == reference.scalar_type()) {
    return tensor;
  }
  return tensor.to(reference.scalar_type());
}

c10::optional<torch::Tensor> cast_optional_like(
    const c10::optional<torch::Tensor>& tensor,
    const torch::Tensor& reference) {
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return cast_tensor_like(tensor.value(), reference);
}

c10::optional<torch::Tensor> cast_packed_optional_like(
    const torch::Tensor& tensor,
    const torch::Tensor& reference) {
  auto optional = packed_optional_tensor(tensor);
  if (!optional.has_value()) {
    return c10::nullopt;
  }
  return cast_tensor_like(optional.value(), reference);
}

torch::Tensor linear2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  auto output = torch::matmul(input, cast_tensor_like(weight, input).transpose(0, 1));
  if (bias.has_value()) {
    output = output + cast_tensor_like(bias.value(), input);
  }
  return output;
}

torch::Tensor layer_norm_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  return torch::layer_norm(
      input,
      {input.size(-1)},
      cast_tensor_like(weight, input),
      cast_optional_like(bias, input),
      1e-5,
      false);
}

torch::Tensor signed_softmax_state(const torch::Tensor& state) {
  auto clean_state = torch::nan_to_num(state);
  auto magnitude = torch::softmax(clean_state.abs(), -1);
  const auto state_mass = static_cast<double>(state.size(-1));
  return torch::sign(clean_state) * magnitude * state_mass;
}

torch::Tensor signed_abs_softmax_scores(const torch::Tensor& scores) {
  auto clean_scores = torch::nan_to_num(scores);
  return torch::nan_to_num(torch::sign(clean_scores) * torch::softmax(clean_scores.abs(), -1));
}

torch::Tensor entmax15_scores(
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

torch::Tensor signed_entmax15_scores(
    const torch::Tensor& scores,
    const c10::optional<torch::Tensor>& mask = c10::nullopt) {
  auto clean_scores = torch::nan_to_num(scores);
  auto probs = entmax15_scores(clean_scores.abs(), mask);
  auto signed_routes = torch::sign(clean_scores) * probs;
  if (mask.has_value()) {
    signed_routes = signed_routes * mask.value().to(signed_routes.scalar_type());
  }
  return torch::nan_to_num(signed_routes);
}

torch::Tensor signed_entmax15_backward_scores(
    const torch::Tensor& scores,
    const torch::Tensor& routes,
    const torch::Tensor& grad_routes,
    const c10::optional<torch::Tensor>& mask = c10::nullopt) {
  auto clean_scores = torch::nan_to_num(scores);
  auto signs = torch::sign(clean_scores);
  auto grad_probs = grad_routes * signs;
  auto probs = routes.abs();
  auto gppr = torch::sqrt(torch::clamp(probs, 0));
  auto grad_input = grad_probs * gppr;
  auto gppr_sum = gppr.sum(-1, true);
  auto correction = torch::where(
      gppr_sum > 0,
      grad_input.sum(-1, true) / gppr_sum.clamp_min(std::numeric_limits<double>::epsilon()),
      torch::zeros_like(gppr_sum));
  auto grad_scores = (grad_input - correction * gppr) * signs;
  if (mask.has_value()) {
    grad_scores = grad_scores * mask.value().to(grad_scores.scalar_type());
  }
  return torch::nan_to_num(grad_scores);
}

torch::Tensor compress_scores(
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
    auto routes = signed_abs_softmax_scores(scores);
    if (mask.has_value()) {
      routes = routes * mask.value().to(routes.scalar_type());
    }
    return routes;
  }
  if (compress_name == "softsign") {
    auto edges = softsign(scores);
    if (mask.has_value()) {
      edges = edges * mask.value().to(edges.scalar_type());
    }
    return edges;
  }
  if (compress_name == "signed_entmax15") {
    return signed_entmax15_scores(scores, mask);
  }
  throw std::runtime_error("Unsupported compress_name: " + compress_name);
}

NativeLayerState layer_with_val_norm(
    const NativeLayerState& layer,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  return {layer.state, layer_norm_last_dim(layer.val, weight, bias)};
}

NativeLayerState apply_delta_to_layer(
    const NativeLayerState& layer,
    const torch::Tensor& delta_state,
    const torch::Tensor& delta_val,
    const torch::Tensor& val_norm_weight,
    const torch::Tensor& val_norm_bias) {
  auto updated_state = signed_softmax_state(layer.state + delta_state);
  auto updated_val = layer_norm_last_dim(layer.val + delta_val, val_norm_weight, val_norm_bias);
  return {updated_state, updated_val};
}

torch::Tensor full_topk_indices(const torch::Tensor& scores) {
  const auto batch = scores.size(0);
  const auto left = scores.size(1);
  const auto right = scores.size(2);
  return torch::arange(right, scores.options().dtype(torch::kLong))
      .view({1, 1, right})
      .expand({batch, left, right});
}

std::tuple<torch::Tensor, torch::Tensor> low_rank_transition_pairwise_topk_signed_abs(
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
    const std::string& compress_name) {
  const bool diagonal =
      (!source_weight.defined() || source_weight.numel() == 0) &&
      (!target_weight.defined() || target_weight.numel() == 0);
#ifdef WITH_CUDA
  if (src_val.is_cuda() &&
      sender_strength.is_cuda() &&
      projected_state.is_cuda() &&
      projected_val.is_cuda() &&
      !diagonal &&
      compress_name == "signed_abs_softmax" &&
      !c10::GradMode::is_enabled() &&
      experimental_cuda_scan_fastpath_enabled() &&
      jakal_net_low_rank_pairwise_topk_forward_cuda_available() &&
      topk > 0 && topk <= 32) {
    const bool multihead = source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2;
    torch::Tensor weighted_projected_source;
    torch::Tensor projected_target;
    if (multihead) {
      auto cast_source = cast_tensor_like(source_weight, src_val);
      auto cast_target = cast_tensor_like(target_weight, dst_val);
      auto cast_core = cast_tensor_like(core_weight, src_val);
      auto projected_source = torch::einsum("bid,hrd->bhir", std::vector<torch::Tensor>{src_val, cast_source}).contiguous();
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[1] = cast_core.size(0);
      core_shape[3] = cast_core.size(1);
      weighted_projected_source = (projected_source * cast_core.view(core_shape)).contiguous();
      projected_target = torch::einsum("bkd,hrd->bhkr", std::vector<torch::Tensor>{dst_val, cast_target}).contiguous();
      if (bias.defined() && bias.numel() != 0) {
        auto cast_bias = cast_tensor_like(bias, src_val).reshape({1, bias.numel(), 1, 1});
        auto bias_column = cast_bias.expand({weighted_projected_source.size(0), weighted_projected_source.size(1), weighted_projected_source.size(2), 1});
        auto target_ones = torch::ones(
            {projected_target.size(0), projected_target.size(1), projected_target.size(2), 1},
            projected_target.options());
        weighted_projected_source = torch::cat({weighted_projected_source, bias_column}, -1).contiguous();
        projected_target = torch::cat({projected_target, target_ones}, -1).contiguous();
      }
    } else {
      auto projected_source = torch::matmul(src_val, cast_tensor_like(source_weight, src_val).transpose(0, 1)).contiguous();
      weighted_projected_source = projected_source * cast_tensor_like(core_weight, projected_source).view({1, 1, -1});
      projected_target = torch::matmul(dst_val, cast_tensor_like(target_weight, dst_val).transpose(0, 1)).contiguous();
    }
    auto weighted_projected_state =
        (sender_strength.to(torch::kFloat32) * projected_state.to(torch::kFloat32)).contiguous();
    auto weighted_projected_val =
        (sender_strength.to(torch::kFloat32).unsqueeze(-1) * projected_val.to(torch::kFloat32)).contiguous();
    auto fused = low_rank_pairwise_topk_forward_cuda_wrapper(
        weighted_projected_source,
        projected_target,
        weighted_projected_state,
        weighted_projected_val,
        topk,
        (!multihead && bias.defined() && bias.numel() != 0) ? bias.item<double>() : 0.0,
        1);
    return {
        std::get<0>(fused).to(projected_state.scalar_type()),
        std::get<1>(fused).to(projected_val.scalar_type()),
    };
  }
#endif
  const bool multihead =
      diagonal ? core_weight.dim() >= 2
               : (source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2);
  auto cast_source_weight = diagonal ? src_val.new_empty({0}) : cast_tensor_like(source_weight, src_val);
  auto cast_target_weight = diagonal ? src_val.new_empty({0}) : cast_tensor_like(target_weight, src_val);
  auto cast_core_weight = cast_tensor_like(core_weight, src_val);
  auto cast_bias = cast_packed_optional_like(bias, src_val);
  auto logits = pairwise_route_block_logits(
      diagonal
          ? (multihead ? "multihead_max_diagonal_bilinear_route" : "diagonal_bilinear_route")
          : (multihead ? "multihead_max_low_rank_bilinear_route" : "low_rank_bilinear_route"),
      src_val,
      dst_val,
      diagonal ? c10::nullopt : c10::optional<torch::Tensor>(cast_source_weight),
      c10::nullopt,
      diagonal ? c10::nullopt : c10::optional<torch::Tensor>(cast_target_weight),
      c10::nullopt,
      cast_core_weight,
      cast_bias,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      1.0);
  const auto dst_nodes = dst_val.size(1);
  const auto k = std::min<int64_t>(std::max<int64_t>(1, topk), dst_nodes);

  torch::Tensor selected_scores;
  torch::Tensor selected_indices;
  if (k == dst_nodes) {
    selected_scores = logits;
    selected_indices = full_topk_indices(logits);
  } else {
    auto topk_result = logits.topk(k, -1, true, true);
    selected_scores = std::get<0>(topk_result);
    selected_indices = std::get<1>(topk_result);
  }

  auto routes = compress_scores(compress_name, selected_scores);
  auto weighted_routes = routes * sender_strength.unsqueeze(-1);
  const auto state_acc_dtype = accumulator_dtype(projected_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(projected_val.scalar_type());
  auto delta_state = torch::zeros(
      {projected_state.size(0), dst_nodes},
      projected_state.options().dtype(state_acc_dtype));
  auto delta_val = torch::zeros(
      {projected_val.size(0), dst_nodes, projected_val.size(2)},
      projected_val.options().dtype(val_acc_dtype));

  auto flat_indices = selected_indices.reshape({selected_indices.size(0), -1});
  auto state_contrib =
      (weighted_routes.to(state_acc_dtype) * projected_state.to(state_acc_dtype).unsqueeze(-1))
          .reshape({projected_state.size(0), -1});
  delta_state.scatter_add_(1, flat_indices, state_contrib);

  auto val_contrib =
      (weighted_routes.to(val_acc_dtype).unsqueeze(-1) *
       projected_val.to(val_acc_dtype).unsqueeze(-2))
          .reshape({projected_val.size(0), -1, projected_val.size(2)});
  auto scatter_index =
      flat_indices.unsqueeze(-1).expand({flat_indices.size(0), flat_indices.size(1), projected_val.size(2)});
  delta_val.scatter_add_(1, scatter_index, val_contrib);

  return {
      delta_state.to(projected_state.scalar_type()),
      delta_val.to(projected_val.scalar_type()),
  };
}

std::tuple<torch::Tensor, torch::Tensor> low_rank_propagation_topk_signed_abs(
    const torch::Tensor& layer_state,
    const torch::Tensor& layer_val,
    const torch::Tensor& source_weight,
    const torch::Tensor& target_weight,
    const torch::Tensor& core_weight,
    const torch::Tensor& bias,
    int64_t topk,
    const std::string& compress_name) {
  const bool diagonal =
      (!source_weight.defined() || source_weight.numel() == 0) &&
      (!target_weight.defined() || target_weight.numel() == 0);
#ifdef WITH_CUDA
  if (layer_val.is_cuda() &&
      layer_state.is_cuda() &&
      !diagonal &&
      compress_name == "signed_abs_softmax" &&
      !c10::GradMode::is_enabled() &&
      experimental_cuda_scan_fastpath_enabled() &&
      jakal_net_low_rank_propagation_topk_forward_cuda_available() &&
      topk > 0 && topk <= 32) {
    const bool multihead = source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2;
    torch::Tensor weighted_projected_source;
    torch::Tensor projected_target;
    if (multihead) {
      auto cast_source = cast_tensor_like(source_weight, layer_val);
      auto cast_target = cast_tensor_like(target_weight, layer_val);
      auto cast_core = cast_tensor_like(core_weight, layer_val);
      auto projected_source = torch::einsum("bid,hrd->bhir", std::vector<torch::Tensor>{layer_val, cast_source}).contiguous();
      std::vector<int64_t> core_shape(projected_source.dim(), 1);
      core_shape[1] = cast_core.size(0);
      core_shape[3] = cast_core.size(1);
      weighted_projected_source = (projected_source * cast_core.view(core_shape)).contiguous();
      projected_target = torch::einsum("bid,hrd->bhir", std::vector<torch::Tensor>{layer_val, cast_target}).contiguous();
      auto maybe_bias = packed_optional_tensor(bias);
      if (maybe_bias.has_value() && maybe_bias.value().numel() != 0) {
        auto cast_bias = cast_tensor_like(maybe_bias.value(), layer_val).reshape({1, maybe_bias.value().numel(), 1, 1});
        auto bias_column = cast_bias.expand({weighted_projected_source.size(0), weighted_projected_source.size(1), weighted_projected_source.size(2), 1});
        auto target_ones = torch::ones(
            {projected_target.size(0), projected_target.size(1), projected_target.size(2), 1},
            projected_target.options());
        weighted_projected_source = torch::cat({weighted_projected_source, bias_column}, -1).contiguous();
        projected_target = torch::cat({projected_target, target_ones}, -1).contiguous();
      }
    } else {
      auto projected_target_single = torch::matmul(layer_val, cast_tensor_like(target_weight, layer_val).transpose(0, 1)).contiguous();
      auto projected_source_single = torch::matmul(layer_val, cast_tensor_like(source_weight, layer_val).transpose(0, 1)).contiguous();
      weighted_projected_source = projected_source_single * cast_tensor_like(core_weight, projected_source_single).view({1, 1, -1});
      projected_target = projected_target_single;
    }
    auto weighted_projected_state = (layer_state.to(torch::kFloat32) * layer_state.to(torch::kFloat32)).contiguous();
    auto weighted_projected_val =
        (layer_state.to(torch::kFloat32).unsqueeze(-1) * layer_val.to(torch::kFloat32)).contiguous();
    const auto score_bias =
        (!multihead && packed_optional_tensor(bias).has_value())
            ? packed_optional_tensor(bias).value().item<double>()
            : 0.0;
    auto fused = low_rank_propagation_topk_forward_cuda_wrapper(
        weighted_projected_source,
        projected_target,
        weighted_projected_state,
        weighted_projected_val,
        topk,
        score_bias,
        true);
    return {
        std::get<0>(fused).to(layer_state.scalar_type()),
        std::get<1>(fused).to(layer_val.scalar_type()),
    };
  }
#endif
  const bool multihead =
      diagonal ? core_weight.dim() >= 2
               : (source_weight.dim() >= 3 || target_weight.dim() >= 3 || core_weight.dim() >= 2);
  auto cast_source_weight = diagonal ? layer_val.new_empty({0}) : cast_tensor_like(source_weight, layer_val);
  auto cast_target_weight = diagonal ? layer_val.new_empty({0}) : cast_tensor_like(target_weight, layer_val);
  auto cast_core_weight = cast_tensor_like(core_weight, layer_val);
  auto cast_bias = cast_packed_optional_like(bias, layer_val);
  auto scores = pairwise_scores(
      diagonal
          ? (multihead ? "multihead_max_diagonal_bilinear" : "diagonal_bilinear")
          : (multihead ? "multihead_max_low_rank_bilinear" : "low_rank_bilinear"),
      layer_val,
      layer_val,
      cast_core_weight,
      cast_bias,
      diagonal ? c10::nullopt : c10::optional<torch::Tensor>(cast_source_weight),
      c10::nullopt,
      diagonal ? c10::nullopt : c10::optional<torch::Tensor>(cast_target_weight),
      c10::nullopt);
  const auto nodes = layer_val.size(1);
  const auto k = std::min<int64_t>(std::max<int64_t>(1, topk), nodes);

  torch::Tensor selected_scores;
  torch::Tensor selected_indices;
  if (k == nodes) {
    selected_scores = scores;
    selected_indices = full_topk_indices(scores);
  } else {
    auto topk_result = scores.topk(k, -1, true, true);
    selected_scores = std::get<0>(topk_result);
    selected_indices = std::get<1>(topk_result);
  }

  auto gathered = gather_by_indices(layer_state, layer_val, selected_indices);
  auto selected_state = std::get<0>(gathered);
  auto selected_val = std::get<1>(gathered);
  auto edges = compress_scores(compress_name, selected_scores);
  auto weighted_edges = edges * selected_state;
  const auto state_acc_dtype = accumulator_dtype(layer_state.scalar_type());
  const auto val_acc_dtype = accumulator_dtype(layer_val.scalar_type());
  auto delta_state =
      (weighted_edges.to(state_acc_dtype) * selected_state.to(state_acc_dtype)).sum(-1);
  auto delta_val =
      (weighted_edges.to(val_acc_dtype).unsqueeze(-1) * selected_val.to(val_acc_dtype)).sum(-2);
  return {
      delta_state.to(layer_state.scalar_type()),
      delta_val.to(layer_val.scalar_type()),
  };
}

torch::Tensor read_memory_vector(
    const std::vector<NativeLayerState>& memory_state,
    const std::vector<torch::Tensor>& val_norm_weights,
    const std::vector<torch::Tensor>& val_norm_biases,
    const torch::Tensor& read_template_val,
    const std::vector<torch::Tensor>& read_projection_weights,
    const std::vector<torch::Tensor>& read_gates) {
  std::vector<torch::Tensor> read_terms;
  read_terms.reserve(memory_state.size());
  for (size_t index = 0; index < memory_state.size(); ++index) {
    auto sender_strength = torch::softplus(memory_state[index].state).unsqueeze(-1);
    auto read_summary = (sender_strength * memory_state[index].val).sum(1);
    read_summary = read_summary + cast_tensor_like(read_template_val, read_summary).unsqueeze(0);
    auto projected = linear2d(read_summary, read_projection_weights[index], c10::nullopt);
    auto gate = torch::sigmoid(cast_tensor_like(read_gates[index], read_summary));
    read_terms.push_back(gate * projected);
  }
  return torch::stack(read_terms, 0).sum(0);
}

void require_vector_size(
    const std::vector<torch::Tensor>& tensors,
    size_t expected,
    const std::string& name) {
  if (tensors.size() != expected) {
    throw std::runtime_error(
        name + " has unexpected length: got " + std::to_string(tensors.size()) +
        ", expected " + std::to_string(expected) + ".");
  }
}

void require_int_vector_size(
    const std::vector<int64_t>& values,
    size_t expected,
    const std::string& name) {
  if (values.size() != expected) {
    throw std::runtime_error(
        name + " has unexpected length: got " + std::to_string(values.size()) +
        ", expected " + std::to_string(expected) + ".");
  }
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> causal_memory_scan_fused(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const c10::optional<torch::Tensor>& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const c10::optional<torch::Tensor>& prediction_input_norm_bias,
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
  require_supported_device(aligned_s, "aligned_s");
#ifdef WITH_CUDA
  if (aligned_s.is_cuda()) {
    auto packed_value_to_state_bias =
        value_to_state_bias.has_value() ? value_to_state_bias.value() : aligned_s.new_empty({0});
    auto packed_prediction_input_norm_bias =
        prediction_input_norm_bias.has_value()
            ? prediction_input_norm_bias.value()
            : aligned_s.new_empty({0});
    return jakal_net_causal_memory_scan_fused_forward_cuda(
        aligned_s,
        flat_memory,
        value_to_state_weight,
        packed_value_to_state_bias,
        s_prediction_weight,
        prediction_input_norm_weight,
        packed_prediction_input_norm_bias,
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
        skip_topks);
  }
#endif
  if (aligned_s.dim() != 3) {
    throw std::runtime_error("aligned_s must have shape [batch, seq_len, dim].");
  }
  if (flat_memory.size() % 2 != 0) {
    throw std::runtime_error("flat_memory must contain alternating state/val tensors.");
  }
  const auto num_levels = flat_memory.size() / 2;
  const auto expected_skip_count = num_levels > 1 ? num_levels - 1 : 0;
  require_vector_size(read_projection_weights, num_levels, "read_projection_weights");
  require_vector_size(read_gates, num_levels, "read_gates");
  require_vector_size(write_source_weights, num_levels, "write_source_weights");
  require_vector_size(write_target_weights, num_levels, "write_target_weights");
  require_vector_size(write_core_weights, num_levels, "write_core_weights");
  require_vector_size(write_biases, num_levels, "write_biases");
  require_int_vector_size(write_topks, num_levels, "write_topks");
  require_vector_size(propagation_source_weights, num_levels, "propagation_source_weights");
  require_vector_size(propagation_target_weights, num_levels, "propagation_target_weights");
  require_vector_size(propagation_core_weights, num_levels, "propagation_core_weights");
  require_vector_size(propagation_biases, num_levels, "propagation_biases");
  require_int_vector_size(propagation_topks, num_levels, "propagation_topks");
  require_vector_size(val_norm_weights, num_levels, "val_norm_weights");
  require_vector_size(val_norm_biases, num_levels, "val_norm_biases");
  require_vector_size(level_transition_source_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_source_weights");
  require_vector_size(level_transition_target_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_target_weights");
  require_vector_size(level_transition_core_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_core_weights");
  require_vector_size(level_transition_biases, num_levels > 0 ? num_levels - 1 : 0, "level_transition_biases");
  require_int_vector_size(level_transition_topks, num_levels > 0 ? num_levels - 1 : 0, "level_transition_topks");
  require_vector_size(level_norm_weights, num_levels, "level_norm_weights");
  require_vector_size(level_norm_biases, num_levels, "level_norm_biases");
  require_vector_size(skip_source_weights, expected_skip_count, "skip_source_weights");
  require_vector_size(skip_target_weights, expected_skip_count, "skip_target_weights");
  require_vector_size(skip_core_weights, expected_skip_count, "skip_core_weights");
  require_vector_size(skip_biases, expected_skip_count, "skip_biases");
  require_vector_size(skip_gates, expected_skip_count, "skip_gates");
  require_int_vector_size(skip_topks, expected_skip_count, "skip_topks");

  std::vector<NativeLayerState> current_memory;
  current_memory.reserve(num_levels);
  for (size_t index = 0; index < num_levels; ++index) {
    current_memory.push_back({flat_memory[index * 2], flat_memory[index * 2 + 1]});
  }

  auto projected_s = linear3d(aligned_s, s_prediction_weight, c10::nullopt);
  std::vector<torch::Tensor> query_steps;
  query_steps.reserve(aligned_s.size(1));

  for (int64_t time_index = 0; time_index < aligned_s.size(1); ++time_index) {
    auto token_val = aligned_s.slice(1, time_index, time_index + 1);
    auto token_state = linear3d(token_val, value_to_state_weight, cast_optional_like(value_to_state_bias, token_val)).squeeze(-1);
    NativeLayerState token_layer{token_state, token_val};

    std::vector<NativeLayerState> next_memory;
    next_memory.reserve(num_levels);

    auto first_write_delta = low_rank_transition_pairwise_topk_signed_abs(
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
        transition_compress_name);
    auto level = apply_delta_to_layer(
        current_memory[0],
        std::get<0>(first_write_delta),
        std::get<1>(first_write_delta),
        val_norm_weights[0],
        val_norm_biases[0]);
    auto first_prop_delta = low_rank_propagation_topk_signed_abs(
        level.state,
        level.val,
        propagation_source_weights[0],
        propagation_target_weights[0],
        propagation_core_weights[0],
        propagation_biases[0],
        propagation_topks[0],
        propagation_compress_name);
    level = apply_delta_to_layer(
        level,
        std::get<0>(first_prop_delta),
        std::get<1>(first_prop_delta),
        val_norm_weights[0],
        val_norm_biases[0]);
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < num_levels; ++level_index) {
      auto current_level = current_memory[level_index];
      auto parent_delta = low_rank_transition_pairwise_topk_signed_abs(
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
          transition_compress_name);
      auto updated_level = apply_delta_to_layer(
          current_level,
          std::get<0>(parent_delta),
          std::get<1>(parent_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index]);

      if (level_index == 1 && expected_skip_count > 0) {
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates[0], token_val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index]);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates[skip_index], next_memory[level_index - 2].val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index]);
      }

      auto propagation_delta = low_rank_propagation_topk_signed_abs(
          updated_level.state,
          updated_level.val,
          propagation_source_weights[level_index],
          propagation_target_weights[level_index],
          propagation_core_weights[level_index],
          propagation_biases[level_index],
          propagation_topks[level_index],
          propagation_compress_name);
      updated_level = apply_delta_to_layer(
          updated_level,
          std::get<0>(propagation_delta),
          std::get<1>(propagation_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      next_memory.push_back(updated_level);
    }

    current_memory = next_memory;
    auto read_vector = read_memory_vector(
        current_memory,
        val_norm_weights,
        val_norm_biases,
        read_template_val,
        read_projection_weights,
        read_gates);
    auto query_input = projected_s.select(1, time_index) + read_vector;
    query_steps.push_back(layer_norm_last_dim(
        query_input,
        prediction_input_norm_weight,
        cast_optional_like(prediction_input_norm_bias, query_input))
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
  return {query_val, next_memory_flat};
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>> causal_memory_scan_fused_trace(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const c10::optional<torch::Tensor>& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const c10::optional<torch::Tensor>& prediction_input_norm_bias,
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
  require_supported_device(aligned_s, "aligned_s");
#ifdef WITH_CUDA
  if (aligned_s.is_cuda()) {
    auto packed_value_to_state_bias =
        value_to_state_bias.has_value() ? value_to_state_bias.value() : aligned_s.new_empty({0});
    auto packed_prediction_input_norm_bias =
        prediction_input_norm_bias.has_value()
            ? prediction_input_norm_bias.value()
            : aligned_s.new_empty({0});
    return jakal_net_causal_memory_scan_fused_trace_cuda(
        aligned_s,
        flat_memory,
        value_to_state_weight,
        packed_value_to_state_bias,
        s_prediction_weight,
        prediction_input_norm_weight,
        packed_prediction_input_norm_bias,
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
        skip_topks);
  }
#endif
  if (aligned_s.dim() != 3) {
    throw std::runtime_error("aligned_s must have shape [batch, seq_len, dim].");
  }
  if (flat_memory.size() % 2 != 0) {
    throw std::runtime_error("flat_memory must contain alternating state/val tensors.");
  }
  const auto num_levels = flat_memory.size() / 2;
  const auto expected_skip_count = num_levels > 1 ? num_levels - 1 : 0;
  require_vector_size(read_projection_weights, num_levels, "read_projection_weights");
  require_vector_size(read_gates, num_levels, "read_gates");
  require_vector_size(write_source_weights, num_levels, "write_source_weights");
  require_vector_size(write_target_weights, num_levels, "write_target_weights");
  require_vector_size(write_core_weights, num_levels, "write_core_weights");
  require_vector_size(write_biases, num_levels, "write_biases");
  require_int_vector_size(write_topks, num_levels, "write_topks");
  require_vector_size(propagation_source_weights, num_levels, "propagation_source_weights");
  require_vector_size(propagation_target_weights, num_levels, "propagation_target_weights");
  require_vector_size(propagation_core_weights, num_levels, "propagation_core_weights");
  require_vector_size(propagation_biases, num_levels, "propagation_biases");
  require_int_vector_size(propagation_topks, num_levels, "propagation_topks");
  require_vector_size(val_norm_weights, num_levels, "val_norm_weights");
  require_vector_size(val_norm_biases, num_levels, "val_norm_biases");
  require_vector_size(level_transition_source_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_source_weights");
  require_vector_size(level_transition_target_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_target_weights");
  require_vector_size(level_transition_core_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_core_weights");
  require_vector_size(level_transition_biases, num_levels > 0 ? num_levels - 1 : 0, "level_transition_biases");
  require_int_vector_size(level_transition_topks, num_levels > 0 ? num_levels - 1 : 0, "level_transition_topks");
  require_vector_size(level_norm_weights, num_levels, "level_norm_weights");
  require_vector_size(level_norm_biases, num_levels, "level_norm_biases");
  require_vector_size(skip_source_weights, expected_skip_count, "skip_source_weights");
  require_vector_size(skip_target_weights, expected_skip_count, "skip_target_weights");
  require_vector_size(skip_core_weights, expected_skip_count, "skip_core_weights");
  require_vector_size(skip_biases, expected_skip_count, "skip_biases");
  require_vector_size(skip_gates, expected_skip_count, "skip_gates");
  require_int_vector_size(skip_topks, expected_skip_count, "skip_topks");

  std::vector<NativeLayerState> current_memory;
  current_memory.reserve(num_levels);
  for (size_t index = 0; index < num_levels; ++index) {
    current_memory.push_back({flat_memory[index * 2], flat_memory[index * 2 + 1]});
  }

  std::vector<std::vector<torch::Tensor>> trace_states(num_levels);
  std::vector<std::vector<torch::Tensor>> trace_vals(num_levels);

  auto projected_s = linear3d(aligned_s, s_prediction_weight, c10::nullopt);
  std::vector<torch::Tensor> query_steps;
  query_steps.reserve(aligned_s.size(1));

  for (int64_t time_index = 0; time_index < aligned_s.size(1); ++time_index) {
    for (size_t trace_index = 0; trace_index < num_levels; ++trace_index) {
      trace_states[trace_index].push_back(current_memory[trace_index].state);
      trace_vals[trace_index].push_back(current_memory[trace_index].val);
    }
    auto token_val = aligned_s.slice(1, time_index, time_index + 1);
    auto token_state = linear3d(token_val, value_to_state_weight, cast_optional_like(value_to_state_bias, token_val)).squeeze(-1);
    NativeLayerState token_layer{token_state, token_val};

    std::vector<NativeLayerState> next_memory;
    next_memory.reserve(num_levels);

    auto first_level_normed = layer_with_val_norm(current_memory[0], val_norm_weights[0], val_norm_biases[0]);
    auto first_write_delta = low_rank_transition_pairwise_topk_signed_abs(
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
        transition_compress_name);
    auto level = apply_delta_to_layer(
        current_memory[0],
        std::get<0>(first_write_delta),
        std::get<1>(first_write_delta),
        val_norm_weights[0],
        val_norm_biases[0]);
    auto level_for_propagation = layer_with_val_norm(level, val_norm_weights[0], val_norm_biases[0]);
    auto first_prop_delta = low_rank_propagation_topk_signed_abs(
        level_for_propagation.state,
        level_for_propagation.val,
        propagation_source_weights[0],
        propagation_target_weights[0],
        propagation_core_weights[0],
        propagation_biases[0],
        propagation_topks[0],
        propagation_compress_name);
    level = apply_delta_to_layer(
        level,
        std::get<0>(first_prop_delta),
        std::get<1>(first_prop_delta),
        val_norm_weights[0],
        val_norm_biases[0]);
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < num_levels; ++level_index) {
      auto current_level = current_memory[level_index];
      auto normalized_level = layer_with_val_norm(
          current_level,
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      auto normalized_parent = layer_with_val_norm(
          next_memory[level_index - 1],
          level_norm_weights[level_index - 1],
          level_norm_biases[level_index - 1]);
      auto parent_delta = low_rank_transition_pairwise_topk_signed_abs(
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
          transition_compress_name);
      auto updated_level = apply_delta_to_layer(
          current_level,
          std::get<0>(parent_delta),
          std::get<1>(parent_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index]);

      if (level_index == 1 && expected_skip_count > 0) {
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates[0], token_val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index]);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto normalized_skip_source = layer_with_val_norm(
            next_memory[level_index - 2],
            level_norm_weights[level_index - 2],
            level_norm_biases[level_index - 2]);
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates[skip_index], normalized_skip_source.val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index]);
      }

      auto updated_level_for_prop = layer_with_val_norm(
          updated_level,
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      auto propagation_delta = low_rank_propagation_topk_signed_abs(
          updated_level_for_prop.state,
          updated_level_for_prop.val,
          propagation_source_weights[level_index],
          propagation_target_weights[level_index],
          propagation_core_weights[level_index],
          propagation_biases[level_index],
          propagation_topks[level_index],
          propagation_compress_name);
      updated_level = apply_delta_to_layer(
          updated_level,
          std::get<0>(propagation_delta),
          std::get<1>(propagation_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      next_memory.push_back(updated_level);
    }

    current_memory = next_memory;
    auto read_vector = read_memory_vector(
        current_memory,
        val_norm_weights,
        val_norm_biases,
        read_template_val,
        read_projection_weights,
        read_gates);
    auto query_input = projected_s.select(1, time_index) + read_vector;
    query_steps.push_back(layer_norm_last_dim(
        query_input,
        prediction_input_norm_weight,
        cast_optional_like(prediction_input_norm_bias, query_input))
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
  trace_flat.reserve(num_levels * 2);
  for (size_t trace_index = 0; trace_index < num_levels; ++trace_index) {
    trace_flat.push_back(torch::stack(trace_states[trace_index], 0));
    trace_flat.push_back(torch::stack(trace_vals[trace_index], 0));
  }
  return {query_val, next_memory_flat, trace_flat};
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
causal_memory_scan_fused_checkpoints(
    const torch::Tensor& aligned_s,
    const std::vector<torch::Tensor>& flat_memory,
    const torch::Tensor& value_to_state_weight,
    const c10::optional<torch::Tensor>& value_to_state_bias,
    const torch::Tensor& s_prediction_weight,
    const torch::Tensor& prediction_input_norm_weight,
    const c10::optional<torch::Tensor>& prediction_input_norm_bias,
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
    int64_t checkpoint_stride) {
  if (checkpoint_stride <= 0) {
    throw std::runtime_error("checkpoint_stride must be positive.");
  }
  require_supported_device(aligned_s, "aligned_s");
  if (aligned_s.dim() != 3) {
    throw std::runtime_error("aligned_s must have shape [batch, seq_len, dim].");
  }
  if (flat_memory.size() % 2 != 0) {
    throw std::runtime_error("flat_memory must contain alternating state/val tensors.");
  }
  const auto num_levels = flat_memory.size() / 2;
  const auto expected_skip_count = num_levels > 1 ? num_levels - 1 : 0;
  require_vector_size(read_projection_weights, num_levels, "read_projection_weights");
  require_vector_size(read_gates, num_levels, "read_gates");
  require_vector_size(write_source_weights, num_levels, "write_source_weights");
  require_vector_size(write_target_weights, num_levels, "write_target_weights");
  require_vector_size(write_core_weights, num_levels, "write_core_weights");
  require_vector_size(write_biases, num_levels, "write_biases");
  require_int_vector_size(write_topks, num_levels, "write_topks");
  require_vector_size(propagation_source_weights, num_levels, "propagation_source_weights");
  require_vector_size(propagation_target_weights, num_levels, "propagation_target_weights");
  require_vector_size(propagation_core_weights, num_levels, "propagation_core_weights");
  require_vector_size(propagation_biases, num_levels, "propagation_biases");
  require_int_vector_size(propagation_topks, num_levels, "propagation_topks");
  require_vector_size(val_norm_weights, num_levels, "val_norm_weights");
  require_vector_size(val_norm_biases, num_levels, "val_norm_biases");
  require_vector_size(level_transition_source_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_source_weights");
  require_vector_size(level_transition_target_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_target_weights");
  require_vector_size(level_transition_core_weights, num_levels > 0 ? num_levels - 1 : 0, "level_transition_core_weights");
  require_vector_size(level_transition_biases, num_levels > 0 ? num_levels - 1 : 0, "level_transition_biases");
  require_int_vector_size(level_transition_topks, num_levels > 0 ? num_levels - 1 : 0, "level_transition_topks");
  require_vector_size(level_norm_weights, num_levels, "level_norm_weights");
  require_vector_size(level_norm_biases, num_levels, "level_norm_biases");
  require_vector_size(skip_source_weights, expected_skip_count, "skip_source_weights");
  require_vector_size(skip_target_weights, expected_skip_count, "skip_target_weights");
  require_vector_size(skip_core_weights, expected_skip_count, "skip_core_weights");
  require_vector_size(skip_biases, expected_skip_count, "skip_biases");
  require_vector_size(skip_gates, expected_skip_count, "skip_gates");
  require_int_vector_size(skip_topks, expected_skip_count, "skip_topks");

  std::vector<NativeLayerState> current_memory;
  current_memory.reserve(num_levels);
  for (size_t index = 0; index < num_levels; ++index) {
    current_memory.push_back({flat_memory[index * 2], flat_memory[index * 2 + 1]});
  }

  std::vector<std::vector<torch::Tensor>> checkpoint_states(num_levels);
  std::vector<std::vector<torch::Tensor>> checkpoint_vals(num_levels);

  auto projected_s = linear3d(aligned_s, s_prediction_weight, c10::nullopt);
  std::vector<torch::Tensor> query_steps;
  query_steps.reserve(aligned_s.size(1));

  for (int64_t time_index = 0; time_index < aligned_s.size(1); ++time_index) {
    if (time_index % checkpoint_stride == 0) {
      for (size_t checkpoint_index = 0; checkpoint_index < num_levels; ++checkpoint_index) {
        checkpoint_states[checkpoint_index].push_back(current_memory[checkpoint_index].state);
        checkpoint_vals[checkpoint_index].push_back(current_memory[checkpoint_index].val);
      }
    }

    auto token_val = aligned_s.slice(1, time_index, time_index + 1);
    auto token_state = linear3d(token_val, value_to_state_weight, cast_optional_like(value_to_state_bias, token_val)).squeeze(-1);
    NativeLayerState token_layer{token_state, token_val};

    std::vector<NativeLayerState> next_memory;
    next_memory.reserve(num_levels);

    auto first_level_normed = layer_with_val_norm(current_memory[0], val_norm_weights[0], val_norm_biases[0]);
    auto first_write_delta = low_rank_transition_pairwise_topk_signed_abs(
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
        transition_compress_name);
    auto level = apply_delta_to_layer(
        current_memory[0],
        std::get<0>(first_write_delta),
        std::get<1>(first_write_delta),
        val_norm_weights[0],
        val_norm_biases[0]);
    auto level_for_propagation = layer_with_val_norm(level, val_norm_weights[0], val_norm_biases[0]);
    auto first_prop_delta = low_rank_propagation_topk_signed_abs(
        level_for_propagation.state,
        level_for_propagation.val,
        propagation_source_weights[0],
        propagation_target_weights[0],
        propagation_core_weights[0],
        propagation_biases[0],
        propagation_topks[0],
        propagation_compress_name);
    level = apply_delta_to_layer(
        level,
        std::get<0>(first_prop_delta),
        std::get<1>(first_prop_delta),
        val_norm_weights[0],
        val_norm_biases[0]);
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < num_levels; ++level_index) {
      auto current_level = current_memory[level_index];
      auto normalized_level = layer_with_val_norm(
          current_level,
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      auto normalized_parent = layer_with_val_norm(
          next_memory[level_index - 1],
          level_norm_weights[level_index - 1],
          level_norm_biases[level_index - 1]);
      auto parent_delta = low_rank_transition_pairwise_topk_signed_abs(
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
          transition_compress_name);
      auto updated_level = apply_delta_to_layer(
          current_level,
          std::get<0>(parent_delta),
          std::get<1>(parent_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index]);

      if (level_index == 1 && expected_skip_count > 0) {
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates[0], token_val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index]);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto normalized_skip_source = layer_with_val_norm(
            next_memory[level_index - 2],
            level_norm_weights[level_index - 2],
            level_norm_biases[level_index - 2]);
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates[skip_index], normalized_skip_source.val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights[level_index],
            val_norm_biases[level_index]);
      }

      auto updated_level_for_prop = layer_with_val_norm(
          updated_level,
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      auto propagation_delta = low_rank_propagation_topk_signed_abs(
          updated_level_for_prop.state,
          updated_level_for_prop.val,
          propagation_source_weights[level_index],
          propagation_target_weights[level_index],
          propagation_core_weights[level_index],
          propagation_biases[level_index],
          propagation_topks[level_index],
          propagation_compress_name);
      updated_level = apply_delta_to_layer(
          updated_level,
          std::get<0>(propagation_delta),
          std::get<1>(propagation_delta),
          val_norm_weights[level_index],
          val_norm_biases[level_index]);
      next_memory.push_back(updated_level);
    }

    current_memory = next_memory;
    auto read_vector = read_memory_vector(
        current_memory,
        val_norm_weights,
        val_norm_biases,
        read_template_val,
        read_projection_weights,
        read_gates);
    auto query_input = projected_s.select(1, time_index) + read_vector;
    query_steps.push_back(layer_norm_last_dim(
        query_input,
        prediction_input_norm_weight,
        cast_optional_like(prediction_input_norm_bias, query_input))
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

  std::vector<torch::Tensor> checkpoint_flat;
  checkpoint_flat.reserve(num_levels * 2);
  for (size_t checkpoint_index = 0; checkpoint_index < num_levels; ++checkpoint_index) {
    if (checkpoint_states[checkpoint_index].empty()) {
      checkpoint_flat.push_back(current_memory[checkpoint_index].state.new_empty(
          {0, current_memory[checkpoint_index].state.size(0), current_memory[checkpoint_index].state.size(1)}));
      checkpoint_flat.push_back(current_memory[checkpoint_index].val.new_empty(
          {0, current_memory[checkpoint_index].val.size(0), current_memory[checkpoint_index].val.size(1), current_memory[checkpoint_index].val.size(2)}));
      continue;
    }
    checkpoint_flat.push_back(torch::stack(checkpoint_states[checkpoint_index], 0));
    checkpoint_flat.push_back(torch::stack(checkpoint_vals[checkpoint_index], 0));
  }
  return {query_val, next_memory_flat, checkpoint_flat};
}



std::vector<torch::Tensor> causal_memory_scan_fused_backward_cuda(
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
#ifdef WITH_CUDA
  if (!aligned_s.is_cuda()) {
    throw std::runtime_error("causal_memory_scan_fused_backward_cuda requires CUDA inputs.");
  }
  pybind11::gil_scoped_release no_gil;
  return jakal_net_causal_memory_scan_fused_backward_cuda(
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
      trace_tensors,
      grad_query_val,
      grad_next_memory);
  if (grad_next_memory.size() != flat_memory.size()) {
    throw std::runtime_error("grad_next_memory must match flat_memory length.");
  }
  c10::AutoGradMode enable_grad(true);

  auto trace_result = causal_memory_scan_fused_trace(
      aligned_s,
      flat_memory,
      value_to_state_weight,
      packed_optional_tensor(value_to_state_bias),
      s_prediction_weight,
      prediction_input_norm_weight,
      packed_optional_tensor(prediction_input_norm_bias),
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
      skip_topks);
  auto recomputed_trace_tensors = std::get<2>(trace_result);

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
  shared_grad_accum.reserve(
      8 + read_projection_weights.size() + read_gates.size() + write_source_weights.size() +
      write_target_weights.size() + write_core_weights.size() + write_biases.size() +
      propagation_source_weights.size() + propagation_target_weights.size() +
      propagation_core_weights.size() + propagation_biases.size() + val_norm_weights.size() +
      val_norm_biases.size() + level_transition_source_weights.size() +
      level_transition_target_weights.size() + level_transition_core_weights.size() +
      level_transition_biases.size() + level_norm_weights.size() + level_norm_biases.size() +
      skip_source_weights.size() + skip_target_weights.size() + skip_core_weights.size() +
      skip_biases.size() + skip_gates.size());
  auto aligned_s_grad = aligned_s.requires_grad() ? torch::zeros_like(aligned_s) : torch::Tensor();

  for (int64_t time_index = aligned_s.size(1) - 1; time_index >= 0; --time_index) {
    auto token_val_leaf = make_leaf(aligned_s.slice(1, time_index, time_index + 1));
    std::vector<NativeLayerState> current_memory;
    std::vector<torch::Tensor> current_memory_leaves;
    current_memory.reserve(flat_memory.size() / 2);
    current_memory_leaves.reserve(flat_memory.size());
    for (size_t level_index = 0; level_index < flat_memory.size() / 2; ++level_index) {
      auto state_leaf = recomputed_trace_tensors[level_index * 2].select(0, time_index).detach();
      state_leaf.set_requires_grad(flat_memory[level_index * 2].requires_grad());
      auto val_leaf = recomputed_trace_tensors[level_index * 2 + 1].select(0, time_index).detach();
      val_leaf.set_requires_grad(flat_memory[level_index * 2 + 1].requires_grad());
      current_memory.push_back({state_leaf, val_leaf});
      current_memory_leaves.push_back(state_leaf);
      current_memory_leaves.push_back(val_leaf);
    }

    auto token_state = linear3d(
        token_val_leaf,
        value_to_state_weight_leaf,
        cast_packed_optional_like(value_to_state_bias_leaf, token_val_leaf)).squeeze(-1);
    auto projected_s_t = linear3d(token_val_leaf, s_prediction_weight_leaf, c10::nullopt).squeeze(1);

    std::vector<NativeLayerState> next_memory;
    next_memory.reserve(current_memory.size());
    auto first_write_delta = low_rank_transition_pairwise_topk_signed_abs(
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
        transition_compress_name);
    auto level = apply_delta_to_layer(
        current_memory[0],
        std::get<0>(first_write_delta),
        std::get<1>(first_write_delta),
        val_norm_weights_leaves[0],
        val_norm_biases_leaves[0]);
    auto first_prop_delta = low_rank_propagation_topk_signed_abs(
        level.state,
        level.val,
        propagation_source_weights_leaves[0],
        propagation_target_weights_leaves[0],
        propagation_core_weights_leaves[0],
        propagation_biases_leaves[0],
        propagation_topks[0],
        propagation_compress_name);
    level = apply_delta_to_layer(
        level,
        std::get<0>(first_prop_delta),
        std::get<1>(first_prop_delta),
        val_norm_weights_leaves[0],
        val_norm_biases_leaves[0]);
    next_memory.push_back(level);

    for (size_t level_index = 1; level_index < current_memory.size(); ++level_index) {
      auto current_level = current_memory[level_index];
      auto parent_delta = low_rank_transition_pairwise_topk_signed_abs(
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
          transition_compress_name);
      auto updated_level = apply_delta_to_layer(
          current_level,
          std::get<0>(parent_delta),
          std::get<1>(parent_delta),
          val_norm_weights_leaves[level_index],
          val_norm_biases_leaves[level_index]);

      if (level_index == 1 && !skip_source_weights_leaves.empty()) {
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates_leaves[0], token_val_leaf));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights_leaves[level_index],
            val_norm_biases_leaves[level_index]);
      }

      if (level_index >= 2) {
        const auto skip_index = level_index - 1;
        auto skip_gate = torch::sigmoid(cast_tensor_like(skip_gates_leaves[skip_index], next_memory[level_index - 2].val));
        auto skip_delta = low_rank_transition_pairwise_topk_signed_abs(
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
            transition_compress_name);
        updated_level = apply_delta_to_layer(
            updated_level,
            std::get<0>(skip_delta) * skip_gate,
            std::get<1>(skip_delta) * skip_gate,
            val_norm_weights_leaves[level_index],
            val_norm_biases_leaves[level_index]);
      }

      auto propagation_delta = low_rank_propagation_topk_signed_abs(
          updated_level.state,
          updated_level.val,
          propagation_source_weights_leaves[level_index],
          propagation_target_weights_leaves[level_index],
          propagation_core_weights_leaves[level_index],
          propagation_biases_leaves[level_index],
          propagation_topks[level_index],
          propagation_compress_name);
      updated_level = apply_delta_to_layer(
          updated_level,
          std::get<0>(propagation_delta),
          std::get<1>(propagation_delta),
          val_norm_weights_leaves[level_index],
          val_norm_biases_leaves[level_index]);
      next_memory.push_back(updated_level);
    }

    auto read_vector = read_memory_vector(
        next_memory,
        val_norm_weights_leaves,
        val_norm_biases_leaves,
        read_template_val_leaf,
        read_projection_weights_leaves,
        read_gates_leaves);
    auto query_input = projected_s_t + read_vector;
    auto query_step = layer_norm_last_dim(
        query_input,
        prediction_input_norm_weight_leaf,
        cast_packed_optional_like(prediction_input_norm_bias_leaf, query_input));

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
    pybind11::gil_scoped_release no_gil;
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
#else
  throw std::runtime_error(
      "causal_memory_scan_fused_backward_cuda requires a CUDA-enabled build.");
#endif
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
      "causal_memory_scan_fused",
      "causal_memory_scan_fused_trace",
      "causal_memory_scan_fused_checkpoints",
  };
  if (jakal_net_compiled_with_cuda_source()) {
    ops.insert(
        ops.end(),
        {
            "query_topk_reduce_cuda",
            "query_topk_reduce_backward_cuda",
            "low_rank_pairwise_topk_forward_cuda",
            "low_rank_propagation_topk_forward_cuda",
            "low_rank_propagation_window_forward_cuda",
            "low_rank_propagation_window_entmax15_forward_cuda",
            "softsign_backward_cuda",
            "softmax_backward_cuda",
            "diagonal_pairwise_topk_backward_cuda",
            "low_rank_pairwise_topk_backward_cuda",
            "causal_memory_scan_fused_backward_cuda",
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
    auto edges = compress_scores(edge_compress_name, scores);
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
      auto edges = compress_scores(edge_compress_name, scores);
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
    auto edges = compress_scores(edge_compress_name, scores);
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
      auto edges = compress_scores(edge_compress_name, scores);
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

#ifdef WITH_CUDA
  if (pairwise_kind == "low_rank_bilinear" &&
      edge_compress_name == "softsign" &&
      flat_val.is_cuda() &&
      flat_projected_state.is_cuda() &&
      flat_projected_val.is_cuda() &&
      in_weight.has_value() &&
      out_weight.has_value() &&
      jakal_net_low_rank_propagation_window_forward_cuda_available() &&
      !flat_val.requires_grad() &&
      !flat_projected_state.requires_grad() &&
      !flat_projected_val.requires_grad() &&
      !in_weight.value().requires_grad() &&
      !out_weight.value().requires_grad() &&
      !weight.requires_grad()) {
    auto projected_target = torch::matmul(flat_val, out_weight.value().transpose(0, 1)).contiguous();
    auto projected_source = torch::matmul(flat_val, in_weight.value().transpose(0, 1)).contiguous();
    auto weighted_projected_source =
        projected_source * weight.view({1, 1, -1}).to(projected_source.scalar_type());
    const auto score_bias = bias.has_value() ? bias.value().item<double>() : 0.0;
    auto fused = low_rank_propagation_window_forward_cuda_wrapper(
        weighted_projected_source,
        projected_target,
        flat_projected_state.to(torch::kFloat32).contiguous(),
        flat_projected_val.to(torch::kFloat32).contiguous(),
        window,
        score_bias);
    return {
        reshape_state(std::get<0>(fused).to(projected_state.scalar_type()), batch_sizes, num_nodes),
        reshape_val(std::get<1>(fused).to(projected_val.scalar_type()), batch_sizes, num_nodes, out_dim),
    };
  }
#endif

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
      auto edges = compress_scores(edge_compress_name, scores, mask.to(torch::kBool));
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

#ifdef WITH_CUDA
  if (pairwise_kind == "low_rank_bilinear" &&
      edge_compress_name == "softsign" &&
      flat_val.is_cuda() &&
      flat_projected_state.is_cuda() &&
      flat_projected_val.is_cuda() &&
      in_weight.has_value() &&
      out_weight.has_value() &&
      jakal_net_low_rank_propagation_topk_forward_cuda_available() &&
      !flat_val.requires_grad() &&
      !flat_projected_state.requires_grad() &&
      !flat_projected_val.requires_grad() &&
      !in_weight.value().requires_grad() &&
      !out_weight.value().requires_grad() &&
      !weight.requires_grad() &&
      k <= 32) {
    auto projected_target = torch::matmul(flat_val, out_weight.value().transpose(0, 1)).contiguous();
    auto projected_source = torch::matmul(flat_val, in_weight.value().transpose(0, 1)).contiguous();
    auto weighted_projected_source =
        projected_source * weight.view({1, 1, -1}).to(projected_source.scalar_type());
    const auto score_bias = bias.has_value() ? bias.value().item<double>() : 0.0;
    auto fused = low_rank_propagation_topk_forward_cuda_wrapper(
        weighted_projected_source,
        projected_target,
        flat_projected_state.to(torch::kFloat32).contiguous(),
        flat_projected_val.to(torch::kFloat32).contiguous(),
        k,
        score_bias,
        false);
    return {
        reshape_state(std::get<0>(fused).to(projected_state.scalar_type()), batch_sizes, num_nodes),
        reshape_val(std::get<1>(fused).to(projected_val.scalar_type()), batch_sizes, num_nodes, out_dim),
    };
  }
#endif

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
    auto edges = compress_scores(edge_compress_name, best_scores);
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
    const std::string& route_compress_name,
    const torch::Tensor& sender_strength,
    const torch::Tensor& src_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t dst_nodes,
    int64_t src_block_size,
    int64_t dst_block_size) {
  require_known_route_compress(route_compress_name);
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
    auto routes = route_compress_name == "signed_entmax15"
                      ? signed_entmax15_scores(logits)
                      : torch::softmax(logits, -1);
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

    auto state_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(state_acc_dtype) *
         flat_projected_state.slice(1, src_start, src_end).to(state_acc_dtype));
    auto val_sender =
        (flat_sender_strength.slice(1, src_start, src_end).to(val_acc_dtype).unsqueeze(-1) *
         flat_projected_val.slice(1, src_start, src_end).to(val_acc_dtype));

    if (route_compress_name == "signed_entmax15") {
      std::vector<torch::Tensor> logits_blocks;
      for (int64_t dst_start = 0; dst_start < dst_nodes; dst_start += dst_step) {
        const auto dst_end = std::min(dst_start + dst_step, dst_nodes);
        logits_blocks.push_back(route_block_logits(
            route_kind, route_context, in_weight, in_bias, out_weight, out_bias, dst_start, dst_end));
      }
      auto routes = signed_entmax15_scores(torch::cat(logits_blocks, -1));
      auto transport = routes.transpose(1, 2).contiguous();
      delta_state = delta_state +
                    torch::bmm(transport.to(state_acc_dtype), state_sender.unsqueeze(-1)).squeeze(-1);
      delta_val = delta_val + torch::bmm(transport.to(val_acc_dtype), val_sender);
      continue;
    }

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
        "softmax",
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

#ifdef WITH_CUDA
  if (route_kind == "low_rank_bilinear_route" &&
      flat_src_val.is_cuda() &&
      flat_dst_val.is_cuda() &&
      flat_sender_strength.is_cuda() &&
      flat_projected_state.is_cuda() &&
      flat_projected_val.is_cuda() &&
      source_weight.has_value() &&
      target_weight.has_value() &&
      jakal_net_low_rank_pairwise_topk_forward_cuda_available() &&
      !flat_src_val.requires_grad() &&
      !flat_dst_val.requires_grad() &&
      !flat_sender_strength.requires_grad() &&
      !flat_projected_state.requires_grad() &&
      !flat_projected_val.requires_grad() &&
      !source_weight.value().requires_grad() &&
      !target_weight.value().requires_grad() &&
      !core_weight.requires_grad() &&
      k <= 32) {
    auto projected_source =
        torch::matmul(flat_src_val, source_weight.value().transpose(0, 1)).contiguous();
    auto weighted_projected_source =
        projected_source * core_weight.view({1, 1, -1}).to(projected_source.scalar_type());
    if (temperature != 1.0) {
      weighted_projected_source = weighted_projected_source / temperature;
    }
    auto projected_target =
        torch::matmul(flat_dst_val, target_weight.value().transpose(0, 1)).contiguous();
    auto weighted_projected_state =
        (flat_sender_strength.to(torch::kFloat32) * flat_projected_state.to(torch::kFloat32))
            .contiguous();
    auto weighted_projected_val =
        (flat_sender_strength.to(torch::kFloat32).unsqueeze(-1) *
         flat_projected_val.to(torch::kFloat32))
            .contiguous();
    auto fused = low_rank_pairwise_topk_forward_cuda_wrapper(
        weighted_projected_source,
        projected_target,
        weighted_projected_state,
        weighted_projected_val,
        k,
        (bias.has_value() && bias.value().defined() && bias.value().numel() != 0)
            ? bias.value().item<double>()
            : 0.0,
        0);
    return {
        reshape_state(std::get<0>(fused).to(projected_state.scalar_type()), batch_sizes, dst_nodes),
        reshape_val(std::get<1>(fused).to(projected_val.scalar_type()), batch_sizes, dst_nodes, out_dim),
    };
  }
#endif

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

TORCH_LIBRARY(jakal_net, m) {
  m.def("signed_entmax15(Tensor scores, Tensor mask) -> Tensor");
  m.def("signed_entmax15_backward(Tensor scores, Tensor routes, Tensor grad_routes, Tensor mask) -> Tensor");
}

TORCH_LIBRARY_IMPL(jakal_net, CPU, m) {
  m.impl(
      "signed_entmax15",
      [](const torch::Tensor& scores, const torch::Tensor& mask) {
        auto packed_mask = packed_optional_tensor(mask);
        return signed_entmax15_scores(scores, packed_mask);
      });
  m.impl(
      "signed_entmax15_backward",
      [](const torch::Tensor& scores,
         const torch::Tensor& routes,
         const torch::Tensor& grad_routes,
         const torch::Tensor& mask) {
        auto packed_mask = packed_optional_tensor(mask);
        return signed_entmax15_backward_scores(scores, routes, grad_routes, packed_mask);
      });
}

TORCH_LIBRARY_IMPL(jakal_net, CUDA, m) {
  m.impl(
      "signed_entmax15",
      [](const torch::Tensor& scores, const torch::Tensor& mask) {
        auto packed_mask = packed_optional_tensor(mask);
        return signed_entmax15_scores(scores, packed_mask);
      });
  m.impl(
      "signed_entmax15_backward",
      [](const torch::Tensor& scores,
         const torch::Tensor& routes,
         const torch::Tensor& grad_routes,
         const torch::Tensor& mask) {
        auto packed_mask = packed_optional_tensor(mask);
        return signed_entmax15_backward_scores(scores, routes, grad_routes, packed_mask);
      });
}

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
  m.def(
      "low_rank_pairwise_topk_forward_cuda",
      &low_rank_pairwise_topk_forward_cuda_wrapper,
      "CUDA fused low-rank pairwise top-k forward");
  m.def(
      "low_rank_propagation_topk_forward_cuda",
      &low_rank_propagation_topk_forward_cuda_wrapper,
      "CUDA fused low-rank propagation top-k forward");
  m.def(
      "low_rank_propagation_window_forward_cuda",
      &low_rank_propagation_window_forward_cuda_wrapper,
      "CUDA fused low-rank propagation window forward");
  m.def(
      "low_rank_propagation_window_entmax15_forward_cuda",
      &low_rank_propagation_window_entmax15_forward_cuda_wrapper,
      "CUDA fused low-rank propagation window forward with signed entmax15");
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
  m.def(
      "causal_memory_scan_fused",
      &causal_memory_scan_fused,
      "Native fused causal-memory scan stub");
  m.def(
      "causal_memory_scan_fused_trace",
      &causal_memory_scan_fused_trace,
      "Native fused causal-memory scan with step trace");
  m.def(
      "causal_memory_scan_fused_checkpoints",
      &causal_memory_scan_fused_checkpoints,
      "Native fused causal-memory scan with chunk checkpoint trace");
  m.def(
      "causal_memory_scan_fused_backward_cuda",
      &causal_memory_scan_fused_backward_cuda,
      "CUDA backward for fused causal-memory scan");
}
