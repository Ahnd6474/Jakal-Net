#pragma once

#include <tuple>

#include <torch/extension.h>

bool jakal_net_compiled_with_cuda_source();
bool jakal_net_query_topk_reduce_cuda_available();
bool jakal_net_low_rank_pairwise_topk_forward_cuda_available();
bool jakal_net_low_rank_propagation_topk_forward_cuda_available();
bool jakal_net_low_rank_propagation_window_forward_cuda_available();

std::tuple<torch::Tensor, torch::Tensor> jakal_net_query_topk_reduce_cuda(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> jakal_net_query_topk_reduce_backward_cuda(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_low_rank_pairwise_topk_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& weighted_projected_state,
    const torch::Tensor& weighted_projected_val,
    int64_t topk,
    bool signed_abs_softmax);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_topk_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    double score_bias);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias);

torch::Tensor jakal_net_softsign_backward_cuda(
    const torch::Tensor& scores,
    const torch::Tensor& grad_edges);

torch::Tensor jakal_net_softmax_backward_cuda(
    const torch::Tensor& routes,
    const torch::Tensor& grad_routes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_diagonal_pairwise_topk_backward_cuda(
    const torch::Tensor& query_val,
    const torch::Tensor& source_val,
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    const torch::Tensor& grad_scores,
    double temperature);

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
    double temperature);
