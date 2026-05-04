#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>

bool jakal_net_compiled_with_cuda_source();
bool jakal_net_query_topk_reduce_cuda_available();
bool jakal_net_low_rank_pairwise_topk_forward_cuda_available();
bool jakal_net_low_rank_propagation_topk_forward_cuda_available();
bool jakal_net_low_rank_propagation_window_forward_cuda_available();
bool jakal_net_low_rank_propagation_window_signed_abs_forward_cuda_available();
bool jakal_net_low_rank_propagation_causal_dense_signed_abs_forward_cuda_available();
bool jakal_net_diagonal_propagation_causal_dense_signed_abs_cuda_available();
bool jakal_net_low_rank_propagation_dense_forward_cuda_available();
bool jakal_net_low_rank_dense_scores_tf32_cuda_available();
bool jakal_net_low_rank_propagation_dense_tf32_forward_cuda_available();

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
    torch::Tensor projected_target,
    const torch::Tensor& weighted_projected_state,
    const torch::Tensor& weighted_projected_val,
    int64_t topk,
    double score_bias,
    int64_t compress_kind);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_topk_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t topk,
    double score_bias,
    int64_t compress_kind);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_dense_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t compress_kind);

torch::Tensor jakal_net_low_rank_dense_scores_tf32_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_dense_tf32_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    double score_bias);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_signed_abs_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    int64_t window,
    double score_bias);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_causal_dense_signed_abs_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    double score_bias);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_diagonal_propagation_causal_dense_signed_abs_forward_cuda(
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& normalized_weight,
    const torch::Tensor& bias);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_diagonal_propagation_causal_dense_signed_abs_backward_cuda(
    const torch::Tensor& layer_val,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val,
    const torch::Tensor& normalized_weight,
    const torch::Tensor& bias,
    const torch::Tensor& grad_delta_state,
    const torch::Tensor& grad_delta_val);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_low_rank_propagation_window_entmax15_forward_cuda(
    const torch::Tensor& weighted_projected_source,
    torch::Tensor projected_target,
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

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_nomemory_exact_scores_backward_cuda(
    const torch::Tensor& scores,
    const torch::Tensor& edges,
    const torch::Tensor& grad_pre_norm,
    const torch::Tensor& val,
    int64_t compress_kind);

std::tuple<torch::Tensor, torch::Tensor>
jakal_net_apply_delta_to_layer_cuda(
    const torch::Tensor& layer_state,
    const torch::Tensor& layer_val,
    const torch::Tensor& delta_state,
    const torch::Tensor& delta_val,
    const torch::Tensor& val_norm_weight,
    const torch::Tensor& val_norm_bias,
    const std::string& state_activation_name,
    bool use_fused_forward);

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_nomemory_low_rank_pairwise_backward_cuda(
    const torch::Tensor& layer_val,
    const torch::Tensor& source_weight,
    const torch::Tensor& target_weight,
    const torch::Tensor& core_weight,
    const torch::Tensor& projected_source,
    const torch::Tensor& projected_target,
    const torch::Tensor& grad_scores);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jakal_net_nomemory_low_rank_exact_val_layer_backward_cuda(
    const torch::Tensor& layer_val,
    const torch::Tensor& source_weight,
    const torch::Tensor& target_weight,
    const torch::Tensor& core_weight,
    const torch::Tensor& scores,
    const torch::Tensor& edges,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& grad_output,
    int64_t compress_kind);

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
    const std::vector<int64_t>& skip_topks);

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
    const std::vector<int64_t>& skip_topks);

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
    const std::vector<torch::Tensor>& grad_next_memory);
