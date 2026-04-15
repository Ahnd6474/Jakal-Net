#pragma once

#include <tuple>

#include <torch/extension.h>

bool jakal_net_compiled_with_cuda_source();
bool jakal_net_query_topk_reduce_cuda_available();

std::tuple<torch::Tensor, torch::Tensor> jakal_net_query_topk_reduce_cuda(
    const torch::Tensor& edges,
    const torch::Tensor& indices,
    const torch::Tensor& projected_state,
    const torch::Tensor& projected_val);
