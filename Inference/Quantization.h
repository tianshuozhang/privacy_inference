#pragma once
#include<torch/torch.h>

torch::Tensor conv_2d(torch::Tensor input,torch::nn::Conv2dImpl* conv_module);
torch::Tensor relu_int(torch::Tensor input);
torch::Tensor maxpool_2d(torch::Tensor input, torch::nn::MaxPool2dImpl* maxpool_module);
torch::Tensor linear(torch::Tensor input, torch::nn::LinearImpl* conv_module);
inline std::string padding_unwrap(torch::enumtype::kValid) {
	return "valid";
}

inline std::string padding_unwrap(torch::enumtype::kSame) {
	return "same";
}

template <size_t D>
torch::IntArrayRef padding_unwrap(const torch::ExpandingArray<D>& array) {
	return array;
}