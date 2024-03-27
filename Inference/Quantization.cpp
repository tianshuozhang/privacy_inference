#include "Quantization.h"
#include <c10/util/variant.h>
torch::Tensor conv_2d(torch::Tensor input, torch::nn::Conv2dImpl* conv_module) {
	// 获取权重、偏置及其他参数
	auto weights = conv_module->weight.to(torch::kInt64);
	auto bias = conv_module->bias.to(torch::kInt64);
	auto options = conv_module->options;
	// 获取步长、填充、膨胀以及卷积分组的值
	int stride = options.stride()->operator[](0);
	//auto padding = options.padding();
	int padding = 0;
	int dilation = options.dilation()->operator[](0);
	auto groups = options.groups();
	// 使用torch::nn::functional::conv2d进行卷积运算
	auto output = torch::nn::functional::detail::conv2d(input, weights, bias,
		options.stride(),
		options.padding(),
		options.dilation(),
		options.groups());
	output = torch::floor(output / 256);
	
	/*
	int batch_size = input.size(0);
	int input_channels = input.size(1);
	int input_height = input.size(2);
	int input_width = input.size(3);
	int kernel_channels = weights.size(0);
	int kernel_height = weights.size(2);
	int kernel_width = weights.size(3);
	
	
	int output_height = 1 + (input_height + 2 * padding- ((kernel_height - 1) *dilation + 1)) / stride;
	int output_width = 1 + (input_width  + 2 * padding - ((kernel_width - 1) * dilation + 1)) / stride;

	torch::Tensor output = torch::zeros({ batch_size, kernel_channels, output_height, output_width }).to(torch::kInt64);
	for (int batch = 0; batch < batch_size; batch++) {
		for (int filter = 0; filter < kernel_channels; filter++) {
			for (int h = 0; h < output_height; h++) {
				for (int w = 0; w < output_width; w++) {
					output[batch] += torch::sum(input[batch]
						.slice(1, h * stride, h * stride + kernel_height)
						.slice(2, w * stride, w * stride + kernel_width)
						* weights[filter]);
					output[batch] = torch::div(output[batch], 256).to(torch::kInt64)+bias[filter];
				}
			}
		}
	}
	
	*/
	output = output.to(torch::kFloat);
	std::cout << "convd finnish\n";
	return output;
	
	
}
torch::Tensor linear(torch::Tensor input, torch::nn::LinearImpl* conv_module)
{
	// 获取权重、偏置及其他参数
	auto weights = conv_module->weight.to(torch::kInt64);
	auto bias = conv_module->bias.to(torch::kInt64);
	
	auto transposed_weights = weights.transpose(0, 1);
	std::cout << "begin\n";
	auto result = torch::mm(input.to(torch::kInt64), transposed_weights);
	std::cout << result.dtype() << bias.dtype() << std::endl;
	result = torch::div(result,256, "trunc").to(torch::kInt64);
	std::cout << result.dtype() << bias.dtype() << std::endl;
	result += bias;
	std::cout << "linear finnish\n";
	return result.to(torch::kFloat);

}

torch::Tensor relu_int(torch::Tensor input)
{
	// 假设input是你的输入张量
	std::cout << input.dtype();
	torch::Tensor zeros = torch::zeros_like(input);
	torch::Tensor relu_output = torch::max(input, zeros);
	return relu_output;
}

torch::Tensor maxpool_2d(torch::Tensor input, torch::nn::MaxPool2dImpl* maxpool_module)
{
	auto weight = maxpool_module;
	return input;
}
