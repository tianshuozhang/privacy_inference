#pragma once
#include <torch/torch.h>
#include"socket.h"
#include"Quantization.h"
// 你的头文件保护代码 （如果有的话）
#ifndef BASE_H
#define BASE_H

struct Base : torch::nn::Module {
	// 声明一个纯虚函数 'forward'
	virtual torch::Tensor forward(torch::Tensor x) = 0;
	virtual torch::Tensor layer_forward(torch::Tensor x) = 0;
	std::string filename;
};

struct ModuleInfo {
	std::shared_ptr<torch::nn::Module> ptr;
	std::string type;
};
#endif