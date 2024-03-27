#pragma once
#include <torch/torch.h>
#include"socket.h"
#include"Quantization.h"
// ���ͷ�ļ��������� ������еĻ���
#ifndef BASE_H
#define BASE_H

struct Base : torch::nn::Module {
	// ����һ�����麯�� 'forward'
	virtual torch::Tensor forward(torch::Tensor x) = 0;
	virtual torch::Tensor layer_forward(torch::Tensor x) = 0;
	std::string filename;
};

struct ModuleInfo {
	std::shared_ptr<torch::nn::Module> ptr;
	std::string type;
};
#endif