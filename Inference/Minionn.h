#pragma once

#include"Base.h"
#include <torch/torch.h>
#ifndef MINIONN_H_
#define MINIONN_H_
struct Minionn: Base{
    Minionn();
    torch::Tensor forward(torch::Tensor x);
    ModuleInfo GetModuleByName(const std::string& name);

    torch::Tensor layer_forward(torch::Tensor x);

    void startServer(torch::Tensor& data);
    void startClient(int port);
    torch::nn::Conv2d layer1, layer2;
    torch::nn::Linear layer3, layer4;
    torch::nn::ReLU act;
    torch::nn::MaxPool2d pool;
};

#endif // MINIONN_H_