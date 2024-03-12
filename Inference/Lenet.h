// lenet.h
#include <torch/torch.h>
#include <torch/nn/module.h>
#include"Base.h"

#ifndef LENET_H
#define LENET_H
struct Lenet : Base {
    Lenet();
    torch::Tensor forward(torch::Tensor x);
    ModuleInfo GetModuleByName(const std::string& name);

    torch::Tensor layer_forward(torch::Tensor x) ;

    void startServer(torch::Tensor& data);
    void startClient(int port);
    torch::nn::Conv2d layer1{ nullptr }, layer2{ nullptr };
    torch::nn::Linear layer3{ nullptr }, layer4{ nullptr };
    torch::nn::ReLU act;
    torch::nn::MaxPool2d pool;
};
#endif // LENET_H
