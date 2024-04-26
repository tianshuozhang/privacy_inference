#pragma once
#include "Base.h"
#include <torch/torch.h>
#include <torch/nn/module.h>
#ifndef VGG11_H
#define VGG11_H
struct vgg11 :
    public Base
{
    vgg11();
    ~vgg11();
    torch::Tensor forward(torch::Tensor x);
    ModuleInfo GetModuleByName(const std::string& name);

    torch::Tensor layer_forward(torch::Tensor x);

    void startServer();
    void startClient(int port);
    std::vector<torch::Tensor> in;
    std::vector<torch::Tensor> out;
    std::mutex mu;
    bool run_thread;
    torch::nn::Conv2d layer1{ nullptr }, layer4{ nullptr }, layer7{ nullptr },
        layer9{ nullptr }, layer12{ nullptr }, layer14{ nullptr },
    layer17{ nullptr }, layer19{ nullptr };
    torch::nn::MaxPool2d layer3{ nullptr }, layer6{ nullptr },
    layer11{ nullptr }, layer16{ nullptr }, layer21{ nullptr };
    torch::nn::ReLU layer2{ nullptr }, layer5{ nullptr }, layer8{ nullptr },
        layer10{ nullptr }, layer13{ nullptr }, layer15{ nullptr },
        layer18{ nullptr }, layer20{ nullptr }, layer24{ nullptr },
        layer27{ nullptr };
    torch::nn::Linear layer23{ nullptr }, layer26{ nullptr },layer28{ nullptr };
    torch::nn::Dropout layer22{ nullptr }, layer25{ nullptr };

};

#endif // LENET_H
