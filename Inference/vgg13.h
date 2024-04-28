#pragma once
#include "Base.h"
#include <torch/torch.h>
#include <torch/nn/module.h>
#ifndef VGG13_H
#define VGG13_H
struct vgg13 :
    public Base
{
    vgg13();
    ~vgg13();
    torch::Tensor forward(torch::Tensor x);
    ModuleInfo GetModuleByName(const std::string& name);

    torch::Tensor layer_forward(torch::Tensor x);

    void startServer();
    void startClient(int port);
    std::vector<torch::Tensor> in;
    std::vector<torch::Tensor> out;
    std::mutex mu;
    bool run_thread;
    torch::nn::Conv2d layer1{ nullptr }, layer3{ nullptr }, layer6{ nullptr },
        layer8{ nullptr }, layer11{ nullptr }, layer13{ nullptr },
        layer16{ nullptr }, layer18{ nullptr }, layer21{ nullptr }
        , layer23{ nullptr };

    torch::nn::MaxPool2d layer5{ nullptr }, layer10{ nullptr },
        layer15{ nullptr }, layer20{ nullptr }, layer25{ nullptr };

    torch::nn::ReLU layer2{ nullptr }, layer4{ nullptr }, layer7{ nullptr },
        layer9{ nullptr }, layer12{ nullptr }, layer14{ nullptr },
        layer17{ nullptr }, layer19{ nullptr }, layer22{ nullptr }, 
        layer24{ nullptr },layer28{ nullptr }, layer31{ nullptr };
    torch::nn::Linear layer27{ nullptr }, layer30{ nullptr }, layer32{ nullptr };
    torch::nn::Dropout layer26{ nullptr }, layer29{ nullptr };

};

#endif // LENET_H
