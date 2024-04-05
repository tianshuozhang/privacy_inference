#include <torch/torch.h>
#include <torch/nn/module.h>
#include"Base.h"

#ifndef RESNET_H
#define RESNET_H
struct Resnet : Base {
    Resnet();
    torch::Tensor forward(torch::Tensor x);
    ModuleInfo GetModuleByName(const std::string& name);

    torch::Tensor layer_forward(torch::Tensor x);

    void startServer(torch::Tensor& data);
    void startClient(int port);
    torch::nn::Conv2d layer1{nullptr}, layer3{ nullptr }, layer5{ nullptr }, layer8{ nullptr },
        layer10{ nullptr }, layer13{ nullptr }, layer15{ nullptr }, layer17_shortcut{ nullptr },
        layer19{ nullptr }, layer21{ nullptr }, layer24{ nullptr }, layer26{ nullptr }, layer28_shortcut{ nullptr }
        ,layer30{ nullptr }, layer32{ nullptr },layer35{ nullptr }, layer37{ nullptr }, layer39_shortcut{ nullptr }
        , layer41{ nullptr },layer43{ nullptr };
    torch::nn::BatchNorm2d layer2{ nullptr }, layer4{ nullptr }, layer6{ nullptr }, layer9{ nullptr }
        , layer11{ nullptr }, layer14{ nullptr }, layer16{ nullptr }, layer18_shortcut{ nullptr }
        , layer20{ nullptr }, layer22{ nullptr }, layer25{ nullptr }, layer27{ nullptr }, layer29_shortcut{ nullptr }
        , layer31{ nullptr }, layer33{ nullptr }, layer36{ nullptr }, layer38{ nullptr }, layer40_shortcut{ nullptr }
        , layer42{ nullptr }, layer44{ nullptr };
    
    torch::nn::Linear layer46{ nullptr };
    torch::nn::ReLU act;
};

#endif