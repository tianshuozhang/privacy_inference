#include <torch/torch.h>
#include <torch/nn/module.h>
#include"Base.h"

#ifndef RESNET34_H
#define RESNET34_H
struct Resnet34 : Base {
    Resnet34();
    ~Resnet34();
    torch::Tensor forward(torch::Tensor x);
    ModuleInfo GetModuleByName(const std::string& name);

    torch::Tensor layer_forward(torch::Tensor x);

    void startServer();
    void startClient(int port);
    torch::nn::Conv2d layer1{ nullptr }, layer3{ nullptr }, layer5{ nullptr }, layer8{ nullptr },
        layer10{ nullptr }, layer13{ nullptr }, layer15{ nullptr }, layer18{ nullptr },
        layer20{ nullptr }, layer22_shortcut{ nullptr }, layer24{ nullptr }, layer26{ nullptr }, layer29{ nullptr }
        , layer31{ nullptr }, layer34{ nullptr }, layer36{ nullptr }, layer39{ nullptr }, layer41{ nullptr }
    , layer43_shortcut{ nullptr }, layer45{ nullptr }, layer47{ nullptr }, layer50{ nullptr }
    , layer52{ nullptr }, layer55{ nullptr }, layer57{ nullptr }, layer60{ nullptr }
    , layer62{ nullptr }, layer65{ nullptr }, layer67{ nullptr }, layer70{ nullptr }
    , layer72{ nullptr }, layer74_shortcut{ nullptr }, layer76{ nullptr }, layer78{ nullptr }
    , layer81{ nullptr }, layer83{ nullptr };
    torch::nn::BatchNorm2d layer2{ nullptr }, layer4{ nullptr }, layer6{ nullptr }, layer9{ nullptr }
        , layer11{ nullptr }, layer14{ nullptr }, layer16{ nullptr }, layer19{ nullptr }
        , layer21{ nullptr }, layer23_shortcut{ nullptr }, layer25{ nullptr }, layer27{ nullptr }, layer29_shortcut{ nullptr }
        , layer30{ nullptr }, layer32{ nullptr }, layer35{ nullptr }, layer37{ nullptr }, layer40{ nullptr }
    , layer42{ nullptr }, layer44_shortcut{ nullptr }, layer46{ nullptr }, layer48{ nullptr }, layer51{ nullptr }
    , layer53{ nullptr }, layer56{ nullptr }, layer58{ nullptr }, layer61{ nullptr }
    , layer63{ nullptr }, layer66{ nullptr }, layer68{ nullptr }, layer71{ nullptr }
    , layer73{ nullptr }, layer75_shortcut{ nullptr }, layer77{ nullptr }, layer79{ nullptr }
    , layer82{ nullptr }, layer84{ nullptr };

    torch::nn::Linear layer86{ nullptr };
    torch::nn::ReLU act;
    std::vector<torch::Tensor> in;
    std::vector<torch::Tensor> out;
    std::mutex mu;
    bool run_thread;
};

#endif