#include"Resnet.h"
Resnet::Resnet()
    : layer1(register_module("layer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)))),
    layer3(register_module("layer3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)))),
    layer5(register_module("layer5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)))),
    layer8(register_module("layer8", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)))),
    layer10(register_module("layer10", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)))),
    layer13(register_module("layer13", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)))),
    layer15(register_module("layer15", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)))),
    layer17_shortcut(register_module("layer17_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).stride(2)))),
    layer19(register_module("layer19", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)))),
    layer21(register_module("layer21", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)))),
    layer24(register_module("layer24", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)))),
    layer26(register_module("layer26", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)))),
    layer28_shortcut(register_module("layer28_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).stride(2)))),
    layer30(register_module("layer30", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)))),
    layer32(register_module("layer32", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)))),
    layer35(register_module("layer35", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)))),
    layer37(register_module("layer37", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)))),
    layer39_shortcut(register_module("layer39_shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1).stride(2)))),
    layer41(register_module("layer41", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)))),
    layer43(register_module("layer43", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)))),

    layer2(register_module("layer2", torch::nn::BatchNorm2d(64))),
    layer4(register_module("layer4", torch::nn::BatchNorm2d(64))),
    layer6(register_module("layer6", torch::nn::BatchNorm2d(64))),
    layer9(register_module("layer9", torch::nn::BatchNorm2d(64))),
    layer11(register_module("layer11", torch::nn::BatchNorm2d(64))),
    layer14(register_module("layer14", torch::nn::BatchNorm2d(128))),
    layer16(register_module("layer16", torch::nn::BatchNorm2d(128))),
    layer18_shortcut(register_module("layer18_shortcut", torch::nn::BatchNorm2d(128))),
    layer20(register_module("layer20", torch::nn::BatchNorm2d(128))),
    layer22(register_module("layer22", torch::nn::BatchNorm2d(128))),
    layer25(register_module("layer25", torch::nn::BatchNorm2d(256))),
    layer27(register_module("layer27", torch::nn::BatchNorm2d(256))),
    layer29_shortcut(register_module("layer29_shortcut", torch::nn::BatchNorm2d(256))),
    layer31(register_module("layer31", torch::nn::BatchNorm2d(256))),
    layer33(register_module("layer33", torch::nn::BatchNorm2d(256))),
    layer36(register_module("layer36", torch::nn::BatchNorm2d(512))),
    layer38(register_module("layer38", torch::nn::BatchNorm2d(512))),
    layer40_shortcut(register_module("layer40_shortcut", torch::nn::BatchNorm2d(512))),
    layer42(register_module("layer42", torch::nn::BatchNorm2d(512))),
    layer44(register_module("layer44", torch::nn::BatchNorm2d(512))),
    layer46(register_module("layer46", torch::nn::Linear(512,10))),
    act(register_module("act", torch::nn::ReLU())){
    filename = "Resnet.txt";
}

ModuleInfo Resnet::GetModuleByName(const std::string& name) {
    ModuleInfo module_info;
    for (auto& pair : this->named_children()) {
        if (pair.key() == name) {
            module_info.ptr = pair.value();
            // Checking the type and setting it
            if (module_info.ptr->as<torch::nn::Conv2d>() != nullptr) {
                module_info.type = "torch::nn::Conv2d";
            }
            else if (module_info.ptr->as<torch::nn::Linear>() != nullptr) {
                module_info.type = "torch::nn::Linear";
            }
            else if (module_info.ptr->as<torch::nn::ReLU>() != nullptr) {
                module_info.type = "torch::nn::ReLU";
            }
            else if (module_info.ptr->as<torch::nn::BatchNorm2d>() != nullptr) {
                module_info.type = "torch::nn::BatchNorm2d";
            }
            else {
                module_info.type = "unknown";
            }
            return module_info;
        }
    }
    // Default return with nullptr and "unknown"
    module_info.ptr = nullptr;
    module_info.type = "unknown";
    return module_info;
}

torch::Tensor Resnet::layer_forward(torch::Tensor x) {
    std::thread serverThread(std::bind(&Resnet::startServer, this, std::ref(x)));
    std::thread clientThread(std::bind(&Resnet::startClient, this, 5010));
    std::thread clientThread_1(std::bind(&Resnet::startClient, this, 5020));
    clientThread.join();
    clientThread_1.join();
    serverThread.join();
    WSACleanup();
    return x;
}

void Resnet::startServer(torch::Tensor& data)
{

}
void Resnet::startClient(int port) {

}

torch::Tensor Resnet::forward(torch::Tensor x) {
    
    auto out = layer1->forward(x);
    out = layer2->forward(out);
    out = act(out);
    x = out.clone();
    out = layer3(out);
    out = layer4(out);
    out = act(out);
    out = layer5(out);
    out = layer6(out);
    out += x;
    out = act(out);
    x = out.clone();
    out = layer8(x);
    out = layer9(out);
    out = act(out);
    out = layer10(out);
    out = layer11(out);
    out += x;
    out = act(out);
    x = out.clone();
    out = layer13(x);
    out = layer14(out);
    out = act(out);
    out = layer15(out);
    out = layer16(out);
    x = layer17_shortcut(x);
    out += layer18_shortcut(x);
    out = act(out);
    x = out.clone();
    out = layer19(x);
    out = layer20(out);
    out = act(out);
    out = layer21(out);
    out = layer22(out);
    out +=x;
    out = act(out);
    x = out.clone();
    out = layer24(x);
    out = layer25(out);
    out = act(out);
    out = layer26(out);
    out = layer27(out);
    x = layer28_shortcut(x);
    out += layer29_shortcut(x);
    out = act(out);
    x = out.clone();
    out = layer30(x);
    out = layer31(out);
    out = act(out);
    out = layer32(out);
    out = layer33(out);
    out += x;
    out = act(out);
    x = out.clone();
    out = layer35(x);
    out = layer36(out);
    out = act(out);
    out = layer37(out);
    out = layer38(out);
    x = layer39_shortcut(x);
    out += layer40_shortcut(x);
    out = act(out);
    x = out.clone();
    out = layer41(x);
    out = layer42(out);
    out = act(out);
    out = layer43(out);
    out = layer44(out);
    out += (x);
    out = act(out);
    out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(4));
    out = out.view({ out.size(0), -1 });
    out = layer46(out);
    return out;
}