#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include "Lenet.h"
#include"Resnet.h"
#include "utils.h"
#include "Minionn.h"
#include "cifar10.h"
int main()
{
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Lenet.ptl";
    //Lenet module;
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Minionn.ptl";
    //Minionn module;
    Resnet module;
    auto path= "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Resnet18.ptl";
    layer_weight_extraction(module, path);
    module.eval();
    auto datapath = "../data/cifar-10-batches-bin";
    auto images = read_batch(datapath);
    auto input = images[0].tensor;
    input = input.to(torch::kFloat32).reshape({1,3,32,32});
    input = input / 128 - 1;
    auto output = module.forward({input});
    std::cout <<"Label: " << static_cast<int>(images[0].label) << "\tout: " << output.argmax(1) << "\n";
    //std::cout << "Accuracy: " << test_layer(module,datapath)<< "%" << std::endl;
    //std::cout << "Accuracy: " << test_acc(module, datapath) << "%" << std::endl;
    
    return 0;
}


