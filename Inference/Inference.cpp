#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include "Lenet.h"
#include "utils.h"
#include "Minionn.h"
int main()
{
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Lenet.ptl";
    auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Minionn.ptl";
    Minionn module;
    layer_weight_extraction(module, path);
    
    auto datapath = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/data/MNIST/raw";
    std::cout << "Accuracy: " << test_layer(module,datapath)<< "%" << std::endl;
    //std::cout << "Accuracy: " << test_acc(module, datapath) << "%" << std::endl;

    return 0;
}


