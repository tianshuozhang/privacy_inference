#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include "Lenet.h"
#include"Resnet.h"
#include "utils.h"
#include "Minionn.h"

int main()
{
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Lenet.ptl";
    //Lenet module;
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Minionn.ptl";
    //Minionn module;
    Resnet module;
    
    auto path= "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Resnet18.ptl";
    
    layer_weight_extraction(module, path);
    auto datapath = "../data/cifar-10-batches-bin";
    auto tag = "cifar10";
    
    std::cout << "Accuracy: " << test_layer(module,datapath,tag)<< "%" << std::endl;
    std::cout << "Accuracy: " << test_acc(module, datapath,tag) << "%" << std::endl;
    
    return 0;
}


