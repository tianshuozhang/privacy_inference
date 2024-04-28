#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include "Lenet.h"
#include"Resnet18.h"
#include"Resnet34.h"
#include "utils.h"
#include "Minionn.h"
#include"vgg11.h"
int main()
{
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Lenet.ptl";
    //Lenet module;
    //auto datapath = "../data/MNIST/raw";
    //auto tag = "mnist";

    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Minionn.ptl";
    //Minionn module;
    //auto datapath = "../data/MNIST/raw";
    //auto tag = "mnist";
   
    Resnet18 module;
    auto path= "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Resnet18.ptl";
    auto datapath = "../data/cifar-10-batches-bin";
    auto tag = "cifar10";

    //Resnet34 module;
    //auto path= "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Resnet34.ptl";
    //auto datapath = "../data/cifar-10-batches-bin";
    //auto tag = "cifar10";

    //vgg11 module;
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/vgg11.ptl";
    //auto datapath = "../data/cifar-10-batches-bin";
    //auto tag = "cifar10";

    layer_weight_extraction(module, path);
    //std::cout << "Accuracy: " << test_acc(module, datapath, tag) << "%" << std::endl;
    std::cout << "Accuracy: " << test_layer(module, datapath, tag) << "%" << std::endl;
    
    return 0;
}


