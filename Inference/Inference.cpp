#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include "Lenet.h"
#include"Resnet.h"
#include "utils.h"
#include "Minionn.h"
#include "cifar10_reader.hpp"
int main()
{
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Lenet.ptl";
    //Lenet module;
    //auto path = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Minionn.ptl";
    //Minionn module;
    Resnet module;
    auto path= "C:/Users/17612/Desktop/MPC/experment/privacy_inference/checkpoints/Resnet18.ptl";
    layer_weight_extraction(module, path);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    std::cout <<dataset.training_labels;
    exit(0);
    auto datapath = "C:/Users/17612/Desktop/MPC/experment/privacy_inference/data/cifar-10-batches-py";
    //std::cout << "Accuracy: " << test_layer(module,datapath)<< "%" << std::endl;
    //std::cout << "Accuracy: " << test_acc(module, datapath) << "%" << std::endl;
    
    return 0;
}


