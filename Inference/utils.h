#pragma once
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/data/datasets/mnist.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <string>
#include <functional>
#include "Lenet.h"
#include"socket.h"
#include "cifar10.h"

float test_acc(Base &module,const std::string &datapath, std::string tag);

float test_layer(Base &module, const std::string &datapath);

void layer_weight_extraction(Base& module, const std::string path);

void weight(Base& module, const std::string path);