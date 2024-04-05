#include <iostream>
#include <torch/torch.h>

#include"Resnet.h"
int main()
{
    std::cout << "begin\n";
    Resnet model;
    std::cout << "model\n";
    auto input = torch::randn({ 32, 3, 32, 32 });
    std::cout << "data\n";
    auto output = model.forward(input);
    std::cout << output.argmax(1);
    return 0;
}

