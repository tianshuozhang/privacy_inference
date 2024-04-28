# privacy_inference
通过两个gpu进行线性层加速，可信部分进行核心计算，包括激活函数等非线性计算。
## training
pytorch框架进行训练，训练完成后通过`torch.jit`的方式来保存，变为可以在`c++`中读取的类型。

## inference
利用libtorch进行推理,对于pytorch训练的模型，在这里重新定义实现模型，然后读取训练之后的参数，提取权重赋值给模型进行推理。

## 数据集
主要实现了MNIST和CIFAR数据集，其中CIFAR数据集在`c++`中的使用独立实现。

## 模型选择
- 对于MNIST数据集实现了Lenet和Minionn模型
- 对于CIFAR数据集实现了Resnet18，Resnet34，Vgg11，vgg13等经典模型

## 计算类型
均通过浮点数类型计算进行推理。
