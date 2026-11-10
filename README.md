<div align="center">

# 🔒 privacy_inference

**Privacy-preserving DNN inference with GPU-offloaded linear layers**
**GPU 加速线性层 + 可信端非线性计算的隐私推理框架**

![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![LibTorch](https://img.shields.io/badge/LibTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Datasets](https://img.shields.io/badge/Datasets-MNIST%20%7C%20CIFAR--10-blue?style=flat-square)

</div>

---

## 📖 Overview · 简介

**English** — Linear layers are offloaded to **two GPUs** for acceleration, while the **trusted side** performs the core computation, including activations and other non-linear operations. Models are trained in PyTorch, exported with `torch.jit`, then re-implemented in C++/LibTorch for inference with the trained weights.

**中文** — 通过 **两块 GPU** 加速线性层计算，**可信部分** 负责核心计算（包括激活函数等非线性运算）。模型先在 PyTorch 中训练并通过 `torch.jit` 导出，再在 C++（LibTorch）中重新定义网络结构、加载训练参数完成推理。

## 🧩 Pipeline · 流程

```text
 PyTorch training ──► torch.jit.trace ──► *.ptl ──► C++ / LibTorch model ──► inference
 (training/)                                          (Inference/)
```

## 🧠 Supported Models · 支持的模型

| Dataset · 数据集 | Models · 模型 |
| --- | --- |
| MNIST | LeNet, MiniONN |
| CIFAR-10 | ResNet-18, ResNet-34, VGG-11, VGG-13 |

> CIFAR-10 loading in C++ is implemented from scratch (`Inference/cifar10.cpp`).
> C++ 端的 CIFAR-10 数据读取为独立实现。All inference uses floating-point arithmetic. 推理均使用浮点数计算。

## 📁 Structure · 目录结构

```text
privacy_inference
├── training/              # PyTorch side
│   ├── models/            # LeNet, MiniONN, ResNet, VGG, SecureML, ...
│   ├── trace.py           # export to TorchScript (*.ptl)
│   ├── transform.py
│   └── utils.py           # accuracy evaluation
└── Inference/             # C++ / LibTorch side
    ├── Inference.cpp      # entry point (choose model / weights / dataset)
    ├── Lenet / Minionn / Resnet18 / Resnet34 / vgg11 / vgg13 (.h/.cpp)
    ├── Quantization.*     # quantisation helpers
    ├── socket.*           # communication between parties
    └── cifar10.*          # CIFAR-10 reader
```

## 🚀 Usage · 使用方法

1. **Train & export · 训练并导出**
   ```bash
   cd training
   python trace.py        # writes checkpoints/<model>.ptl
   ```
2. **Inference · 推理**
   Set the model, `.ptl` path and dataset path in `Inference/Inference.cpp`, then build against LibTorch.
   在 `Inference/Inference.cpp` 中选择模型、权重路径和数据集路径，链接 LibTorch 编译运行。
