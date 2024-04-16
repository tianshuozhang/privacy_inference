import torch
from models.Resnet18 import Resnet18
model=torch.load("./checkpoints/Resnet182024-03-27-13-42-56.pth").to('cpu')
example = torch.rand(1,3,32,32) # 注意，我这里导出的是CUDA版的模型，因为我的模型是在GPU中进行训练的
traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(torch.ones(1,3,32,32))
traced_script_module.save('./checkpoints/Resnet18.ptl')
print(traced_script_module)
for name,value in traced_script_module.named_parameters():
    print(name,value.shape)