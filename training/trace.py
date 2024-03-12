import torch
model=torch.load("./checkpoints/Minionn2024-01-14-21-38-51.pth").to('cpu')
example = torch.rand(1,1,28,28) # 注意，我这里导出的是CUDA版的模型，因为我的模型是在GPU中进行训练的
model = model.eval()
traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(torch.ones(1,1,28,28))
traced_script_module.save('./checkpoints/Minionn.ptl')
print(traced_script_module)
