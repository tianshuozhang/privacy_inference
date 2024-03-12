import torch.nn as nn
class Minionn(nn.Module):
  def __init__(self):
    super(Minionn, self).__init__()
    self.layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
    self.layer2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0)
    self.layer3 = nn.Linear(256, 100, bias=True)
    self.layer4 = nn.Linear(100, 10, bias=True)

    self.act = nn.ReLU()
    self.pool = nn.MaxPool2d((2, 2))

  def forward(self, x):
    out = self.act(self.layer1(x))
    out = self.pool(out)
    out = self.act(self.layer2(out))
    out = self.pool(out)
    out = out.view(-1, 256)
    out = self.act(self.layer3(out))
    out = self.act(self.layer4(out))
    return out

  def output(self, x):
    out1 = self.act(self.layer1(x))
    out1 = self.pool(out1)
    out2 = self.act(self.layer2(out1))
    out2 = self.pool(out2)
    out2 = out2.view(-1, 256)
    out3 = self.act(self.layer3(out2))
    out4 = self.act(self.layer4(out3))    
    return out1, out2, out3, out4