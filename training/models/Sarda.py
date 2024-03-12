import torch.nn as nn
class Sarda(nn.Module):
  def __init__(self):
    super(Sarda, self).__init__()
    self.layer1 = nn.Conv2d(1, 5, kernel_size=2, stride=2, padding=0)
    self.layer2 = nn.Linear(980, 100, bias=True)
    self.layer3 = nn.Linear(100, 10, bias=True)
    self.act = nn.ReLU()

  def forward(self, x):
    out = self.act(self.layer1(x))
    out = out.view(-1, 980)
    out = self.act(self.layer2(out))
    out = self.layer3(out)
    return out

  def output(self, x):
    out1 = self.act(self.layer1(x))
    out1 = out1.view(-1, 980)
    out2 = self.act(self.layer2(out1))
    out3 = self.layer3(out2)
    return out1, out2, out3