import torch.nn as nn
class SecureML(nn.Module):
  def __init__(self):
    super(SecureML, self).__init__()
    self.layer1 = nn.Linear(784, 128, bias=True)
    self.layer2 = nn.Linear(128, 128, bias=True)
    self.layer3 = nn.Linear(128, 10, bias=True)
    # self.act = nn.Hardtanh()
    self.act = nn.ReLU()

  def forward(self, x):
    return self.layer3(self.act(self.layer2(self.act(self.layer1(x.view(-1, 784))))))

  def output(self, x):
    out1 = self.act(self.layer1(x))
    out2 = self.act(self.layer2(out1))
    out3 = self.layer3(out2)
    return out1, out2, out3