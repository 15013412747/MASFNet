import torch
from torch import nn


class NonLocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=1, stride=1, bias=None, modulation=False):
        super().__init__()
        self.qconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.kconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.vconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        q = self.qconv(x)
        k = self.kconv(x)
        v = self.vconv(x)
        att = torch.softmax(q @ k, -1) * v
        # print('att.shape',att.shape)
        # print('x.shape',x.shape)
        return att


if __name__ == '__main__':
    _x1 = torch.ones(3, 3)
    print(_x1)
    _x2 = torch.ones(3, 3)
    print(_x2)
    print(_x1 @ _x2)
    print(_x1 * _x2)

    x = torch.rand(1, 3, 4, 4)
    conv4 = NonLocalConv2d(3, 128, kernel_size=3, padding=1)
    x = conv4(x)
    print(x.size())
    print(x.shape)
    pass
