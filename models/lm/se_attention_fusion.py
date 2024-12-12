import torch

from models.experimental import *
import torch.nn.functional as F
from models.lm.deform_conv2 import DeformConv2d
from models.lm.non_local import NonLocalConv2d


class SeAttentionFusion(nn.Module):

    def __init__(self, input_layers_list=[128, 256, 512]):
        super(SeAttentionFusion, self).__init__()
        # self.resize = lambda x, s: F.interpolate(
        #     x, size=s, mode="bilinear", align_corners=True)
        # self.feature_list = []

        self.se_conv_list = nn.ModuleList()
        for input_layer in input_layers_list:
            self.se_conv_list.append(SeBlock(input_layer, input_layer))
            # self.non_local_conv_list.append(GlobalContextConv(input_layer, input_layer))  # 全局模块list

    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

    def forward(self, x):
        out = []
        x[0] = self.se_conv_list[0](x[0])
        x[1] = self.se_conv_list[1](x[1])
        x[2] = self.se_conv_list[2](x[2])

        return x


class SeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=1, stride=1, bias=None, modulation=False):
        super().__init__()
        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.g1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1, stride=stride)
        self.g2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1, stride=stride)
        self.bn = nn.ReLU()
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        g = self.globalpool(x)
        g1 = self.g1(g)
        g2 = self.g2(g)
        g = self.bn(torch.cat((g1, g2), 1))
        g = self.g(g)
        g = self.sigmod(g)
        x = x * g

        return x


if __name__ == '__main__':

    # x1 = torch.rand(2, 2, 2)
    # print(x1)
    # print(nn.AdaptiveAvgPool2d(1)(x1))
    print(' === se block === ')
    x = torch.rand(1, 128, 80, 80)
    se = SeBlock(128, 128)
    x = se(x)
    print(x.shape)
    print(' === se block === \n')

    input_layers_list = [128, 256, 512]
    fpn = SeAttentionFusion(*[input_layers_list])
    x = []
    x.append(torch.rand(1, 128, 80, 80))
    x.append(torch.rand(1, 256, 40, 40))
    x.append(torch.rand(1, 512, 20, 20))
    res = fpn(x)
    print(type(res))
    for r in res:
        print(r.shape)
