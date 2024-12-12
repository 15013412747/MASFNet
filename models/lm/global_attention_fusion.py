from models.experimental import *
import torch.nn.functional as F
from models.lm.deform_conv2 import DeformConv2d
from models.lm.non_local import NonLocalConv2d


class GlobalAttentionFusion(nn.Module):

    def __init__(self, input_layers_list=[128, 256, 512]):
        super(GlobalAttentionFusion, self).__init__()
        # self.resize = lambda x, s: F.interpolate(
        #     x, size=s, mode="bilinear", align_corners=True)
        # self.feature_list = []

        self.non_local_conv_list = nn.ModuleList()
        for input_layer in input_layers_list:
            self.non_local_conv_list.append(GlobalContextConv(input_layer, input_layer))  # 全局模块list

    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

    def forward(self, x):
        out = []

        x[0] = self.non_local_conv_list[0](x[0])
        x[1] = self.non_local_conv_list[1](x[1])
        x[2] = self.non_local_conv_list[2](x[2])

        out = [x[0], x[1], x[2]]

        return out


class GlobalContextConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=1, stride=1, bias=None, modulation=False):
        super().__init__()
        self.kconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.w1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.w2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        k = self.kconv(x)
        _x = torch.softmax(k, -1) * x
        _x = self.w1(x)
        _x = self.bn(x) + x
        # _x = self.w2(x) + x
        # print('att.shape',att.shape)
        # print('x.shape',x.shape)
        return x


if __name__ == '__main__':
    input_layers_list = [128, 256, 512]
    fpn = GlobalAttentionFusion(*[input_layers_list])
    x = []
    x.append(torch.rand(1, 128, 80, 80))
    x.append(torch.rand(1, 256, 40, 40))
    x.append(torch.rand(1, 512, 20, 20))
    # print((x[0].shape))

    res = fpn(x)
    print(type(res))
    # print((res))

    for r in res:
        print(r.shape)
