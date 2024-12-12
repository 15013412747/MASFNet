import torch

from models.experimental import *
import torch.nn.functional as F
from models.lm.deform_conv2 import DeformConv2d
from models.lm.non_local import NonLocalConv2d


# 利用mask机制采样
class SpatialAttentionMaskFusion(nn.Module):
    def __init__(self, input_layers_list=[128, 256, 512]):
        super().__init__()
        # self.w1 = 1
        # self.w2 = 1
        self.w1 = 0.5
        self.w2 = (1 - self.w1) / 2

        print("=== SpatialAttentionMaskFusion ==> ")
        print("funsion parameter w1 ==> ", self.w1)
        print("funsion parameter w2 ==> ", self.w2)

        self.feature_list = nn.ModuleList()
        for output_layer in input_layers_list:
            for input_layer in input_layers_list:
                # print(input_layer,output_layer)
                self.feature_list.append(MaskAtten(input_layer, output_layer, kernel_size=3))

    def forward(self, x):
        out = []


        out.append(self.w1 * self.feature_list[0](x[0]) +
                   self.w2 * self.resize(self.feature_list[1](x[1]), x[0].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[2](x[2]), x[0].shape[-2:]))

        out.append(self.w2 * self.resize(self.feature_list[3](x[0]), x[1].shape[-2:]) +
                   self.w1 * self.feature_list[4](x[1]) +
                   self.w2 * self.resize(self.feature_list[5](x[2]), x[1].shape[-2:]))

        out.append(self.w1 * self.resize(self.feature_list[6](x[0]), x[2].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[7](x[1]), x[2].shape[-2:]) +
                   self.w2 * self.feature_list[8](x[2]))

        return out

    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)


class MaskAtten(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(input_channel, 1, kernel_size=3, padding=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.bn(self.conv1(x))
        x_s = self.sigmod(self.conv2(x))
        x = x1 * x_s
        return x


# 利用可变形卷积采样
class SpatialAttentionFusionDeforbaleMask(nn.Module):

    def __init__(self, input_layers_list=[128, 256, 512]):
        super().__init__()
        # self.resize = lambda x, s: F.interpolate(
        #     x, size=s, mode="bilinear", align_corners=True)
        # self.feature_list = []

        # self.w1 = 1
        # self.w2 = 1
        self.w1 = 0.5
        self.w2 = (1 - self.w1) / 2

        print("=== SpatialAttentionFusionDeforbaleMask ==> ")
        print("funsion parameter w1 ==> ", self.w1)
        print("funsion parameter w2 ==> ", self.w2)

        self.feature_list = nn.ModuleList()
        for output_layer in input_layers_list:
            for input_layer in input_layers_list:
                # if (input_layer != output_layer):
                self.feature_list.append(DeformableMask(input_layer, output_layer, kernel_size=3))
                # self.feature_list.append(NonLocalConv2d(input_layer, output_layer, kernel_size=1))

    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

    def feature_funsion(self, x):
        pass

    def forward(self, x):

        # (0):
        # (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (1): SubRegions(
        # (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (2): SubRegions(
        # (conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (3): SubRegions(
        # (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (4): SubRegions(
        # (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (5): SubRegions(
        # (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (6): SubRegions(
        # (conv): Conv2d(128, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (7): SubRegions(
        # (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (8): SubRegions(
        # (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)

        out = []

        out.append(self.w1 * self.feature_list[0](x[0]) +
                   self.w2 * self.resize(self.feature_list[1](x[1]), x[0].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[2](x[2]), x[0].shape[-2:]))

        out.append(self.w2 * self.resize(self.feature_list[3](x[0]), x[1].shape[-2:]) +
                   self.w1 * self.feature_list[4](x[1]) +
                   self.w2 * self.resize(self.feature_list[5](x[2]), x[1].shape[-2:]))

        out.append(self.w1 * self.resize(self.feature_list[6](x[0]), x[2].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[7](x[1]), x[2].shape[-2:]) +
                   self.w2 * self.feature_list[8](x[2]))

        return out


class DeformableMask(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = DeformConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        self.sigmod = nn.Sigmoid()
        self.gn = nn.GroupNorm(32, out_channels)

    def gate_activate(self, x):
        tau = 1.5
        ttau = math.tanh(tau)
        # torch.tanh(x).clamp(min=0)
        # torch.sigmoid(x)
        return ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)

    def forward(self, x, masked_func=None):
        # print(111,x.shape)
        x1 = self.gn(self.conv(x))
        # print(222,x1.shape)
        x_s = self.sigmod(self.conv2(x1))
        # print(333,x_s.shape)
        x = x1 * x_s
        return x


# 利用可变形卷积采样
class SpatialAttentionDeformableFusion(nn.Module):

    def __init__(self, input_layers_list=[128, 256, 512]):
        super(SpatialAttentionDeformableFusion, self).__init__()
        # self.resize = lambda x, s: F.interpolate(
        #     x, size=s, mode="bilinear", align_corners=True)
        # self.feature_list = []

        self.w1 = 1
        self.w2 = 1
        self.w1 = 0.5
        self.w2 = (1 - self.w1) / 2

        print("=== SpatialAttentionDeformableFusion ==> ")
        print("funsion parameter w1 ==> ", self.w1)
        print("funsion parameter w2 ==> ", self.w2)

        self.feature_list = nn.ModuleList()
        for output_layer in input_layers_list:
            for input_layer in input_layers_list:
                # if (input_layer != output_layer):
                self.feature_list.append(SubRegions(input_layer, output_layer, kernel_size=3))
                # self.feature_list.append(NonLocalConv2d(input_layer, output_layer, kernel_size=1))

    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

    def feature_funsion(self, x):
        pass

    def forward(self, x):

        # (0):
        # (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (1): SubRegions(
        # (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (2): SubRegions(
        # (conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (3): SubRegions(
        # (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (4): SubRegions(
        # (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (5): SubRegions(
        # (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (6): SubRegions(
        # (conv): Conv2d(128, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (7): SubRegions(
        # (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)

        # (8): SubRegions(
        # (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)

        out = []

        out.append(self.w1 * self.feature_list[0](x[0]) +
                   self.w2 * self.resize(self.feature_list[1](x[1]), x[0].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[2](x[2]), x[0].shape[-2:]))

        out.append(self.w2 * self.resize(self.feature_list[3](x[0]), x[1].shape[-2:]) +
                   self.w1 * self.feature_list[4](x[1]) +
                   self.w2 * self.resize(self.feature_list[5](x[2]), x[1].shape[-2:]))

        out.append(self.w1 * self.resize(self.feature_list[6](x[0]), x[2].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[7](x[1]), x[2].shape[-2:]) +
                   self.w2 * self.feature_list[8](x[2]))

        return out


class SpatialAttention_old(nn.Module):

    def __init__(self, input_layers_list=[128, 256, 512]):
        super().__init__()
        # self.resize = lambda x, s: F.interpolate(
        #     x, size=s, mode="bilinear", align_corners=True)
        # self.feature_list = []

        self.w1 = 1
        self.w2 = 1
        self.w1 = 0.5
        self.w2 = (1 - self.w1) / 2

        print("=== SpatialAttention2 ==> ")
        print("funsion parameter w1 ==> ", self.w1)
        print("funsion parameter w2 ==> ", self.w2)

        self.feature_list = nn.ModuleList()
        for input_layer in input_layers_list:
            for output_layer in input_layers_list:
                if (input_layer != output_layer):
                    self.feature_list.append(SubRegions(input_layer, output_layer, kernel_size=3))

    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

    def forward(self, x):
        out = []

        # 0 128 256
        # 1 128 512
        # 2 256 128
        # 3 256 512
        # 4 512 128
        # 5 512 256

        out.append(self.w1 * x[0] +
                   self.w2 * self.resize(self.feature_list[2](x[1]), x[0].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[4](x[2]), x[0].shape[-2:]))
        out.append(self.w1 * x[1] +
                   self.w2 * self.resize(self.feature_list[0](x[0]), x[1].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[5](x[2]), x[1].shape[-2:]))
        out.append(self.w1 * x[2] +
                   self.w2 * self.resize(self.feature_list[1](x[0]), x[2].shape[-2:]) +
                   self.w2 * self.resize(self.feature_list[3](x[1]), x[2].shape[-2:]))

        return out


class SubRegions(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = DeformConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.gn = nn.GroupNorm(32, out_channels)
        # # self.gn = nn.BatchNorm2d(out_channels)
        #
        # self.mask_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, padding=1)
        # tau = 1.5
        # ttau = math.tanh(tau)
        # self.gate_activate = lambda x: ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        # self.gate_activate = lambda x: torch.tanh(x).clamp(min=0)

    def gate_activate(self, x):
        tau = 1.5
        ttau = math.tanh(tau)
        # torch.tanh(x).clamp(min=0)
        # torch.sigmoid(x)
        return ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)

    def forward(self, x, masked_func=None):
        # gate = self.mask_conv(x)
        # gate = self.gate_activate(gate)

        x = self.conv(x)
        x = self.gn(x)
        # self.update_running_cost(gate)
        # if masked_func is not None:
        #     data_input = masked_func(x, gate)
        # data, gate = self.encode(data_input, gate)
        # output, = self.decode(data * gate)
        # output = x * gate
        return x


class test_fpn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = test_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv3 = SubRegions(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        # self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        return x


class test_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        # self.mask_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    input_layers_list = [128, 256, 512]
    # fpn = SpatialAttentionFusion(*[input_layers_list])
    fpn = SpatialAttentionFusionDeforbaleMask(*[input_layers_list])

    print(fpn)
    x = []
    x.append(torch.rand(1, 128, 80, 80))
    x.append(torch.rand(1, 256, 40, 40))
    x.append(torch.rand(1, 512, 20, 20))
    res = fpn(x)
    print(type(res))
    for r in res:
        print(r.shape)
    a = 'DeformFeatureFusion'
    # # b = eval(a)
    # print(b)
    # print(b is SpatialAttentionFusion)
    # print(isinstance(b, SpatialAttentionFusion))
    exit()

    x = torch.rand(1, 3, 5, 5)
    test_fpn = test_fpn(3, 3, 3)
    x = test_fpn(x)
    print(x.shape)
    fff = lambda x: torch.tanh(x)
    fun = torch.tanh(x)

    print(fff(x))
    print(fun)
    input = torch.rand(2, 3, 5, 5)
    print(input)
    print()
    outputs = [x.view(x.shape[0] * 1, -1, *x.shape[2:]) for x in input]
    print(outputs)
    # for p in fpn.parameters():
    #     print(p)
    #     print(p.numel())
