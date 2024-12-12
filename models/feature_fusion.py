from models.experimental import *
import torch.nn.functional as F



class FpnFeatureFusion(nn.Module):

    def __init__(self, input_layers_list=[128, 256, 512]):
        super(FpnFeatureFusion, self).__init__()
        # self.resize = lambda x, s: F.interpolate(
        #     x, size=s, mode="bilinear", align_corners=True)
        # self.feature_list = []

        self.feature_list = nn.ModuleList()
        for output_layer in input_layers_list:
            for input_layer in input_layers_list:
                if (input_layer != output_layer):
                    self.feature_list.append(SubRegions(input_layer, output_layer, kernel_size=3))
    def resize(self, x, s):
        return F.interpolate(x, size=s, mode="bilinear", align_corners=True)

    def forward(self, x):
        out = []
        out.append(x[0] +
                   self.resize(self.feature_list.__getitem__(0)(x[1]), x[0].shape[-2:]) +
                   self.resize(self.feature_list.__getitem__(1)(x[2]), x[0].shape[-2:]))
        out.append(x[1] +
                   self.resize(self.feature_list.__getitem__(2)(x[0]), x[1].shape[-2:]) +
                   self.resize(self.feature_list.__getitem__(3)(x[2]), x[1].shape[-2:]))
        out.append(x[2] +
                   self.resize(self.feature_list.__getitem__(4)(x[0]), x[2].shape[-2:]) +
                   self.resize(self.feature_list.__getitem__(5)(x[1]), x[2].shape[-2:]))
        return out


class SubRegions(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.gn = nn.GroupNorm(32, out_channels)
        # self.gn = nn.BatchNorm2d(out_channels)

        self.mask_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, padding=1)
        tau = 1.5
        ttau = math.tanh(tau)
        # self.gate_activate = lambda x: ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        # self.gate_activate = lambda x: torch.tanh(x).clamp(min=0)

    def gate_activate(self, x):
        tau = 1.5
        ttau = math.tanh(tau)
        # torch.tanh(x).clamp(min=0)
        # torch.sigmoid(x)
        return ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)

    def forward(self, x, masked_func=None):
        gate = self.mask_conv(x)
        gate = self.gate_activate(gate)

        x = self.conv(x)
        x = self.gn(x)
        # self.update_running_cost(gate)
        if masked_func is not None:
            data_input = masked_func(x, gate)
        # data, gate = self.encode(data_input, gate)
        # output, = self.decode(data * gate)
        output = x * gate
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
    fpn = FpnFeatureFusion(*[input_layers_list])
    x = []
    x.append(torch.rand(1, 128, 80, 80))
    x.append(torch.rand(1, 256, 40, 40))
    x.append(torch.rand(1, 512, 20, 20))
    res = fpn(x)
    print(type(res))
    for r in res:
        print(r.shape)
    a = 'FpnFeatureFusion'
    b = eval(a)
    print(b)
    print(b is FpnFeatureFusion)
    print(isinstance(b, FpnFeatureFusion))

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
