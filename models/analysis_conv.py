from feature_fusion import *
from common import *
import numpy as np
import torch
import torch.nn as nn


def analysis_parameters(m: nn.Module):
    sum = 0
    for x in m.parameters():
        print(x.shape)
        # s = 1
        # for i in x.shape:
        #     s = i * s
        # print(s)
        p = x.numel()
        print(p)
        sum =sum + p
    return sum


if __name__ == '__main__':
    """test_conv"""
    test_conv = test_conv(512, 512, 3)
    # print([x.numel() for x in test_conv.parameters()])
    # np = sum(x.numel() for x in test_conv.parameters())  # number params
    # print(np)
    analysis_parameters(test_conv)
    input_layers_list = [128, 256, 512]

    """FpnFeatureFusion"""
    fpn = FpnFeatureFusion(*[input_layers_list])
    total = analysis_parameters(fpn)
    print(total)
    pass
