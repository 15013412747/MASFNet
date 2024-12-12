# YOLOv5 🚀 by lm


import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.swintransformer import SwinStage, PatchMerging, PatchEmbed
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

from models.feature_fusion import FpnFeatureFusion, SubRegions, test_fpn
from models.lm.deform_feature_fusion import DeformFeatureFusion

from models.lm.global_attention_fusion import GlobalAttentionFusion
from models.lm.se_attention_fusion import SeAttentionFusion
from models.lm.spatial_attention_fusion import SpatialAttentionMaskFusion
from models.lm.spatial_attention_fusion import SpatialAttentionFusionDeforbaleMask
from models.lm.spatial_attention_fusion import SpatialAttentionDeformableFusion


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Segment:
    pass


class FusionDetect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        self.GlobalAttentionFusion = GlobalAttentionFusion(ch)
        self.SeAttentionFusion = SeAttentionFusion(ch)
        self.SpatialAttentionFusion = SpatialAttentionFusionDeforbaleMask(ch)  # 全局+通道+空间融合

        # self.SpatialAttentionFusion = SpatialAttentionMaskFusion(ch)  # 全局+通道+空间融合
        # self.SpatialAttentionFusion = SpatialAttentionDeformableFusion(ch)  # 全局+通道+空间融合

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        print("ch -->", ch, "self.nl -->", self.nl, "self.na -->", self.na, "anchors -->", anchors, "self.no -->",
              self.no)
        self.inplace = inplace  # use in00place ops (e.g. slice assignment)

    def forward(self, x):
        x = self.GlobalAttentionFusion(x)
        x = self.SeAttentionFusion(x)
        x = self.SpatialAttentionFusion(x)

        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        res = x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        return res

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


if __name__ == '__main__':
    input_layers_list = [128, 256, 512]
    anchor = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    fpn = FusionDetect(80, anchor, [128, 256, 512])
    x = []
    x.append(torch.rand(1, 128, 80, 80))
    x.append(torch.rand(1, 256, 40, 40))
    x.append(torch.rand(1, 512, 20, 20))
    # print((x[0].shape))

    res = fpn(x)
    print('type(res)', type(res))
    # print((res))

    for r in res:
        print(r.shape)
