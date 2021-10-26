import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import *
from models.experimental import *
from utils.autoanchor import  check_anchors
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr,fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync


try:
    import thop
except ImportError:
    thop = None
LOGGER = logging.getLogger(__name__)

class Detect(nn.Model):
    stride = None
    onnx_dynamic = False
    def __init__(self, nc = 80, anchors = () ,ch = (), inplace=True):
        super().__init__()
        self.nc = nc #number of cilass
        self.no = nc + 5 #number of outputsper anchor
        self.nl = len(anchors) #number of detection layer
        self.na = len(anchors[0]) // 2 #number of anchors
        self.grid = [torch.zeros(1)] * self.nl #init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)#shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))#shape(nl, 1, na, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no*self.na, 1) for x in ch) #output conv
        self.inplace = inplace #use in-place ops
    def forward(self, x):
        z = [] #inference out
        for i in range(self.nl):
            x[i] = self.m[i](x[i]) #conv
            bs, _, ny, nx = x[i].shape#x(bs, 255m 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:#inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2]*2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4]*2) ** 2 * self.anchor_grid[i] #wh
                else:
                    xy = (y[..., 0:2]*2. - 0.5 + self.grid[i])*self.stride[i]
                    wh = (y[..., 2:4]*2)**2*self.anchor_grid[i].view(1, self.na, 1, 1, 2)
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)
    @staticmethod
    def