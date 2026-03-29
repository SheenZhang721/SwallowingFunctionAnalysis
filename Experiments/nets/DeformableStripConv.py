import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d

class DeformableStripConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, deformable_groups=1):
        super(DeformableStripConv, self).__init__()
        # calculate padding
        if padding is None:
            padding = (kernel_size[0] //2 , kernel_size[1] // 2)
            print("padding: ", padding)
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, deformable_groups=deformable_groups)
        self.offset_conv = nn.Conv2d(in_channels, 2 * deformable_groups * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        def forward(self, x):
            offsets = self.offset_conv(x)
            out = self.conv(x, offsets)
            return out