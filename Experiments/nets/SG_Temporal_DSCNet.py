# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import dropout
from nets.DSConv import DSConv_pro
from nets.TemporalTrans import TemporalTransformer

"""Dynamic Snake Convolution Network"""

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x



def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)


    def forward(self, x):
        out = self.maxpool(x)

        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def  __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        # self.h_conv = DSConv_pro(in_channels, in_channels, kernel_size=9)
        # self.v_conv = DSConv_pro(in_channels, in_channels, kernel_size=9)



    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        # x_ = x.transpose(2, 3)
        # x_h = self.h_conv(x)
        # x_v = self.v_conv(x_).transpose(2, 3)

        return self.nConvs(x)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.apply(self._init_weights)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def local_conv(dim):
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

class Attention(nn.Module):
    def __init__(self, dim, mask, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio=sr_ratio
        if sr_ratio>1:
            if mask:
                # print('mask')
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                if self.sr_ratio==8:
                    f1, f2, f3 = 14*14, 56, 28
                elif self.sr_ratio==4:
                    f1, f2, f3 = 49, 14, 7
                elif self.sr_ratio==2:
                    f1, f2, f3 = 2, 1, None
                self.f1 = nn.Linear(f1, 1)
                self.f2 = nn.Linear(f2, 1)
                if f3 is not None:
                    self.f3 = nn.Linear(f3, 1)
            else:
                # print('no mask')
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim//2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        # self.apply(self._init_weights)


    def forward(self, x, H, W, mask):
        B, N, C = x.shape
        # print(x.shape)
        lepe = self.lepe_conv(
            self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)).view(B, C, -1).transpose(-1, -2) # B N C

        if self.sr_ratio > 1:
            if mask is None:
                # global
                q1 = self.q1(x).reshape(B//8, 8*N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)  # B H//2 N C//H
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # downsample [B, C, HW/sr^2] -> [B, HW/sr^2, C]
                x_1 = self.act(self.norm(x_1))
                kv1 = self.kv1(x_1).reshape(B//8, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]  # B H//2 Nkv C//H; Nkv=HW/sr^2


                attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # [Nq, C//H] @ [C//H, Nk] = [Nq, Nk]
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)  # B, H//2, Nq, Nkv
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)  # [Nq, Nk] @ [Nk, C//H] = [Nq, C//H] -> [B, N, C//2]

                global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) # mean of heads, mean of Nq
                global_mask_value = F.interpolate(global_mask_value.view(B,1,H//self.sr_ratio,W//self.sr_ratio),
                                                  (H, W), mode='nearest')[:, 0]  # B head Nq^1/2 Nq^1/2

                # local
                q2 = self.q2(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3) #B head N C
                kv2 = self.kv2(x_.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, -1, 2, self.num_heads // 2,
                                                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
                k2, v2 = kv2[0], kv2[1]
                q_window = 7
                window_size= 7
                q2, k2, v2 = window_partition(q2, q_window, H, W), window_partition(k2, window_size, H, W), \
                             window_partition(v2, window_size, H, W)
                attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
                # (B*numheads*num_windows, window_size*window_size, window_size*window_size)
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)

                x2 = (attn2 @ v2)  # B*numheads*num_windows, window_size*window_size, C   .transpose(1, 2).reshape(B, N, C)
                x2 = window_reverse(x2, q_window, H, W, self.num_heads // 2)

                local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads//2, H//window_size*W//window_size, window_size*window_size, window_size*window_size).mean(1), dim=2)
                local_mask_value = local_mask_value.view(B, H // window_size, W // window_size, window_size, window_size)
                local_mask_value=local_mask_value.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)  # B H=N^1/2 W

                # mask B H W
                x = torch.cat([x1, x2], dim=-1)  # cat at Channel dim
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                # cal mask
                mask = local_mask_value+global_mask_value
                mask_1 = mask.view(B, H * W)
                mask_2 = mask.permute(0, 2, 1).reshape(B, H * W)
                mask = [mask_1, mask_2]
            else:  # mask is not None
                q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # mask [local_mask global_mask]  local_mask [value index]  value [B, H, W]
                # use mask to fuse
                mask_1, mask_2 = mask
                mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)  # horizontal, order: small->large
                mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)  # vertical, order: small->large
                if self.sr_ratio == 8:
                    token1, token2, token3 = H * W // (14 * 14), H * W // 56, H * W // 28
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 4:
                    token1, token2, token3 =  H * W // 49, H * W // 14, H * W // 7
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 2:
                    token1, token2 = H * W // 2, H * W // 1
                    token1, token2 = token1 // 2, token2 // 2
                if self.sr_ratio==4 or self.sr_ratio==8:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C, less salient
                    p2 = torch.gather(x, 1, mask_sort_index1[:,  H * W // 4: H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))  # B N//2 C, medium salient
                    p3 = torch.gather(x, 1, mask_sort_index1[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))  # B N//4 C, most salient
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C

                    x_ = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)  # B H*W C -> B W*H C, vertical
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, : H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4: H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3_.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C
                elif self.sr_ratio==2:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, : H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                    x_ = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, : H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                kv1 = self.kv1(seq1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4) # kv B heads N C
                kv2 = self.kv2(seq2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                mask=None

        else:  # sr_ratio=1， original transformer
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            x = x.contiguous().view(B, N, C)

            x = self.proj(x+lepe)
            x = self.proj_drop(x)
            mask=None

        return x, mask

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.GroupNorm(1, b)#torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

def window_partition(x, window_size, H, W):
    B, num_heads, N, C = x.shape
    x = x.contiguous().view(B*num_heads, N, C).contiguous().view(B*num_heads, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C).\
        view(-1, window_size*window_size, C)
    return windows  #(B*numheads*num_windows, window_size**2, C)


def window_reverse(windows, window_size, H, W, head):
    Bhead = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(Bhead, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(Bhead, H, W, -1).view(Bhead//head, head, H, W, -1)\
        .contiguous().permute(0,2,3,1,4).contiguous().view(Bhead//head, H, W, -1).view(Bhead//head, H*W, -1)
    return x #(B, H, W, C)

class Block(nn.Module):

    def __init__(self, dim, mask, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, mask,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        # self.apply(self._init_weights)


    def forward(self, x, H, W, mask):
        x_, mask = self.attn(self.norm1(x), H, W, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x), H, W)

        return x, mask

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, hidden, n_patch = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))  # sqrt(196) = 14
        # x = x.permute(0, 2, 1)  # (B, n_patch, hidden) -> (B, hidden, n_patch)
        x = x.contiguous().view(B, hidden, h, w)  # (B, hidden, n_patch) -> (B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x) # upsample is bilinear interpolation
        out = self.conv(x)  # 1x1 conv
        out = self.norm(out)
        out = self.activation(out)
        return out

class SGFormerEncoder(nn.Module):
    def __init__(self, img_size=224, in_channel=1, embed_dims=64, num_heads=2,
                 mlp_ratios=4, depths=2, sr_ratio=4, pos_embed=True, without_mask=False, patch_embed_downscale=4):
        super().__init__()
        self.depths = depths
        if patch_embed_downscale == 4:
            self.num_patches = (img_size // 4) ** 2  # patch size 4x4
            self.patch_embed = nn.Sequential(
                Conv2d_BN(in_channel, embed_dims, 3, 2, 1),  # downsample 2x
                nn.GELU(),
                Conv2d_BN(embed_dims, embed_dims, 3, 1, 1),
                nn.GELU(),
                Conv2d_BN(embed_dims, embed_dims, 3, 2, 1),  # downsample 2x
            )
        elif patch_embed_downscale == 2:
            self.num_patches = (img_size // 2) ** 2  # patch size 2x2
            self.patch_embed = nn.Sequential(
                Conv2d_BN(in_channel, embed_dims, 3, 2, 1),
                nn.GELU(),
                Conv2d_BN(embed_dims, embed_dims, 3, 1, 1),
                nn.GELU(),
            )
        self.norm = nn.LayerNorm(embed_dims)
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dims))
        self.block = nn.ModuleList([Block(
            dim=embed_dims, mask=True if (i%2==1 and not without_mask) else False, num_heads=num_heads, mlp_ratio=mlp_ratios, sr_ratio=sr_ratio)
            for i in range(self.depths)])
        self.reconstruct = Reconstruct(embed_dims, embed_dims, 1, patch_embed_downscale)
    def forward(self, x):
        mask = None
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]
        x = self.norm(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.block:
            x, mask = blk(x, H, W, mask)
        x = self.norm(x)
        x = x.contiguous().view(B, C, H*W)
        x = self.reconstruct(x)
        return x


class SG_Temporal_DSCNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernel_size,
        extend_scope,
        if_offset,
        device,
        number,
        dim,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv0y = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv2y = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv4y = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        # self.conv60_1 = EncoderConv(8*self.number, 8*self.number)
        self.conv6x = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv6y = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv12y = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)


        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv14y = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv16y = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

        # self.temporal_Trans1 = SGFormerEncoder(img_size=224, in_channel=self.number, embed_dims=self.number, num_heads=2, depths=2, sr_ratio=4)
        # self.convFuse1 = EncoderConv(2 * self.number, self.number)
        # self.temporal_Trans3 = SGFormerEncoder(img_size=112, in_channel=2*self.number, embed_dims=2*self.number, num_heads=4, depths=4, sr_ratio=4)
        # self.convFuse3 = EncoderConv(4 * self.number, 2 * self.number)
        self.temporal_Trans5 = SGFormerEncoder(img_size=56, in_channel=4*self.number, embed_dims=4*self.number, num_heads=8, depths=4, sr_ratio=4)
        self.convFuse5 = EncoderConv(8 * self.number, 4 * self.number)
        self.temporal_Trans7 = SGFormerEncoder(img_size=28, in_channel=8*self.number, embed_dims=8*self.number, num_heads=16, depths=1, sr_ratio=1, patch_embed_downscale=2)
        self.convFuse7 = EncoderConv(16 * self.number, 8 * self.number)




    def forward(self, x):
        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))  # 224
        # x_0_2 = self.temporal_Trans1(x_0_1)
        # x_0_1 = self.convFuse1(torch.cat([x_0_1, x_0_2], dim=1))

        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))   # 112
        # x_1_2 = self.temporal_Trans3(x_1_1)
        # x_1_1 = self.convFuse3(torch.cat([x_1_1, x_1_2], dim=1))

        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1))  # 4*number  56
        x_2_2 = self.temporal_Trans5(x_2_1)
        x_2_1 = self.convFuse5(torch.cat([x_2_1, x_2_2], dim=1))


        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1))  # 8*number  28
        x_3_2 = self.temporal_Trans7(x_3_1)
        x_3_f = self.convFuse7(torch.cat([x_3_1, x_3_2], dim=1))


        # block4
        x = self.up(x_3_f)
        x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(torch.cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(torch.cat([x, x_2_1], dim=1))
        x_2_3 = self.conv13(torch.cat([x_120_2, x_12x_2, x_12y_2], dim=1))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(torch.cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(torch.cat([x, x_1_1], dim=1))
        x_1_3 = self.conv15(torch.cat([x_140_2, x_14x_2, x_14y_2], dim=1))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(torch.cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(torch.cat([x, x_0_1], dim=1))
        x_0_3 = self.conv17(torch.cat([x_160_2, x_16x_2, x_16y_2], dim=1))
        x = self.dropout(x)
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)

        return out

