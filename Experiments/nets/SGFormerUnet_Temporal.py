import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
from .TemporalTrans_CrossAtten import TemporalTransformer_CrossAtten as TemporalTransformer
from .DSConv import DSConv_pro
# from nets.DeformableStripConv import DeformableStripConv




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

class upFusion(nn.Module):
    def __init__(self, in_channels, out_channels, activation='GeLU'):
        super(upFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )

    def forward(self, x, xt):
        x = torch.cat([x, xt], dim=1)
        return self.fusion(x)

class SEFusionModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEFusionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, cnn_features, vit_features):
        combined = torch.cat([cnn_features, vit_features], dim=1)
        b, c, _, _ = combined.size()
        y = self.avg_pool(combined).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        scaled = combined * y.expand_as(combined)  # expand_as: expand this tensor to the same size as other
        cnn_out, vit_out = torch.split(scaled, c // 2, dim=1)
        return cnn_out + vit_out

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

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

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//8, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, cnn_feat, temporal_feat):
        fused = torch.cat([cnn_feat, temporal_feat], dim=1)
        att_weights = self.attention(fused)
        fused = att_weights * fused
        out = self.conv_reduce(fused)
        return out

class TemporalConsistencyBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.temporal_conv(x)

class SGFormerUnet_Temporal(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        # self.number = 32
        # self.kernel_size = 9
        # self.extend_scope = 1.0
        # self.if_offset = True
        if n_classes != 1:
            self.n_classes = n_classes + 1
        # Question here
        self.maxpool = nn.MaxPool2d(2)
        self.SGe1 = SGFormerEncoder(img_size=224, in_channel=1, embed_dims=in_channels, num_heads=2, depths=2, sr_ratio=4)
        self.SGe2 = SGFormerEncoder(img_size=112, in_channel=in_channels, embed_dims=in_channels*2, num_heads=4, depths=4, sr_ratio=4)
        self.SGe3 = SGFormerEncoder(img_size=56, in_channel=in_channels*2, embed_dims=in_channels*4, num_heads=8, depths=4, sr_ratio=4)
        self.SGe4 = SGFormerEncoder(img_size=28, in_channel=in_channels*4, embed_dims=in_channels*8, num_heads=16, depths=1, sr_ratio=4)
        self.SGe5 = SGFormerEncoder(img_size=14, in_channel=in_channels*8, embed_dims=in_channels*8, num_heads=16, depths=1, sr_ratio=1, patch_embed_downscale=2)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)


        # in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.CNNdown1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.CNNdown2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.CNNdown3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.CNNdown4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)

        self.Fuse1 = FeatureFusionBlock(in_channels*2, in_channels)
        self.Fuse2 = FeatureFusionBlock(in_channels*4, in_channels*2)
        self.Fuse3 = FeatureFusionBlock(in_channels*8, in_channels*4)
        self.Fuse4 = FeatureFusionBlock(in_channels*16, in_channels*8)
        self.Fuse5 = FeatureFusionBlock(in_channels*16, in_channels*8)

        # self.Fuse1 = nn.Conv2d(in_channels*3, in_channels, kernel_size=(1, 1))
        # self.Fuse2 = nn.Conv2d(in_channels*6, in_channels*2, kernel_size=(1, 1))
        # self.Fuse3 = nn.Conv2d(in_channels*12, in_channels*4, kernel_size=(1, 1))
        # self.Fuse4 = nn.Conv2d(in_channels*24, in_channels*8, kernel_size=(1, 1))
        # self.Fuse5 = nn.Conv2d(in_channels*24, in_channels*8, kernel_size=(1, 1))

        # self.latent_fusion = nn.Conv2d(in_channels*16, in_channels*8, kernel_size=(1, 1))

        # self.upFus4 = upFusion(in_channels*16, in_channels*8)
        # self.upFus3 = upFusion(in_channels*8, in_channels*4)
        # self.upFus2 = upFusion(in_channels*4, in_channels*2)
        # self.upFus1 = upFusion(in_channels*2, in_channels)

        # self.SEFuse4 = SEFusionModule(in_channels*16)
        # self.SEFuse3 = SEFusionModule(in_channels*8)
        # self.SEFuse2 = SEFusionModule(in_channels*4)
        # self.SEFuse1 = SEFusionModule(in_channels*2)

        self.Tempo_gate1 = AttentionGate(in_channels, in_channels, in_channels)
        self.Tempo_gate2 = AttentionGate(in_channels*2, in_channels*2, in_channels*2)
        self.Tempo_gate3 = AttentionGate(in_channels*4, in_channels*4, in_channels*4)
        self.Tempo_gate4 = AttentionGate(in_channels*8, in_channels*8, in_channels*8)

        self.ViT_gate1 = AttentionGate(in_channels, in_channels, in_channels)
        self.ViT_gate2 = AttentionGate(in_channels*2, in_channels*2, in_channels*2)
        self.ViT_gate3 = AttentionGate(in_channels*4, in_channels*4, in_channels*4)
        self.ViT_gate4 = AttentionGate(in_channels*8, in_channels*8, in_channels*8)
        # self.ViT_gate5 = AttentionGate(in_channels*16, in_channels*16, in_channels*8)

        self.CNN_gate1 = AttentionGate(in_channels, in_channels, in_channels)
        self.CNN_gate2 = AttentionGate(in_channels*2, in_channels*2, in_channels*2)
        self.CNN_gate3 = AttentionGate(in_channels*4, in_channels*4, in_channels*4)
        self.CNN_gate4 = AttentionGate(in_channels*8, in_channels*8, in_channels*8)
        # self.CNN_gate5 = AttentionGate(in_channels*16, in_channels*16, in_channels*8)

        self.decoder4 = UpBlock(in_channels * 16, in_channels * 4, 2)
        self.decoder3 = UpBlock(in_channels * 8, in_channels * 2, 2)
        self.decoder2 = UpBlock(in_channels * 4, in_channels, 2)
        self.decoder1 = UpBlock(in_channels * 2, in_channels, 2)
        self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1, 1))


        # self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        # self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        # self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        # self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        # self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        x = x.float()

        x1 = self.inc(x) # 224x224x64  CNN branch
        x1_s = self.SGe1(x)  # 224x224x64 ViT branch
        x1 = self.Fuse1(x1, x1_s)  #todo

        x2 = self.CNNdown1(self.ViT_gate1(x1_s, x1)) # 112x112x128
        x2_s = self.SGe2(self.maxpool(self.CNN_gate1(x1, x1_s)))  # 112x112x128
        x2 = self.Fuse2(x2, x2_s)  #todo

        x3 = self.CNNdown2(self.ViT_gate2(x2_s, x2))  # 56x56x256
        x3_s = self.SGe3(self.maxpool(self.CNN_gate2(x2, x2_s)))  # 56x56x256
        x3 = self.Fuse3(x3, x3_s)  #todo


        x4 = self.CNNdown3(self.ViT_gate3(x3_s, x3))  # 28x28x512
        x4_s = self.SGe4(self.maxpool(self.CNN_gate3(x3, x3_s)))  # 28x28x512
        x4 = self.Fuse4(x4, x4_s)  #todo


        x5 = self.CNNdown4(self.ViT_gate4(x4_s, x4))  # 14x14x512  # todo: add temporal transformer
        x5_s = self.SGe5(self.maxpool(self.CNN_gate4(x4, x4_s)))  # 14x14x512
        x5 = self.Fuse5(x5, x5_s)  #todo

        # x5_ = self.ViT_gate5(x5_t, x5)  # 14x14x1024
        # x5_t_ = self.CNN_gate5(x5, x5_t)  # 14x14x1024
        # x5 = x5_ + x5_t_

        # x5 = self.latent_fusion(x5)

        x = self.decoder4(x5, x4)  # 28x28x512
        x = self.decoder3(x,  x3)  # 56x56x256
        x = self.decoder2(x, x2)  # 112x112x128
        x = self.decoder1(x, x1)  # 224x224x64

        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits

# if __name__ == '__main__':
#     input_img = torch.randn(4,1,224,224)
#     #check input dtype
#     # print(input_img.dtype)
#
#     SG_Encoder = SGFormerEncoder()
#
#     output = SG_Encoder(input_img)
#     #check output shape
#     print(output.shape)
