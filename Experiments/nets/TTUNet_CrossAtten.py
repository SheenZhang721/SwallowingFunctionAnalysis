import torch.nn as nn
import torch
import torchvision.models as models
from .TemporalTrans_CrossAtten import TemporalTransformer_CrossAtten as TemporalTransformer

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

        self.up = nn.Upsample(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class TFusion(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(TFusion, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.activation = get_activation(activation)
        self.fusion = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1)),
            # nn.BatchNorm2d(in_channels),
            # get_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )

    def forward(self, x, xt):
        x = torch.cat([x, xt], dim=1)
        return self.fusion(x)

class Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):  # F_g for gating signal, F_l for local signal, F_int for intermediate
        super(Attention_Gate, self).__init__()
        self.W_g = nn.Sequential(  # gating signal
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_l = nn.Sequential(  # local signal
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)   # gating signal from down sampled input
        x1 = self.W_l(x)  # local signal from the up sampled input
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.up(x)
#
# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)

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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Resize skip connection if needed
            if x.shape != skip.shape:
                skip = nn.functional.interpolate(skip, size=x.shape[2:],
                                              mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class TTUNet_CrossAtten(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        if n_classes != 1:
            self.n_classes = n_classes + 1
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.TT1 = TemporalTransformer(224, in_channels, 16)
        self.fuse1 = TFusion(in_channels*2, in_channels, activation='GeLU')
        # self.SE1 = SEFusionModule(in_channels*2, reduction=4)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.TT2 = TemporalTransformer(112, in_channels*2, 8)
        self.fuse2 = TFusion(in_channels*4, in_channels*2, activation='GeLU')
        # self.SE2 = SEFusionModule(in_channels*4, reduction=4)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.TT3 = TemporalTransformer(56, in_channels*4, 4)
        self.fuse3 = TFusion(in_channels*8, in_channels*4, activation='GeLU')
        # self.SE3 = SEFusionModule(in_channels*8, reduction=4)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.TT4 = TemporalTransformer(28, in_channels*8, 2)
        self.fuse4 = TFusion(in_channels*16, in_channels*8, activation='GeLU')
        # self.SE4 = SEFusionModule(in_channels*16, reduction=4)
        self.down4 = DownBlock(in_channels*8, in_channels*16, nb_Conv=2)
        self.latent_fusion = nn.Conv2d(in_channels*16, in_channels*8, kernel_size=(1, 1))
        # self.upFus4 = upFusion(in_channels*16, in_channels*8, activation='GeLU')
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        # self.upFus3 = upFusion(in_channels*8, in_channels*4, activation='GeLU')
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        # self.upFus2 = upFusion(in_channels*4, in_channels*2, activation='GeLU')
        self.up2 = UpBlock(in_channels*4, in_channels*1, nb_Conv=2)
        # self.upFus1 = upFusion(in_channels*2, in_channels, activation='GeLU')
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1, 1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        xt1 = self.TT1(x1)
        x1 = self.fuse1(x1, xt1)
        # x1 = self.SE1(x1, xt1)
        # x1 = x1 + xt1
        x2 = self.down1(x1)
        xt2 = self.TT2(x2)
        x2 = self.fuse2(x2, xt2)
        # x2 = self.SE2(x2, xt2)
        # x2 = x2 + xt2
        x3 = self.down2(x2)
        xt3 = self.TT3(x3)
        x3 = self.fuse3(x3, xt3)
        # x3 = self.SE3(x3, xt3)
        # x3 = x3 + xt3
        x4 = self.down3(x3)
        xt4 = self.TT4(x4)
        x4 = self.fuse4(x4, xt4)
        # x4 = self.SE4(x4, xt4)
        # x4 = x4 + xt4
        x5 = self.down4(x4)

        x5 = self.latent_fusion(x5)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits

#
# class TTUNet_CrossAtten(nn.Module):
#     def __init__(self, n_channels=1, n_classes=1):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         if n_classes != 1:
#             self.n_classes = n_classes + 1
#         # Question here
#         in_channels = 64
#
#         resnet = models.resnet50(pretrained=False)
#
#         # Initial convolution to match input channels
#         self.initial_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.initial_conv.weight.data = resnet.conv1.weight.data[:, :n_channels, :, :]
#
#         # Encoder layers
#         self.firstconv = nn.Sequential(
#             self.initial_conv,
#             resnet.bn1,
#             resnet.relu
#         )
#         self.firstpool = resnet.maxpool
#         self.encoder1 = resnet.layer1  # 256 channels
#         self.encoder2 = resnet.layer2  # 512 channels
#         self.encoder3 = resnet.layer3  # 1024 channels
#         self.encoder4 = resnet.layer4  # 2048 channels
#
#         self.inc = ConvBatchNorm(n_channels, in_channels)
#         self.TT1 = TemporalTransformer(56, in_channels*4, 8)
#         self.TT2 = TemporalTransformer(28, in_channels * 8, 4)
#         self.TT3 = TemporalTransformer(14, in_channels * 16, 2)
#         self.TT4 = TemporalTransformer(7, in_channels * 32, 1)
#
#         self.fuse1 = TFusion(in_channels*8, in_channels*4, activation='GeLU')
#         self.fuse2 = TFusion(in_channels*16, in_channels*8, activation='GeLU')
#         self.fuse3 = TFusion(in_channels*32, in_channels*16, activation='GeLU')
#         self.fuse4 = TFusion(in_channels*64, in_channels*32, activation='GeLU')
#
#         # Decoder layers
#         self.decoder4 = DecoderBlock(2048, 1024)  # /16
#         self.decoder3 = DecoderBlock(1024, 512)  # /8
#         self.decoder2 = DecoderBlock(512, 256)  # /4
#         self.decoder1 = DecoderBlock(256, 64)  # /2
#
#         # Final upsampling to original size
#         self.final_upsample = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#
#         self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1, 1))
#         if n_classes == 1:
#             self.last_activation = nn.Sigmoid()
#         else:
#             self.last_activation = None
#
#     def _upsampling_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def _conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         # Question here
#         x = x.float()
#         x1 = self.firstconv(x)
#         x1 = self.firstpool(x1)  # B, 64, 56, 56
#         x2 = self.encoder1(x1)   # B, 256, 56, 56
#         xt2 = self.TT1(x2)
#         x2 = self.fuse1(x2, xt2)
#         x3 = self.encoder2(x2)   # B, 512, 28, 28
#         xt3 = self.TT2(x3)
#         x3 = self.fuse2(x3, xt3)
#         x4 = self.encoder3(x3)   # B, 1024, 14, 14
#         xt4 = self.TT3(x4)
#         x4 = self.fuse3(x4, xt4)
#         x5 = self.encoder4(x4)   # B, 2048, 7, 7
#         xt5 = self.TT4(x5)
#         x5 = self.fuse4(x5, xt5)
#
#         # Decoder path with skip connections and size matching
#         d4 = self.decoder4(x5, x4)  # /16
#         d3 = self.decoder3(d4, x3)  # /8
#         d2 = self.decoder2(d3, x2)  # /4
#         d1 = self.decoder1(d2, x1)  # /2
#
#         # Final upsampling to match input size
#         x = self.final_upsample(d1)  # Original size
#
#         if self.last_activation is not None:
#             logits = self.last_activation(self.outc(x))
#             # print("111")
#         else:
#             logits = self.outc(x)
#             # print("222")
#         # logits = self.outc(x) # if using BCEWithLogitsLoss
#         # print(logits.size())
#         return logits

