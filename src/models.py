"""Models.

UNet / UNetDeepSup: from the original notebook (legacy/train.ipynb, Hyeonwook Kim).

Cleaned up without changing the architecture: removed a no-op line and an unused
'bridge' layer. Down-sampling and up-sampling are done with bilinear interpolation,
and skip connections are added (not concatenated), as in the original.

SwinUNet: Swin Transformer encoder (ImageNet-pretrained, via timm) with a
convolutional U-Net decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1),
                nn.InstanceNorm2d(out_channels),
                nn.PReLU()))

    def forward(self, x, skip=None, upscale=False):
        _, _, H, W = x.shape
        if skip is not None:
            x = x + skip
        for i, conv in enumerate(self.convs):
            x = conv(x) if i == 0 else x + conv(x)      # residual after the first conv
        size = (H * 2, W * 2) if upscale else (H // 2, W // 2)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class UNet(nn.Module):
    """Single output head. Input size must be divisible by 32."""

    def __init__(self, in_channels=3, up_layers=(1024, 512, 256, 128, 64, 64)):
        super().__init__()
        down_layers = [in_channels, 64, 128, 256, 512, 1024]
        self.down = nn.ModuleList(
            [ConvBlock(down_layers[i], down_layers[i + 1]) for i in range(4)]
            + [ConvBlock(down_layers[4], down_layers[5], layers=1)])
        self.up = nn.ModuleList(
            [ConvBlock(up_layers[0], up_layers[1], layers=1)]
            + [ConvBlock(up_layers[i], up_layers[i + 1]) for i in range(1, len(up_layers) - 1)])
        self.head = nn.Conv2d(up_layers[-1], 1, 3, 1, 1)

    def features(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
        outs = []
        for u in self.up:
            x = u(x, skips.pop(), upscale=True)
            outs.append(x)
        return outs

    def forward(self, x):
        return self.head(self.features(x)[-1])


class UNetDeepSup(UNet):
    """Five output heads, one per decoder level (coarse to fine), all upsampled to the
    input size. Train with DeepSupervisionLoss; predict with the last (finest) head."""

    def __init__(self, in_channels=3):
        up_layers = (1024, 512, 256, 128, 64, 32)
        super().__init__(in_channels, up_layers)
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, 3, 1, 1) for c in up_layers[1:]])
        del self.head

    def forward(self, x):
        size = x.shape[-2:]
        return [h(F.interpolate(f, size=size, mode="bilinear", align_corners=False))
                for h, f in zip(self.heads, self.features(x))]


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SwinUNet(nn.Module):
    """Swin-Tiny encoder (4 stages, strides 4/8/16/32) + U-Net decoder with concatenated skips.
    Use img_size 224 or 448: every stage's feature map must divide evenly into 7x7 windows."""

    def __init__(self, in_channels=3, img_size=448, pretrained=True,
                 name="swin_tiny_patch4_window7_224"):
        super().__init__()
        import timm
        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True,
                                         img_size=img_size, in_chans=in_channels)
        self.chs = self.encoder.feature_info.channels()          # [96, 192, 384, 768]
        self.up3 = UpBlock(self.chs[3], self.chs[2], 256)
        self.up2 = UpBlock(256, self.chs[1], 128)
        self.up1 = UpBlock(128, self.chs[0], 64)
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, 1))
        # ImageNet normalisation for the RGB channels (pretrained weights expect it)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], dim=1)
        feats = self.encoder(x)
        # some timm versions return Swin features channels-last (B, H, W, C)
        feats = [f if f.shape[1] == c else f.permute(0, 3, 1, 2).contiguous()
                 for f, c in zip(feats, self.chs)]
        x = self.up3(feats[3], feats[2])
        x = self.up2(x, feats[1])
        x = self.up1(x, feats[0])
        return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)
