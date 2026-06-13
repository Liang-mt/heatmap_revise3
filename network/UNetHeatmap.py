"""
UNet Heatmap

参考：U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
核心思想：encoder-decoder + 跳跃连接，保留空间细节

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 128, 128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv→BN→ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetHeatmap(nn.Module):
    """标准 UNet：3 级 encoder-decoder + 跳跃连接

    Encoder: 64→128→256
    Decoder: ConvTranspose2d 上采样 + concat 跳跃连接
    Output: num_keypoints 通道，Sigmoid 激活
    """
    def __init__(self, num_keypoints=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128(up) + 128(skip)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)   # 64(up) + 64(skip)

        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # [B, 64, 128, 128]
        e2 = self.enc2(self.pool1(e1)) # [B, 128, 64, 64]
        e3 = self.enc3(self.pool2(e2)) # [B, 256, 32, 32]

        # Decoder
        d3 = self.up3(e3)              # [B, 128, 64, 64]
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # 跳跃连接
        d2 = self.up2(d3)              # [B, 64, 128, 128]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # 跳跃连接

        return self.out_conv(d2)       # [B, num_keypoints, 128, 128]


if __name__ == "__main__":
    model = UNetHeatmap(num_keypoints=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
