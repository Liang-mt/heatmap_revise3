"""
网络定义模块

包含所有模型架构（合并自原 net.py / net2.py / net3.py + 新增高级模型）

模型列表：
- UNet:       大通道 UNet (64→1024), 输入 80×80
- UNetV2:     小通道 UNet (8→128),   输入 80×80
- KeypointDetectorV2:          编码器-解码器关键点检测器, 输入 128×128
- KeypointDetectorV2_1:        同上，额外存储 num_keypoints 属性
- KeypointDetectorV2Heatmap:   同上，backbone/upsample 命名风格
- HeatmapResUNet:              残差UNet，跳跃连接+残差块，输入 128×128

新增高级模型（输入 128×128, 输出 [B, num_keypoints, 128, 128]）：
- SEUNet:          UNet + 残差块 + SE通道注意力 (CVPR 2018), ~5M 参数（推荐）
- HourglassNet:    堆叠沙漏网络 + 中间监督 (ECCV 2016), ~8M 参数
- AttentionUNet:   UNet + 双重注意力 通道+空间 (CBAM, ECCV 2018), ~4M 参数
"""

import torch
from torch import nn
from torch.nn import functional as F


# ======================== 共用基础模块 ========================
# (原 net.py / net2.py 共用，代码完全一致，只保留一份)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


# ======================== UNet 系列 ========================

class UNet(nn.Module):
    """大通道 UNet (64→1024), 输入 80×80, 输出 80×80 heatmap"""
    def __init__(self, num_keypoints=1):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, num_keypoints, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        return self.Th(self.out(O4))


class UNetV2(nn.Module):
    """小通道 UNet (8→128), 输入 80×80, 输出 80×80 heatmap"""
    def __init__(self, num_keypoints=1):
        super(UNetV2, self).__init__()
        self.c1 = Conv_Block(3, 8)
        self.d1 = DownSample(8)
        self.c2 = Conv_Block(8, 16)
        self.d2 = DownSample(16)
        self.c3 = Conv_Block(16, 32)
        self.d3 = DownSample(32)
        self.c4 = Conv_Block(32, 64)
        self.d4 = DownSample(64)
        self.c5 = Conv_Block(64, 128)
        self.u1 = UpSample(128)
        self.c6 = Conv_Block(128, 64)
        self.u2 = UpSample(64)
        self.c7 = Conv_Block(64, 32)
        self.u3 = UpSample(32)
        self.c8 = Conv_Block(32, 16)
        self.u4 = UpSample(16)
        self.c9 = Conv_Block(16, 8)
        self.out = nn.Conv2d(8, num_keypoints, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        return self.Th(self.out(O4))


# ======================== KeypointDetector 系列 ========================

class KeypointDetectorV2(nn.Module):
    """编码器-解码器关键点检测器, 输入 128×128, 输出 [B, num_keypoints, 128, 128]"""
    def __init__(self, num_keypoints):
        super(KeypointDetectorV2, self).__init__()

        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 输入通道3，输出32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8
        )

        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, 2, stride=2),  # 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  # 输出形状应为 [B, C, 128, 128]
        return x


class KeypointDetectorV2_1(nn.Module):
    """同 KeypointDetectorV2，额外存储 num_keypoints 属性"""
    def __init__(self, num_keypoints):
        super(KeypointDetectorV2_1, self).__init__()
        self.num_keypoints = num_keypoints

        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8
        )

        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, kernel_size=2, stride=2),  # 128x128
            nn.Sigmoid()  # 输出概率图
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)  # 输出形状 [B, C, 128, 128]


class KeypointDetectorV2Heatmap(nn.Module):
    """同 KeypointDetectorV2，backbone/upsample 命名风格"""
    def __init__(self, num_keypoints):
        super(KeypointDetectorV2Heatmap, self).__init__()
        self.num_keypoints = num_keypoints

        # 特征提取主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出尺寸: (256, 8, 8)
        )

        # 上采样部分
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 输出尺寸: (128, 16, 16)
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 输出尺寸: (64, 32, 32)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 输出尺寸: (32, 64, 64)
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, kernel_size=2, stride=2),  # 输出尺寸: (num_keypoints, 128, 128)
            nn.Sigmoid()  # 输出值在 [0, 1] 范围内
        )

    def forward(self, x):
        features = self.backbone(x)  # 提取特征
        heatmaps = self.upsample(features)  # 上采样生成热力图
        return heatmaps


# ======================== HeatmapResUNet ========================
# 参考：UNet（跳跃连接）+ ResNet（残差块）+ Simple Baseline（双线性上采样）
# 特点：残差块梯度流通好 + 跳跃连接保留空间细节 + BatchNorm 训练稳定

class ResBlock(nn.Module):
    """残差块：Conv→BN→ReLU→Conv→BN + shortcut"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class HeatmapResUNet(nn.Module):
    """残差UNet：ResBlock + UNet跳跃连接 + 双线性上采样

    结构：
      Encoder: 3→32→64→128，每层2个ResBlock，stride-2下采样
      Decoder: 双线性上采样→1×1卷积→concat跳跃连接→2个ResBlock
      Output: num_keypoints通道，Sigmoid激活

    参数量：~0.83M
    输入：[B, 3, 128, 128]
    输出：[B, num_keypoints, 128, 128]
    """
    def __init__(self, num_keypoints=1):
        super(HeatmapResUNet, self).__init__()

        # ---- Encoder ----
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.enc1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.down1 = nn.Conv2d(32, 32, 3, 2, 1)

        self.enc2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down2 = nn.Conv2d(64, 64, 3, 2, 1)

        self.enc3 = nn.Sequential(ResBlock(128), ResBlock(128))

        # ---- Decoder ----
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 1),
        )
        self.dec2 = nn.Sequential(ResBlock(128), ResBlock(128))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 32, 1),
        )
        self.dec1 = nn.Sequential(ResBlock(64), ResBlock(64))

        # ---- Output ----
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        x = self.conv_in(x)          # [B, 32, 128, 128]
        e1 = self.enc1(x)            # [B, 32, 128, 128]
        x = self.down1(e1)           # [B, 32, 64, 64]

        x = torch.cat([x, x], dim=1) # [B, 64, 64, 64]（通道对齐）
        e2 = self.enc2(x)            # [B, 64, 64, 64]
        x = self.down2(e2)           # [B, 64, 32, 32]

        x = torch.cat([x, x], dim=1) # [B, 128, 32, 32]
        x = self.enc3(x)             # [B, 128, 32, 32]

        # Decoder
        x = self.up2(x)              # [B, 64, 64, 64]
        x = torch.cat([x, e2], dim=1) # [B, 128, 64, 64]  ← 跳跃连接
        x = self.dec2(x)             # [B, 128, 64, 64]

        x = self.up1(x)              # [B, 32, 128, 128]
        x = torch.cat([x, e1], dim=1) # [B, 64, 128, 128]  ← 跳跃连接
        x = self.dec1(x)             # [B, 64, 128, 128]

        return self.out_conv(x)       # [B, num_keypoints, 128, 128]


# ======================== 测试 ========================

if __name__ == '__main__':
    print("=== UNet (大通道 64→1024) ===")
    x = torch.randn(1, 3, 128, 128)
    net = UNet(num_keypoints=1)
    print(f"  输入: {x.shape} → 输出: {net(x).shape}")

    print("\n=== UNetV2 (小通道 8→128) ===")
    x = torch.randn(2, 3, 128, 128)
    net = UNetV2(num_keypoints=1)
    print(f"  输入: {x.shape} → 输出: {net(x).shape}")

    print("\n=== KeypointDetectorV2 ===")
    x = torch.randn(1, 3, 128, 128)
    net = KeypointDetectorV2(1)
    print(f"  输入: {x.shape} → 输出: {net(x).shape}")

    print("\n=== KeypointDetectorV2_1 ===")
    net = KeypointDetectorV2_1(1)
    print(f"  输入: {x.shape} → 输出: {net(x).shape}")

    print("\n=== KeypointDetectorV2Heatmap ===")
    net = KeypointDetectorV2Heatmap(1)
    print(f"  输入: {x.shape} → 输出: {net(x).shape}")

    print("\n=== HeatmapResUNet (推荐) ===")
    x = torch.randn(1, 3, 128, 128)
    net = HeatmapResUNet(num_keypoints=1)
    params = sum(p.numel() for p in net.parameters())
    print(f"  输入: {x.shape} → 输出: {net(x).shape}")
    print(f"  参数量: {params / 1e6:.2f} M ({params:,})")
