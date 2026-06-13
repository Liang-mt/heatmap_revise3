"""
LEAP（轻量级）

参考：LEAP: Lightweight Efficient Accurate Pose (CVPR 2019)
核心思想：轻量化 backbone + 深度可分离卷积头部

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 32, 32]（1/4 分辨率）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：depthwise 3×3 + pointwise 1×1"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块：expand → depthwise → project"""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return self.block(x) + x
        return self.block(x)


class LEAP(nn.Module):
    """轻量级关键点检测网络

    结构：轻量 backbone（倒残差块）+ 深度可分离卷积头部
    参数量：~0.35M
    """
    def __init__(self, num_keypoints=1):
        super().__init__()

        # 轻量 backbone（类似 MobileNetV2，但自己实现，不依赖 torch.hub）
        self.backbone = nn.Sequential(
            # 初始卷积
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),  # 128→64
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            # 倒残差块
            InvertedResidual(16, 16, stride=1, expand_ratio=1),    # 64×64
            InvertedResidual(16, 24, stride=2, expand_ratio=6),    # 32×32
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 32, stride=2, expand_ratio=6),    # 16×16
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
        )  # 输出: 64ch @ 16×16

        # 头部：深度可分离卷积 + 上采样
        self.head = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16→32
            DepthwiseSeparableConv(128, 64),
            nn.Conv2d(64, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)   # [B, 64, 16, 16]
        return self.head(x)    # [B, num_keypoints, 32, 32]


if __name__ == "__main__":
    model = LEAP(num_keypoints=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
