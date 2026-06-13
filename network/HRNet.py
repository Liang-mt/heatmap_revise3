"""
HRNet（简化版）

参考：Deep High-Resolution Representation Learning (CVPR 2019)
核心思想：多分辨率并行特征 + 跨分辨率融合

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 128, 128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """标准残差块（支持通道数变化）"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class HRNet(nn.Module):
    """简化 HRNet：多分辨率并行 + 跨分辨率融合

    Stage 1: 32ch @ 64×64
    Stage 2: 64ch @ 32×32
    Stage 3: 128ch @ 16×16
    Fusion: 全部上采样到 64×64 → concat → 1×1 conv → 上采样到 128×128
    """
    def __init__(self, num_keypoints=1, base_channel=32):
        super().__init__()

        # 初始下采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # 128→64

        # 多分辨率分支
        self.stage1 = self._make_stage(64, base_channel, 4)       # 32ch @ 64×64
        self.stage2 = self._make_stage(base_channel, base_channel * 2, 4)  # 64ch @ 32×32
        self.down2 = nn.MaxPool2d(2)
        self.stage3 = self._make_stage(base_channel * 2, base_channel * 4, 4)  # 128ch @ 16×16
        self.down3 = nn.MaxPool2d(2)

        # 跨分辨率融合：各分支上采样到 64×64 后 concat
        self.fuse1 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel * 4, 1, bias=False),
            nn.BatchNorm2d(base_channel * 4),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 4, 1, bias=False),
            nn.BatchNorm2d(base_channel * 4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(base_channel * 4, base_channel * 4, 1, bias=False),
            nn.BatchNorm2d(base_channel * 4),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

        # 输出头：concat(128×3=384ch) → 128ch → 上采样到 128×128 → num_keypoints
        self.head = nn.Sequential(
            nn.Conv2d(base_channel * 4 * 3, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
            nn.Conv2d(128, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def _make_stage(self, in_ch, out_ch, num_blocks):
        layers = [Residual(in_ch, out_ch)]
        for _ in range(1, num_blocks):
            layers.append(Residual(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)          # [B, 64, 64, 64]

        x1 = self.stage1(x)        # [B, 32, 64, 64]
        x2 = self.stage2(self.down2(x1))  # [B, 64, 32, 32]
        x3 = self.stage3(self.down3(x2))  # [B, 128, 16, 16]

        f1 = self.fuse1(x1)        # [B, 128, 64, 64]
        f2 = self.fuse2(x2)        # [B, 128, 64, 64]
        f3 = self.fuse3(x3)        # [B, 128, 64, 64]

        fused = torch.cat([f1, f2, f3], dim=1)  # [B, 384, 64, 64]
        return self.head(fused)     # [B, num_keypoints, 128, 128]


if __name__ == "__main__":
    model = HRNet(num_keypoints=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
