"""
Simple Baseline

参考：Simple Baselines for Human Pose Estimation (ECCV 2018)
核心思想：ResNet backbone + 反卷积头，简单但有效

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 32, 32]（1/4 分辨率）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet 基础残差块"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """简化 ResNet-18（不依赖 torch.hub）

    输入 128×128 → 输出 512ch @ 4×4
    """
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128→32
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 32×32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 16×16
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 8×8
        self.layer4 = self._make_layer(256, 512, 2, stride=2) # 4×4

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 512, 4, 4]


class SimpleBaseline(nn.Module):
    """Simple Baseline = ResNet-18 + 反卷积头

    反卷积头：3 层 ConvTranspose2d 逐步上采样 4×4 → 32×32
    """
    def __init__(self, num_keypoints=1):
        super().__init__()
        self.backbone = ResNet18()  # 512ch @ 4×4

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4→8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),  # 8→16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),  # 16→32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)   # [B, 512, 4, 4]
        x = self.deconv(x)     # [B, 256, 32, 32]
        return self.head(x)    # [B, num_keypoints, 32, 32]


if __name__ == "__main__":
    model = SimpleBaseline(num_keypoints=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
