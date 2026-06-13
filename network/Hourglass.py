"""
Stacked Hourglass Network

参考：Stacked Hourglass Networks for Human Pose Estimation (ECCV 2016)
核心思想：多级 encoder-decoder 堆叠，逐级精化预测

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 32, 32]（1/4 分辨率）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """瓶颈残差块：1×1→3×3→1×1"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity, inplace=True)


class Hourglass(nn.Module):
    """单个 Hourglass 模块（递归 encoder-decoder）"""
    def __init__(self, depth, features):
        super().__init__()
        self.depth = depth
        self.up1 = Residual(features, features)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(features, features)
        self.low2 = Hourglass(depth - 1, features) if depth > 1 else Residual(features, features)
        self.low3 = Residual(features, features)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.low1(self.pool(x))
        low3 = self.low3(self.low2(low1))
        return up1 + self.up2(low3)


class StackedHourglass(nn.Module):
    """堆叠 Hourglass 网络

    Args:
        nstack: 堆叠的 hourglass 数量
        num_keypoints: 关键点数量
        feat_channels: 中间特征通道数
    """
    def __init__(self, num_keypoints=1, nstack=2, feat_channels=256):
        super().__init__()
        self.nstack = nstack

        # 初始特征提取
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, feat_channels),
        )

        # 堆叠的 hourglass 模块
        self.hgs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        self.merge_features = nn.ModuleList()
        self.merge_preds = nn.ModuleList()

        for _ in range(nstack):
            self.hgs.append(Hourglass(depth=4, features=feat_channels))
            self.out_convs.append(nn.Sequential(
                Residual(feat_channels, feat_channels),
                nn.Conv2d(feat_channels, feat_channels, 1, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
            ))
            self.merge_features.append(nn.Conv2d(feat_channels, feat_channels, 1, bias=False))
            self.merge_preds.append(nn.Conv2d(num_keypoints, feat_channels, 1, bias=False))

        # 每个 stack 的输出预测
        self.pred_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 1, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channels, num_keypoints, 1),
                nn.Sigmoid(),
            ) for _ in range(nstack)
        ])

    def forward(self, x):
        x = self.pre(x)  # [B, 256, 32, 32]

        preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feat = self.out_convs[i](hg)
            pred = self.pred_convs[i](feat)
            preds.append(pred)

            if i < self.nstack - 1:
                x = x + self.merge_features[i](feat) + self.merge_preds[i](pred)

        return preds[-1]  # 返回最后一个 stack 的预测 [B, num_keypoints, 32, 32]


if __name__ == "__main__":
    model = StackedHourglass(num_keypoints=1, nstack=2)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
