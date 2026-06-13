"""
OpenPose（简化版）

参考：OpenPose: Realtime Multi-Person 2D Pose Estimation (TPAMI 2019)
核心思想：多阶段预测 PAF（部位亲和场）+ 热力图，逐步精化

输入: [B, 3, 128, 128]
输出: (pafs, heatmaps)
  pafs:      [B, num_stages, num_paf, 16, 16]
  heatmaps:  [B, num_stages, num_keypoints, 16, 16]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFeature(nn.Module):
    """简化的 VGG 特征提取器（前 10 层）

    3→64→64→pool→128→128→pool→256→256→256→256
    输入 128×128 → 输出 256ch @ 16×16
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64ch
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128→64
            # Block 2: 128ch
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64→32
            # Block 3: 256ch
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16
        )

    def forward(self, x):
        return self.features(x)  # [B, 256, 16, 16]


class StageBlock(nn.Module):
    """单阶段预测模块：PAF（回归）+ Heatmap（Sigmoid）"""
    def __init__(self, in_channels, num_keypoints, num_paf):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.paf_head = nn.Conv2d(128, num_paf, 1)
        self.heatmap_head = nn.Sequential(nn.Conv2d(128, num_keypoints, 1), nn.Sigmoid())

    def forward(self, x):
        feat = self.shared(x)
        return self.paf_head(feat), self.heatmap_head(feat)


class OpenPose(nn.Module):
    """简化 OpenPose

    Args:
        num_keypoints: 关键点数量
        num_paf: PAF 通道数（骨骼连接数 × 2，单人场景可设小值）
        num_stages: 预测阶段数
    """
    def __init__(self, num_keypoints=1, num_paf=2, num_stages=3):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_paf = num_paf
        self.num_stages = num_stages

        self.backbone = VGGFeature()  # 256ch @ 16×16

        self.stages = nn.ModuleList()
        pred_ch = num_keypoints + num_paf
        in_ch = 256
        for _ in range(num_stages):
            self.stages.append(StageBlock(in_ch, num_keypoints, num_paf))
            in_ch = in_ch + pred_ch  # 每次拼接后通道递增

    def forward(self, x):
        features = self.backbone(x)  # [B, 256, 16, 16]

        pafs = []
        heatmaps = []
        for i in range(self.num_stages):
            paf, heatmap = self.stages[i](features)

            if i < self.num_stages - 1:
                pred = torch.cat([paf, heatmap], dim=1)
                features = torch.cat([features, pred], dim=1)

            pafs.append(paf)
            heatmaps.append(heatmap)

        return torch.stack(pafs, dim=1), torch.stack(heatmaps, dim=1)


if __name__ == "__main__":
    model = OpenPose(num_keypoints=1, num_paf=2, num_stages=3)
    x = torch.randn(1, 3, 128, 128)
    pafs, heatmaps = model(x)
    print(f"输入: {x.shape}")
    print(f"PAFs: {pafs.shape}")       # [1, 3, 2, 16, 16]
    print(f"Heatmaps: {heatmaps.shape}")  # [1, 3, 1, 16, 16]
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
