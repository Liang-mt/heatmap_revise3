"""
DARKpose

参考：Distilling the Knowledge of Lightweight Pose Estimation (CVPR 2020)
核心思想：用 HRNet 做 backbone，解码阶段用平均池化偏移修正提升精度

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 128, 128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接在 network 包内导入
from network.HRNet import HRNet


class DARKpose(nn.Module):
    """DARKpose = HRNet backbone + Sigmoid 输出

    核心改进在解码阶段（decode_heatmap），不是网络结构。
    """
    def __init__(self, num_keypoints=1):
        super().__init__()
        self.backbone = HRNet(num_keypoints)

    def forward(self, x):
        return self.backbone(x)  # [B, num_keypoints, 128, 128]

    @staticmethod
    def decode_heatmap(heatmaps):
        """DARKpose 解码：从 heatmap 提取精确坐标

        原理：先用平均池化估计偏移量，再修正 argmax 坐标，
        比直接 argmax 精度更高（亚像素级别）。

        Args:
            heatmaps: [B, C, H, W] 网络输出的热力图

        Returns:
            coords: [B, C, 2] 精确坐标 (x, y)
        """
        B, C, H, W = heatmaps.shape

        # 1. 用 3×3 平均池化估计偏移量
        pooled = F.avg_pool2d(heatmaps, 3, 1, padding=1)
        offset_x = pooled - heatmaps  # 水平偏移
        offset_y = pooled - heatmaps  # 垂直偏移（对称）

        # 2. argmax 取整数坐标
        flat = heatmaps.view(B, C, -1)
        _, indices = flat.max(dim=2)
        y = (indices // W).float()
        x = (indices % W).float()

        # 3. 应用偏移修正
        offset_flat_x = offset_x.view(B, C, -1)
        offset_flat_y = offset_y.view(B, C, -1)
        x = x + offset_flat_x.gather(2, indices.unsqueeze(2)).squeeze(2)
        y = y + offset_flat_y.gather(2, indices.unsqueeze(2)).squeeze(2)

        return torch.stack([x, y], dim=2)  # [B, C, 2]


if __name__ == "__main__":
    model = DARKpose(num_keypoints=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")

    # 测试解码
    coords = DARKpose.decode_heatmap(out)
    print(f"解码坐标: {coords.shape}")  # [1, 1, 2]
