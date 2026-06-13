"""
TransPose（简化版）

参考：TransPose: Towards Efficient and Accurate Pose Estimation (ICCV 2021)
核心思想：CNN backbone + Transformer 自注意力，捕获全局关键点关系

输入: [B, 3, 128, 128]
输出: [B, num_keypoints, 16, 16]（1/8 分辨率）
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


class TransformerBlock(nn.Module):
    """Transformer 编码器块：自注意力 + FFN"""
    def __init__(self, d_model=256, nhead=8, dim_ff=1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, C]（batch_first=True）
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransPose(nn.Module):
    """CNN + Transformer 关键点检测

    结构：
      Backbone: 简化 ResNet-18 → 256ch @ 16×16
      Transformer: 2 层自注意力（全局关系建模）
      Head: 1×1 conv → num_keypoints
    """
    def __init__(self, num_keypoints=1, d_model=256, nhead=8, num_layers=2):
        super().__init__()

        # CNN backbone（简化 ResNet-18）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 128→32
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),   # 32→16
            BasicBlock(128, 256, stride=1),
            BasicBlock(256, 256),
        )  # 输出: 256ch @ 16×16

        # 投影到 Transformer 维度
        self.proj = nn.Conv2d(256, d_model, 1)

        # Transformer 编码器
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, nhead) for _ in range(num_layers)]
        )

        # 输出头
        self.head = nn.Sequential(
            nn.Conv2d(d_model, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B = x.shape[0]

        # CNN 特征提取
        feat = self.backbone(x)        # [B, 256, 16, 16]
        feat = self.proj(feat)          # [B, d_model, 16, 16]

        # 转为序列格式送入 Transformer
        H, W = feat.shape[2], feat.shape[3]
        feat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, d_model]
        feat = self.transformer(feat)              # [B, H*W, d_model]

        # 恢复空间格式
        feat = feat.permute(0, 2, 1).view(B, -1, H, W)  # [B, d_model, 16, 16]

        return self.head(feat)  # [B, num_keypoints, 16, 16]


if __name__ == "__main__":
    model = TransPose(num_keypoints=1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
