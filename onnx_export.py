"""
ONNX 导出脚本

支持所有网络模型，自动处理不同输出格式
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.onnx

from net import UNet, UNetV2, KeypointDetectorV2, KeypointDetectorV2_1, KeypointDetectorV2Heatmap, HeatmapResUNet
from network import HRNet, StackedHourglass, DARKpose, LEAP, OpenPose, SimpleBaseline, TransPose, UNetHeatmap


class ExportWrapper(torch.nn.Module):
    """包装器：统一输出格式

    OpenPose 返回 (pafs, heatmaps) 元组，ONNX 只能导出单个输出。
    此包装器提取 heatmaps 的最后一个 stage 作为唯一输出。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            # OpenPose: (pafs [B,nstack,paf,H,W], heatmaps [B,nstack,kp,H,W])
            return out[1][:, -1]  # 取最后一个 stage 的 heatmaps
        return out


if __name__ == '__main__':
    # ========== 配置 ==========
    model_type = 'OpenPose'  # 模型类型
    num_keypoints = 1
    weight_path = './param/OpenPose/exp1/best_model.pth'
    onnx_path = f'./onnx/{model_type}.onnx'
    # ==========================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model_map = {
        'UNet': UNet, 'UNetV2': UNetV2,
        'KeypointDetectorV2': KeypointDetectorV2,
        'KeypointDetectorV2_1': KeypointDetectorV2_1,
        'KeypointDetectorV2Heatmap': KeypointDetectorV2Heatmap,
        'HeatmapResUNet': HeatmapResUNet,
        'HRNet': HRNet, 'StackedHourglass': StackedHourglass,
        'DARKpose': DARKpose, 'LEAP': LEAP,
        'OpenPose': OpenPose, 'SimpleBaseline': SimpleBaseline,
        'TransPose': TransPose, 'UNetHeatmap': UNetHeatmap,
    }
    model_cls = model_map[model_type]
    model = model_cls(num_keypoints).to(device)

    # 加载权重
    if os.path.exists(weight_path):
        save_dict = torch.load(weight_path, map_location=device, weights_only=True)
        if isinstance(save_dict, dict) and 'model_state_dict' in save_dict:
            model.load_state_dict(save_dict['model_state_dict'])
        else:
            model.load_state_dict(save_dict)
        print(f"✅ 加载权重: {weight_path}")
    else:
        print(f"⚠️  未找到权重: {weight_path}，使用随机权重导出")

    # 包装模型（统一输出格式）
    wrapper = ExportWrapper(model).to(device)
    wrapper.eval()

    # 测试输出
    dummy = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"📊 模型: {model_type}")
    print(f"   输入: {dummy.shape}")
    print(f"   输出: {out.shape}")

    # 导出 ONNX
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    print(f"💾 已导出: {onnx_path}")
