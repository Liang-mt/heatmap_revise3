"""
关键点检测推理脚本（GPU）

读取 test_image/ 目录下的图片，推理后将关键点标注保存到 test_result/
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from PIL import Image

from net import UNet, UNetV2, KeypointDetectorV2, KeypointDetectorV2_1, KeypointDetectorV2Heatmap, HeatmapResUNet
from network import HRNet, StackedHourglass, DARKpose, LEAP, OpenPose, SimpleBaseline, TransPose, UNetHeatmap
from data import transform


def cv2Img(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")

    # ========== 模型配置 ==========
    model_type = 'StackedHourglass'  # 可选: UNet, UNetV2, KeypointDetectorV2, ...
    num_keypoints = 1
    weights = './param/StackedHourglass/exp1/best_model.pth'  # 权重路径
    # ==============================

    # 初始化模型
    model_map = {
        'UNet': UNet,
        'UNetV2': UNetV2,
        'KeypointDetectorV2': KeypointDetectorV2,
        'KeypointDetectorV2_1': KeypointDetectorV2_1,
        'KeypointDetectorV2Heatmap': KeypointDetectorV2Heatmap,
        'HeatmapResUNet': HeatmapResUNet,
        'HRNet': HRNet,
        'StackedHourglass': StackedHourglass,
        'DARKpose': DARKpose,
        'LEAP': LEAP,
        'OpenPose': OpenPose,
        'SimpleBaseline': SimpleBaseline,
        'TransPose': TransPose,
        'UNetHeatmap': UNetHeatmap,
    }
    model_cls = model_map.get(model_type, KeypointDetectorV2)
    net = model_cls(num_keypoints).to(device)

    # 加载权重（兼容新旧两种格式）
    if os.path.exists(weights):
        save_dict = torch.load(weights, map_location=device, weights_only=True)
        if isinstance(save_dict, dict) and 'model_state_dict' in save_dict:
            net.load_state_dict(save_dict['model_state_dict'])
        else:
            net.load_state_dict(save_dict)
        print(f"✅ 成功加载权重: {weights}")
    else:
        print(f"⚠️  未找到权重文件: {weights}")
        sys.exit(1)

    net.eval()

    # 推理
    os.makedirs('test_result', exist_ok=True)

    for j in os.listdir('test_image'):
        img_path = os.path.join('test_image', j)
        img = Image.open(img_path).convert('RGB')
        img_data = transform(img).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            out = net(img_data)
        end_time = time.time()

        inference_time = int((end_time - start_time) * 1000)

        # OpenPose 返回 (pafs, heatmaps) 元组
        if isinstance(out, tuple):
            out = out[1][:, -1]  # 取最后一个 stage 的 heatmaps

        out = out.squeeze().cpu()

        # 在原始分辨率取 argmax，再缩放到原图尺寸（避免插值误差）
        hm_h, hm_w = out.shape[-2], out.shape[-1]
        idx = out.argmax().item()
        hy, hx = divmod(idx, hm_w)
        scale_x = img.width / hm_w
        scale_y = img.height / hm_h
        x = int(hx * scale_x)
        y = int(hy * scale_y)

        img_cv = cv2Img(img)
        cv2.circle(img_cv, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

        save_path = f'test_result/{j}'
        cv2.imwrite(save_path, img_cv)
        print(f"  {j}: 关键点=({x},{y}) | 推理={inference_time}ms")

    print(f"\n🏁 完成！结果保存在 test_result/")
