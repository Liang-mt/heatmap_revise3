import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from net import UNet, UNetV2, KeypointDetectorV2, KeypointDetectorV2_1, KeypointDetectorV2Heatmap, HeatmapResUNet
from network import HRNet, StackedHourglass, DARKpose, LEAP, OpenPose, SimpleBaseline, TransPose, UNetHeatmap
from data import transform
import time


def cv2Img(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def cpu_inference():
    # CPU优化设置
    device = torch.device("cpu")
    torch.set_num_threads(4)  # 根据CPU核心数调整
    torch.set_flush_denormal(True)  # 刷新非正规数

    # 模型加载
    net = KeypointDetectorV2(num_keypoints=1)
    weights = './param/KeypointDetectorV2/exp1/best_model.pth'

    if os.path.exists(weights):
        save_dict = torch.load(weights, map_location=device, weights_only=True)
        if isinstance(save_dict, dict) and 'model_state_dict' in save_dict:
            net.load_state_dict(save_dict['model_state_dict'])
        else:
            net.load_state_dict(save_dict)
        print('成功加载权重')
    else:
        raise FileNotFoundError("权重文件未找到")

    net.eval()

    # 预热
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        for _ in range(10):
            _ = net(dummy_input)

    os.makedirs('test_result', exist_ok=True)

    # 测量参数
    num_runs = 50  # CPU测试次数可适当减少
    timings = []

    for j in os.listdir('test_image'):
        # 准备数据
        img_path = os.path.join('test_image', j)
        img = Image.open(img_path).convert('RGB')
        img_data = transform(img).unsqueeze(0)

        # 精确推理
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter_ns()

                out = net(img_data)

                end = time.perf_counter_ns()
                timings.append((end - start) / 1e6)  # 转换为毫秒

        # 后处理
        if isinstance(out, tuple):
            out = out[1][:, -1]

        out = out.squeeze().cpu()

        # 在原始分辨率取 argmax，再缩放到原图尺寸（避免插值误差）
        hm_h, hm_w = out.shape[-2], out.shape[-1]
        idx = out.argmax().item()
        hy, hx = divmod(idx, hm_w)
        orig_img = cv2.imread(img_path)
        scale_x = orig_img.shape[1] / hm_w
        scale_y = orig_img.shape[0] / hm_h
        x, y = int(hx * scale_x), int(hy * scale_y)
        cv2.circle(orig_img, (x, y), 2, (0, 0, 255), -1)
        cv2.imwrite(f'test_result/{j}', img_np)

    # 统计结果
    timings = torch.tensor(timings)
    print(f"\n=== CPU推理统计 ===")
    print(f"平均时间: {timings.mean():.2f}±{timings.std():.2f}ms")
    print(f"最快: {timings.min():.2f}ms | 最慢: {timings.max():.2f}ms")
    print(f"总测试次数: {len(timings)}次")
    print(f"使用线程数: {torch.get_num_threads()}")


if __name__ == '__main__':
    cpu_inference()