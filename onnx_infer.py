"""
ONNX 推理脚本

支持所有网络模型的 ONNX 推理，自动处理不同输出分辨率
"""

import os
import sys
import time
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# 把 PyTorch 的 CUDA/cuDNN DLL 路径加到 PATH（让 ONNX Runtime 能找到）
import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if torch_lib not in os.environ.get('PATH', ''):
    os.environ['PATH'] = torch_lib + ';' + os.environ.get('PATH', '')

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def create_session(onnx_model_path):
    """创建 ONNX 推理会话（优先 GPU，失败则静默用 CPU）"""
    # 先尝试 CUDA，抑制错误输出
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': 0}],
        )
        sys.stderr.close()
        sys.stderr = old_stderr
        # 确认真的用了 CUDA
        if 'CUDAExecutionProvider' in session.get_providers():
            print(f"🖥️  Provider: CUDAExecutionProvider")
            return session
    except Exception:
        pass
    finally:
        try:
            sys.stderr.close()
        except:
            pass
        sys.stderr = old_stderr

    # 回退 CPU
    session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider'],
    )
    print(f"🖥️  Provider: CPUExecutionProvider")
    return session


def infer_onnx(session, image_path, output_path, input_size=128):
    """ONNX 推理单张图片"""
    # 准备输入
    img = Image.open(image_path).convert('RGB').resize((input_size, input_size))
    img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    # 推理
    start_time = time.time()
    output = session.run(None, {'input': input_data})
    end_time = time.time()

    inference_time = int((end_time - start_time) * 1000)

    # 后处理：在原始分辨率取 argmax，再缩放到原图尺寸
    out = output[0].squeeze()
    if out.ndim == 3:
        out = out[0]

    hm_h, hm_w = out.shape
    idx = out.argmax().item()
    hy, hx = divmod(idx, hm_w)

    orig_img = np.array(Image.open(image_path).convert('RGB'))
    orig_h, orig_w = orig_img.shape[:2]
    x = int(hx * orig_w / hm_w)
    y = int(hy * orig_h / hm_h)

    # 绘制结果
    img_cv = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    cv2.circle(img_cv, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(output_path, img_cv)

    print(f"  {os.path.basename(image_path)}: 关键点=({x},{y}) | "
          f"heatmap={hm_h}×{hm_w} | 推理={inference_time}ms")


if __name__ == '__main__':
    # ========== 配置 ==========
    onnx_model_path = './onnx/OpenPose.onnx'
    test_image_dir = './test_image'
    output_dir = './onnx_result'
    input_size = 128
    # ==========================

    os.makedirs(output_dir, exist_ok=True)

    # 创建 Session（只创建一次）
    session = create_session(onnx_model_path)

    # 预热
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    for _ in range(3):
        session.run(None, {'input': dummy})

    print(f"📦 模型: {onnx_model_path}")
    print(f"📁 输入: {test_image_dir}")
    print(f"📁 输出: {output_dir}")
    print("-" * 50)

    for filename in os.listdir(test_image_dir):
        img_path = os.path.join(test_image_dir, filename)
        out_path = os.path.join(output_dir, filename)
        infer_onnx(session, img_path, out_path)

    print(f"\n🏁 完成！结果保存在 {output_dir}/")
