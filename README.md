# 热力图关键点检测

基于热力图回归的关键点检测项目，支持 14 种网络架构。

## 项目简介

输入 128×128 RGB 图像，输出 128×128 高斯热力图，通过 argmax 提取关键点坐标。

适用于单关键点定位任务（如目标中心点检测、标记点定位等）。

## 目录结构

```
heatmap_revise2/
├── net.py                # 6 个自定义网络（UNet 系列 + KeypointDetector 系列 + HeatmapResUNet）
├── network/              # 8 个参考网络（HRNet / Hourglass / DARKpose / LEAP / OpenPose / SimpleBaseline / TransPose / UNetHeatmap）
│   ├── __init__.py
│   ├── HRNet.py
│   ├── Hourglass.py
│   ├── DARKpose.py
│   ├── LEAP.py
│   ├── OpenPose.py
│   ├── SimpleBaseline.py
│   ├── TransPose.py
│   └── UNetHeatmap.py
├── train.py              # 训练脚本（原始参数，支持 14 个模型）
├── train2.py             # 训练脚本（数据增强 + PCK 评估，支持 14 个模型）
├── train3.py             # 训练脚本（数据增强 + PCK 评估，仅 net.py 模型）
├── train4.py             # 训练脚本（原始参数，仅 net.py 模型）
├── predict.py            # GPU 推理
├── test.py               # CPU 推理 + 计时
├── onnx_export.py        # ONNX 导出
├── onnx_infer.py         # ONNX 推理（CUDA 加速）
├── data.py               # 数据集（支持数据增强）
├── heatmap_label.py      # 高斯热力图生成
├── utils.py              # 工具函数
├── show_augment.py       # 数据增强可视化 & 增强数据集生成
├── datasets/             # 数据集
│   ├── data_center_train.txt
│   ├── data_center_val.txt
│   ├── data_train/images/
│   └── data_val/images/
├── param/                # 模型权重（自动生成）
├── train_image/          # 训练可视化（自动生成）
└── test_result/          # 推理结果（自动生成）
```

## 环境要求

```
Python >= 3.8
PyTorch >= 2.0
torchvision
opencv-python
numpy
Pillow
onnxruntime-gpu  # 可选，ONNX GPU 推理
```

## 快速开始

### 1. 训练

```python
from train4 import Trainer

trainer = Trainer(
    model_type='KeypointDetectorV2',
    num_keypoints=1,
    epoch=500,
)
trainer.train()
```

### 2. 推理

修改 `predict.py` 顶部配置后运行：

```python
# predict.py 顶部
model_type = 'KeypointDetectorV2'
weights = './param/KeypointDetectorV2/exp1/best_model.pth'
```

```bash
python predict.py
```

### 3. ONNX 导出 & 推理

```bash
# 导出
python onnx_export.py

# 推理
python onnx_infer.py
```

## 支持的模型

### net.py — 自定义网络（6 个）

| 模型 | 参数量 | 输入 | 输出 | 说明 |
|------|--------|------|------|------|
| `UNet` | 32.08M | 80×80 | 80×80 | 大通道 UNet（64→1024） |
| `UNetV2` | 0.50M | 80×80 | 80×80 | 小通道 UNet（8→128） |
| `KeypointDetectorV2` | 0.56M | 128×128 | 128×128 | 编码器-解码器 |
| `KeypointDetectorV2_1` | 0.56M | 128×128 | 128×128 | 同上，存储 num_keypoints |
| `KeypointDetectorV2Heatmap` | 0.56M | 128×128 | 128×128 | backbone/upsample 命名 |
| `HeatmapResUNet` | 1.59M | 128×128 | 128×128 | 残差 UNet（推荐） |

### network/ — 参考网络（8 个）

| 模型 | 参数量 | 输出 | 说明 |
|------|--------|------|------|
| `HRNet` | 1.99M | 128×128 | 多分辨率并行特征融合 |
| `StackedHourglass` | 6.74M | 128×128 | 多级 encoder-decoder 堆叠 |
| `DARKpose` | 1.99M | 128×128 | HRNet + 亚像素解码 |
| `LEAP` | 0.13M | 128×128 | 轻量级（倒残差块） |
| `OpenPose` | 4.55M | 16×16 | 多阶段 PAF + Heatmap |
| `SimpleBaseline` | 15.37M | 128×128 | ResNet-18 + 反卷积头 |
| `TransPose` | 4.09M | 128×128 | CNN + Transformer |
| `UNetHeatmap` | 1.86M | 128×128 | 标准 UNet |

### 使用方式

```python
# 任意模型都可以通过 model_type 切换
trainer = Trainer(model_type='HRNet', num_keypoints=1, epoch=500)
trainer = Trainer(model_type='HeatmapResUNet', num_keypoints=1, epoch=500)
trainer = Trainer(model_type='OpenPose', num_keypoints=1, epoch=500)
```

## 训练脚本对比

| 脚本 | 模型来源 | 数据增强 | PCK 评估 | GT resize | 适用场景 |
|------|---------|---------|---------|-----------|---------|
| `train.py` | net.py + network/ | ❌ | ❌ | ✅ | 基础训练（全模型） |
| `train2.py` | net.py + network/ | ✅ | ✅ | ✅ | 优化训练（全模型） |
| `train3.py` | net.py | ✅ | ✅ | ❌ | 优化训练（128×128 模型） |
| `train4.py` | net.py | ❌ | ❌ | ❌ | 基础训练（128×128 模型） |

- **GT resize**：低分辨率网络（如 OpenPose 16×16）需要把 GT 下采样到网络分辨率
- **元组处理**：OpenPose 返回 (pafs, heatmaps) 元组，需要特殊处理

## 数据增强

在 `data.py` 中通过 `augment=True` 开启：

| 增强方式 | 概率 | 说明 |
|---------|------|------|
| 水平/垂直翻转 | 50% | 图像和坐标同步变换 |
| 旋转 ±15° | 50% | 图像和坐标同步变换 |
| 亮度/对比度 | 50% | alpha 0.8~1.2, beta ±20 |
| 高斯噪声 | 30% | std=10 |

可视化增强效果：

```bash
python show_augment.py
```

## 精确率评估（PCK）

训练过程中每轮计算：

| 指标 | 说明 |
|------|------|
| PCK@3px | 预测坐标与 GT 距离 ≤ 3 像素的比例 |
| PCK@5px | 预测坐标与 GT 距离 ≤ 5 像素的比例 |
| PCK@10px | 预测坐标与 GT 距离 ≤ 10 像素的比例 |
| Mean Error | 平均欧氏距离（像素） |

```
📊 精确率     | PCK@3= 62.5% | PCK@5= 83.3% | PCK@10= 95.8% | MeanErr=3.21px
```

## sigma 参数

`data.py` 中 `MyDataset` 的 `sigma` 参数控制高斯热力图的宽度：

| sigma | 128×128 峰值 | 下采样到 16×16 峰值 | 适用场景 |
|-------|------------|-------------------|---------|
| 5 | 1.000 | 0.550 | 仅 128×128 输出的模型 |
| 10 | 1.000 | 0.845 | 32×32 及以上 |
| **20** | 1.000 | **0.958** | 所有分辨率（默认） |

默认 sigma=20，适用于所有模型。修改 sigma 后需要重新训练。

## 模型保存

训练自动创建递增实验目录：

```
./param/{model_type}/exp1/best_model.pth   ← 最佳模型
./param/{model_type}/exp1/last_model.pth   ← 最新模型
./param/{model_type}/exp2/best_model.pth   ← 下一次训练
```

权重格式：

```python
{'model_state_dict': model.state_dict()}
```

加载权重（兼容新旧格式）：

```python
save_dict = torch.load('best_model.pth')
if isinstance(save_dict, dict) and 'model_state_dict' in save_dict:
    model.load_state_dict(save_dict['model_state_dict'])
else:
    model.load_state_dict(save_dict)
```

## ONNX 导出 & 推理

### 导出

修改 `onnx_export.py` 顶部配置：

```python
model_type = 'KeypointDetectorV2'
weight_path = './param/KeypointDetectorV2/exp1/best_model.pth'
```

```bash
python onnx_export.py
```

### 推理

```bash
python onnx_infer.py
```

输出：
```
🖥️  Provider: CUDAExecutionProvider
📦 模型: ./onnx/KeypointDetectorV2.onnx
--------------------------------------------------
  244.png: 关键点=(58,62) | heatmap=128×128 | 推理=5ms
```

## 坐标提取

所有推理脚本使用相同策略：**原始分辨率取 argmax + 缩放**（避免插值误差）。

```python
# 在网络原生分辨率取 argmax
hm_h, hm_w = out.shape[-2], out.shape[-1]
idx = out.argmax().item()
hy, hx = divmod(idx, hm_w)

# 缩放到原图尺寸
x = int(hx * orig_width / hm_w)
y = int(hy * orig_height / hm_h)
```

| 网络输出分辨率 | 精度级别 |
|--------------|---------|
| 128×128 | 1 像素 |
| 64×64 | 2 像素 |
| 32×32 | 4 像素 |
| 16×16 | 8 像素 |

## 参考项目

- UNet: U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
- HRNet: Deep High-Resolution Representation Learning (CVPR 2019)
- Stacked Hourglass: Stacked Hourglass Networks for Human Pose Estimation (ECCV 2016)
- Simple Baseline: Simple Baselines for Human Pose Estimation (ECCV 2018)
- DARKpose: Distilling the Knowledge of Lightweight Pose Estimation (CVPR 2020)
- OpenPose: OpenPose: Realtime Multi-Person 2D Pose Estimation (TPAMI 2019)
- TransPose: TransPose: Towards Efficient and Accurate Pose Estimation (ICCV 2021)
- LEAP: LEAP: Lightweight Efficient Accurate Pose (CVPR 2019)
