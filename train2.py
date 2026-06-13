"""
关键点检测训练脚本（优化版）

基于 train.py 的有效参数，增加：
1. 数据增强 → 小数据集泛化更好
2. PCK 精确率评估 → 关键点检测标准指标
3. best/last 模型保存 → 自动递增实验目录
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import MyDataset
from net import UNet, UNetV2, KeypointDetectorV2, KeypointDetectorV2_1, KeypointDetectorV2Heatmap, HeatmapResUNet
from network import HRNet, StackedHourglass, DARKpose, LEAP, OpenPose, SimpleBaseline, TransPose, UNetHeatmap


class Trainer:
    """关键点检测训练器"""

    def __init__(self, weight=None, model_type='KeypointDetectorV2', num_keypoints=1,
                 epoch=500,
                 train_data_path=r'./datasets/data_center_train.txt',
                 val_data_path=r'./datasets/data_center_val.txt',
                 save_path='train_image',
                 save_dir='./param'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = weight
        self.model_type = model_type
        self.save_path = save_path
        self.save_dir = save_dir
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

        os.makedirs(save_path, exist_ok=True)

        # 数据集（训练集开启数据增强）
        self.train_data_size = len(MyDataset(train_data_path))
        self.test_data_size = len(MyDataset(val_data_path))

        self.train_dataloader = DataLoader(MyDataset(train_data_path, augment=True), batch_size=16, shuffle=True)
        self.test_dataloader = DataLoader(MyDataset(val_data_path, augment=False), batch_size=16, shuffle=True)

        # 模型
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
        self.net = model_cls(num_keypoints).to(self.device)

        total_params = sum(p.numel() for p in self.net.parameters())

        # 加载权重（兼容新旧两种格式）
        if weight and os.path.exists(weight):
            save_dict = torch.load(weight, weights_only=True)
            if isinstance(save_dict, dict) and 'model_state_dict' in save_dict:
                self.net.load_state_dict(save_dict['model_state_dict'])
            else:
                self.net.load_state_dict(save_dict)
            print(f"✅ 成功加载权重: {weight}")
        else:
            print(f"⚠️  未找到权重文件，从头开始训练")

        # 优化器 & 损失（与 train.py 一致）
        self.optimizer = optim.Adam(self.net.parameters())
        self.loss_fun = nn.BCELoss()

        # 训练参数
        self.epoch = epoch

        # 跟踪最佳 Mean Error（越低越好）
        self.best_mean_err = float('inf')

        # ========== 打印配置信息 ==========
        print("=" * 60)
        print(f"🖥️  使用设备: {self.device}")
        print(f"🏗️  模型结构: {model_type} (num_keypoints={num_keypoints})")
        print(f"⚙️  模型参数量: {total_params / 1e6:.2f} M ({total_params:,})")
        print(f"📦  batch_size: 16")
        print(f"📈  optimizer: Adam (lr=0.001)")
        print(f"📉  loss: BCELoss")
        print(f"🔄  总轮数: {self.epoch}")
        print(f"🎲  数据增强: 开启 (翻转/旋转/亮度/噪声)")
        print(f"📊  精确率评估: PCK@3/5/10px + Mean Error")
        print(f"💾  保存目录: {self.save_dir}/{model_type}/expN/")
        print("=" * 60)

    def train(self):
        # ========== 数据集信息 ==========
        print("\n📊 数据集信息")
        print("-" * 60)
        print(f"训练数据集: {self.train_data_size:>4d} 张 (每轮随机增强，每张图见4种不同变换)")
        print(f"测试数据集: {self.test_data_size:>4d} 张 (不增强)")
        print("-" * 60)

        for i in range(self.epoch):
            # ========== 轮次开始 ==========
            print(f"\n🚀 第 {i + 1} 轮训练开始")
            print("-" * 60)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"🔧 当前学习率: {current_lr:f}")
            print("-" * 60)

            # ---------- 训练 ----------
            self.net.train()
            running_loss = 0.0

            for number, (image, segment_image) in enumerate(self.train_dataloader):
                imgs, targets = image.to(self.device), segment_image.to(self.device)
                outputs = self.net(imgs)

                # OpenPose 返回 (pafs, heatmaps) 元组，取 heatmaps
                if isinstance(outputs, tuple):
                    outputs = outputs[1][:, -1]  # 取最后一个 stage 的 heatmaps

                # GT resize 到网络原生分辨率（不是把网络输出 resize 到 GT）
                if outputs.shape != targets.shape:
                    targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

                loss = self.loss_fun(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if (number + 1) % 5 == 0:
                    print(f"  {i + 1:2d}{number + 1:5d} | train_loss = {loss.item():>6.12f}")

                if number % 1 == 0:
                    _segment_image = targets[0]
                    _out_image = outputs[0]
                    img = torch.stack([_segment_image[0], _out_image[0]], dim=0)
                    img = torch.unsqueeze(img, dim=1)
                    try:
                        save_image(img, f'{self.save_path}/{number}.png', nrow=5)
                    except OSError:
                        pass

            # ---------- 验证 ----------
            avg_train_loss = running_loss / len(self.train_dataloader)
            print("-" * 60)
            print(f"📈 第 {i + 1} 轮训练 | 平均训练损失 = {avg_train_loss:>6.12f}")

            total_test_loss = 0
            all_distances = []

            self.net.eval()
            with torch.no_grad():
                for n, (image, segment_image) in enumerate(self.test_dataloader):
                    imgs, targets = image.to(self.device), segment_image.to(self.device)
                    outputs = self.net(imgs)

                    if isinstance(outputs, tuple):
                        outputs = outputs[1][:, -1]

                    if outputs.shape != targets.shape:
                        targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

                    loss = self.loss_fun(outputs, targets)
                    total_test_loss += loss.item()

                    distances = self._calc_keypoint_accuracy(outputs, targets)
                    all_distances.extend(distances)

            # 计算 PCK 指标
            distances_tensor = torch.tensor(all_distances)
            mean_err = distances_tensor.mean().item()
            pck3 = (distances_tensor <= 3).float().mean().item() * 100
            pck5 = (distances_tensor <= 5).float().mean().item() * 100
            pck10 = (distances_tensor <= 10).float().mean().item() * 100

            print(f"📝 第 {i + 1} 轮验证 | test_loss = {total_test_loss:>6.12f}")
            print(f"📊 精确率     | PCK@3={pck3:5.1f}% | PCK@5={pck5:5.1f}% | PCK@10={pck10:5.1f}% | MeanErr={mean_err:.2f}px")
            print("-" * 60)

            # ---------- 保存模型（best + last） ----------
            if mean_err < self.best_mean_err:
                self.best_mean_err = mean_err
                self._save_model('best')

            self._save_model('last')

        # ========== 训练完成 ==========
        print("\n" + "=" * 60)
        print("🏁 训练完成！")
        print(f"🏆 最佳 Mean Error: {self.best_mean_err:.2f} px")
        print(f"📂 模型保存在: {self._exp_dir}")
        print("=" * 60)

    def _calc_keypoint_accuracy(self, pred_heatmap, gt_heatmap):
        """从 heatmap 提取关键点坐标，计算欧氏距离

        在各 heatmap 原始分辨率取 argmax，再统一缩放到 GT 尺寸比较
        """
        distances = []
        pred = pred_heatmap.detach()
        gt = gt_heatmap.detach()

        B, C, pH, pW = pred.shape
        _, _, gH, gW = gt.shape
        scale_x = gW / pW
        scale_y = gH / pH

        for b in range(B):
            for c in range(C):
                # 预测坐标：在原始分辨率取 argmax，再缩放
                pred_flat = pred[b, c].view(-1)
                pred_idx = pred_flat.argmax().item()
                pred_y, pred_x = divmod(pred_idx, pW)
                pred_x = pred_x * scale_x
                pred_y = pred_y * scale_y

                # GT 坐标：在 GT 分辨率取 argmax
                gt_flat = gt[b, c].view(-1)
                gt_idx = gt_flat.argmax().item()
                gt_y, gt_x = divmod(gt_idx, gW)

                dist = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
                distances.append(dist)

        return distances

    def _get_exp_dir(self):
        """自动递增实验目录: ./param/{model_type}/exp1, exp2, ..."""
        base = os.path.join(self.save_dir, self.model_type)
        i = 1
        while os.path.exists(os.path.join(base, f'exp{i}')):
            i += 1
        return os.path.join(base, f'exp{i}')

    def _save_model(self, version):
        """保存模型（dict 格式，方便恢复训练）"""
        if not hasattr(self, '_exp_dir'):
            self._exp_dir = self._get_exp_dir()
            os.makedirs(self._exp_dir)
            print(f"📁 模型保存目录: {self._exp_dir}")

        save_path = os.path.join(self._exp_dir, f'{version}_model.pth')
        torch.save({'model_state_dict': self.net.state_dict()}, save_path)
        print(f"💾 {version.upper()}模型已保存至: {save_path}")


if __name__ == '__main__':
    trainer = Trainer(
        weight=None,
        epoch=150,
        model_type='StackedHourglass',
        num_keypoints=1,
        train_data_path=r'./datasets/data_center_train.txt',
        val_data_path=r'./datasets/data_center_val.txt',
        save_path='train_image',
    )
    trainer.train()
