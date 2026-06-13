import os
import cv2
import numpy as np
import torch

from utils import *
from torch.utils.data import Dataset
from torchvision import transforms
from heatmap_label import *

transform = transforms.Compose([
    transforms.ToTensor()
])

# 增强方式名称列表
AUGMENT_NAMES = ['flip', 'rotate', 'brightness', 'noise']


class MyDataset(Dataset):
    def __init__(self, path, augment=False, sigma=20):
        self.path = path
        self.augment = augment
        self.sigma = sigma
        f = open(path)
        self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def _augment_flip(self, img_np, points):
        """随机水平/垂直翻转"""
        h, w = img_np.shape[:2]
        if np.random.random() < 0.5:
            img_np = cv2.flip(img_np, 1)
            for i in range(0, len(points), 2):
                points[i] = w - points[i]
        if np.random.random() < 0.5:
            img_np = cv2.flip(img_np, 0)
            for i in range(1, len(points), 2):
                points[i] = h - points[i]
        return img_np, points

    def _augment_rotate(self, img_np, points):
        """随机旋转 ±15°"""
        h, w = img_np.shape[:2]
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_np = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            points[i] = int(np.clip(new_x, 0, w - 1))
            points[i + 1] = int(np.clip(new_y, 0, h - 1))
        return img_np, points

    def _augment_brightness(self, img_np, points):
        """随机亮度/对比度扰动"""
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-20, 20)
        img_np = np.clip(alpha * img_np.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return img_np, points

    def _augment_noise(self, img_np, points):
        """随机高斯噪声"""
        noise = np.random.normal(0, 10, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img_np, points

    def _augment(self, image, points):
        """数据增强：随机组合多种增强方式"""
        img_np = np.array(image)

        img_np, points = self._augment_flip(img_np, points)
        img_np, points = self._augment_rotate(img_np, points)
        img_np, points = self._augment_brightness(img_np, points)
        img_np, points = self._augment_noise(img_np, points)

        image = Image.fromarray(img_np)
        return image, points

    def apply_single_augment(self, image, points, aug_name):
        """应用单种增强方式，返回 (augmented_image, augmented_points)

        Args:
            image:    PIL Image
            points:   list [x1, y1, ...]
            aug_name: 'flip' / 'rotate' / 'brightness' / 'noise'

        Returns:
            image:  augmented PIL Image
            points: augmented list
        """
        img_np = np.array(image)
        method = getattr(self, f'_augment_{aug_name}')
        img_np, points = method(img_np, points.copy())
        return Image.fromarray(img_np), points

    def __getitem__(self, index):
        data = self.dataset[index]
        img_path = data.split(' ')[0]
        image = Image.open(img_path).convert('RGB')
        points = data.split(' ')[1:]
        points = [int(i.rstrip("\n")) for i in points]

        if self.augment:
            image, points = self._augment(image, points.copy())

        label = []
        for i in range(0, len(points), 2):
            heatmap = CenterLabelHeatMap(128, 128, points[i], points[i + 1], self.sigma)
            label.append(heatmap)
        label = np.stack(label)
        return transform(image), torch.Tensor(label)


if __name__ == '__main__':
    data = MyDataset('./datasets/data_center_val.txt')
    print(data[5][0].shape)
    print(data[5][1].shape)
    for i in data:
        print(i[0].shape)
