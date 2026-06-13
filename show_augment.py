"""
数据增强可视化 & 数据集生成脚本

功能1：可视化增强效果（augment_vis/）
  flat=False — 每个样本一个文件夹
  flat=True  — 按类型平铺

功能2：生成增强数据集（augment_dataset/）
  生成新的图片 + data_center_train2.txt / data_center_val2.txt
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from data import MyDataset, AUGMENT_NAMES, transform
from heatmap_label import CenterLabelHeatMap


def save_heatmap_img(heatmap, save_path):
    hm = heatmap[0].numpy()
    hm = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, hm_color)


def save_img_tensor(img_tensor, save_path):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)


def draw_keypoint_on_img(img_tensor, heatmap_tensor, save_path):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hm = heatmap_tensor[0].numpy()
    idx = np.argmax(hm)
    y, x = divmod(idx, hm.shape[1])
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(img, f'({x},{y})', (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.imwrite(save_path, img)


def apply_augment(image, points, aug_name):
    """应用单种增强，返回 (img_tensor, hm_tensor)"""
    ds_dummy = MyDataset.__new__(MyDataset)
    aug_img, aug_pts = ds_dummy.apply_single_augment(image, points, aug_name)

    hm_list = []
    for k in range(0, len(aug_pts), 2):
        hm = CenterLabelHeatMap(128, 128, aug_pts[k], aug_pts[k + 1], 20)
        hm_list.append(hm)
    hm_tensor = torch.Tensor(np.stack(hm_list))
    img_tensor = transform(aug_img)

    return img_tensor, hm_tensor, aug_pts


# ==================== 功能1：可视化增强效果 ====================

def process_dataset(txt_path, output_dir, flat=False):
    ds = MyDataset(txt_path, augment=False)

    print(f"\n📊 数据集: {txt_path}")
    print(f"   样本数: {len(ds)}")
    print(f"   模式: {'平铺' if flat else '按样本分文件夹'}")

    if flat:
        img_dir = os.path.join(output_dir, 'images')
        hm_dir = os.path.join(output_dir, 'heatmaps')
        kp_dir = os.path.join(output_dir, 'with_kp')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(hm_dir, exist_ok=True)
        os.makedirs(kp_dir, exist_ok=True)

        for idx in range(len(ds)):
            img_tensor, hm_tensor = ds[idx]
            image = Image.open(ds.dataset[idx].split(' ')[0]).convert('RGB')
            points = [int(p.rstrip("\n")) for p in ds.dataset[idx].split(' ')[1:]]

            save_img_tensor(img_tensor, os.path.join(img_dir, f'{idx}_original_img.png'))
            save_heatmap_img(hm_tensor, os.path.join(hm_dir, f'{idx}_original_heatmap.png'))
            draw_keypoint_on_img(img_tensor, hm_tensor, os.path.join(kp_dir, f'{idx}_original_with_kp.png'))

            for aug_name in AUGMENT_NAMES:
                aug_img_t, aug_hm_t, _ = apply_augment(image, points, aug_name)
                save_img_tensor(aug_img_t, os.path.join(img_dir, f'{idx}_{aug_name}_img.png'))
                save_heatmap_img(aug_hm_t, os.path.join(hm_dir, f'{idx}_{aug_name}_heatmap.png'))
                draw_keypoint_on_img(aug_img_t, aug_hm_t, os.path.join(kp_dir, f'{idx}_{aug_name}_with_kp.png'))

            print(f"  ✅ [{idx + 1}/{len(ds)}]")

        total = len(ds) * (1 + len(AUGMENT_NAMES))
        print(f"   📁 {img_dir}/ ({total} 张)")
        print(f"   📁 {hm_dir}/ ({total} 张)")
        print(f"   📁 {kp_dir}/ ({total} 张)")

    else:
        for idx in range(len(ds)):
            img_tensor, hm_tensor = ds[idx]
            image = Image.open(ds.dataset[idx].split(' ')[0]).convert('RGB')
            points = [int(p.rstrip("\n")) for p in ds.dataset[idx].split(' ')[1:]]

            sample_dir = os.path.join(output_dir, f'{idx}')
            os.makedirs(sample_dir, exist_ok=True)

            save_img_tensor(img_tensor, os.path.join(sample_dir, 'original_img.png'))
            save_heatmap_img(hm_tensor, os.path.join(sample_dir, 'original_heatmap.png'))
            draw_keypoint_on_img(img_tensor, hm_tensor, os.path.join(sample_dir, 'original_with_kp.png'))

            for aug_name in AUGMENT_NAMES:
                aug_img_t, aug_hm_t, _ = apply_augment(image, points, aug_name)
                save_img_tensor(aug_img_t, os.path.join(sample_dir, f'{aug_name}_img.png'))
                save_heatmap_img(aug_hm_t, os.path.join(sample_dir, f'{aug_name}_heatmap.png'))
                draw_keypoint_on_img(aug_img_t, aug_hm_t, os.path.join(sample_dir, f'{aug_name}_with_kp.png'))

            print(f"  ✅ [{idx + 1}/{len(ds)}] {sample_dir}/")


# ==================== 功能2：生成增强数据集 ====================

def generate_dataset(txt_path, output_img_dir, output_txt_path, name_offset=0):
    """生成增强图片 + 对应的 txt 标签文件"""
    ds = MyDataset(txt_path, augment=False)

    print(f"\n📊 数据集: {txt_path}")
    print(f"   样本数: {len(ds)}")

    os.makedirs(output_img_dir, exist_ok=True)
    lines = []

    for idx in range(len(ds)):
        file_idx = idx + name_offset

        data = ds.dataset[idx]
        image = Image.open(data.split(' ')[0]).convert('RGB')
        points = [int(p.rstrip("\n")) for p in data.split(' ')[1:]]
        img_np = np.array(image)

        # 原图
        save_name = f'{file_idx}_original.png'
        cv2.imwrite(os.path.join(output_img_dir, save_name), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        lines.append(f'{output_img_dir}/{save_name} ' + ' '.join(str(p) for p in points))

        # 每种增强
        for aug_name in AUGMENT_NAMES:
            _, _, aug_pts = apply_augment(image, points, aug_name)
            aug_img_pil, _ = MyDataset.__new__(MyDataset).apply_single_augment(image, points, aug_name)
            aug_np = np.array(aug_img_pil)

            save_name = f'{file_idx}_{aug_name}.png'
            cv2.imwrite(os.path.join(output_img_dir, save_name), cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR))
            lines.append(f'{output_img_dir}/{save_name} ' + ' '.join(str(p) for p in aug_pts))

        print(f"  ✅ [{idx + 1}/{len(ds)}]")

    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"   📝 标签: {output_txt_path} ({len(lines)} 行)")
    print(f"   📁 图片: {output_img_dir}/ ({len(lines)} 张)")


# ==================== 主函数 ====================

def main():
    # ========== 参数配置 ==========
    mode = 'dataset'  # 'visualize'=可视化增强效果, 'dataset'=生成增强数据集
    flat = True       # 仅 visualize 模式有效：True=平铺, False=按样本分文件夹
    # ==============================

    print("=" * 60)
    print(f"🎲 增强方式: {AUGMENT_NAMES}")
    print(f"📂 模式: {mode}")
    print("=" * 60)

    if mode == 'visualize':
        process_dataset('./datasets/data_center_train.txt', 'augment_vis/train', flat)
        process_dataset('./datasets/data_center_val.txt', 'augment_vis/val', flat)
        print("\n🏁 完成！augment_vis/train/ 和 augment_vis/val/")

    elif mode == 'dataset':
        generate_dataset(
            './datasets/data_center_train.txt',
            'augment_dataset/data_train/images',
            'augment_dataset/data_center_train2.txt',
            name_offset=0,
        )
        generate_dataset(
            './datasets/data_center_val.txt',
            'augment_dataset/data_val/images',
            'augment_dataset/data_center_val2.txt',
            name_offset=0,
        )
        print("\n🏁 完成！")
        print("   augment_dataset/data_train/images/")
        print("   augment_dataset/data_val/images/")
        print("   augment_dataset/data_center_train2.txt")
        print("   augment_dataset/data_center_val2.txt")

    print("=" * 60)


if __name__ == '__main__':
    main()
