"""
融合质量分析脚本
- 计算 PSNR/SSIM（全图与掩码局部）
- 计算接缝带（seam band）梯度能量与拉普拉斯方差
- 与直接（硬/软）拼接对比
- 可配置金字塔层数与掩码类型
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from laplacian_pyramid_fusion import (
    LaplacianPyramid,
    create_vertical_split_mask,
    create_horizontal_split_mask,
    create_circular_mask,
)


def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    return g.astype(np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """PSNR，支持掩码。a, b: uint8 或 float (0..255/1)"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.max() <= 1.0 and b.max() <= 1.0:
        scale = 1.0
    else:
        scale = 255.0
    if mask is not None:
        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        diff = (a - b)
        if diff.ndim == 3:
            diff = np.mean(diff, axis=2)
        mse = (np.sum((diff**2) * m) / (np.sum(m) + 1e-8))
    else:
        diff = (a - b)
        if diff.ndim == 3:
            diff = np.mean(diff, axis=2)
        mse = np.mean(diff**2)
    if mse <= 1e-12:
        return 99.0
    return 20 * np.log10(scale) - 10 * np.log10(mse)


def ssim_global(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """
    简化版全局SSIM（非滑窗），支持掩码权重；范围约在[0,1]。
    对于报告的相对比较足够，避免额外依赖。
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.ndim == 3:
        # 对每通道取均值
        scores = []
        for c in range(3):
            scores.append(ssim_global(a[:, :, c], b[:, :, c], mask))
        return float(np.mean(scores))

    # 灰度
    if mask is None:
        m = np.ones_like(a, dtype=np.float32)
    else:
        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m / (m.max() + 1e-8)

    # 归一化到[0,1]
    if a.max() > 1.0 or b.max() > 1.0:
        a = a / 255.0
        b = b / 255.0

    # 加权均值/方差/协方差
    w = m
    w_sum = np.sum(w) + 1e-8
    mu_a = np.sum(a * w) / w_sum
    mu_b = np.sum(b * w) / w_sum
    sigma_a2 = np.sum(w * (a - mu_a) ** 2) / w_sum
    sigma_b2 = np.sum(w * (b - mu_b) ** 2) / w_sum
    sigma_ab = np.sum(w * (a - mu_a) * (b - mu_b)) / w_sum

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(num / (den + 1e-12))


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    g = to_gray_float(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def laplacian_var(img: np.ndarray) -> float:
    g = to_gray_float(img)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap.var())


def seam_band_from_mask(mask: np.ndarray, band_width: int = 16) -> np.ndarray:
    """根据二值掩码生成接缝带，返回0/1浮点图。"""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # 规范化并二值化
    if mask.dtype != np.uint8:
        m = (mask > 0.5).astype(np.uint8) * 255
    else:
        m = ((mask > 127).astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (band_width, band_width))
    dil = cv2.dilate(m, k)
    ero = cv2.erode(m, k)
    band = cv2.subtract(dil, ero)
    band = (band > 0).astype(np.float32)
    return band


def analyze(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray, levels: int = 6,
            save_prefix: str = "analysis") -> Dict[str, float]:
    # 准备直接拼接与拉普拉斯融合
    lp = LaplacianPyramid(levels=levels)
    fused = lp.blend_images(image1, image2, mask)
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    if mask.ndim == 2:
        mask3 = mask[:, :, None]
    else:
        mask3 = mask
    direct = (image1 * mask3 + image2 * (1 - mask3)).astype(np.uint8)

    # 区域掩码
    left_mask = (mask3[:, :, 0] >= 0.5).astype(np.float32)
    right_mask = 1.0 - left_mask
    seam_band = seam_band_from_mask(mask3, band_width=16)

    # 指标
    metrics: Dict[str, float] = {}
    metrics["PSNR_full_fused_vs_img1"] = psnr(fused, image1)
    metrics["PSNR_full_fused_vs_img2"] = psnr(fused, image2)
    metrics["PSNR_left(fused,img1)"] = psnr(fused, image1, left_mask)
    metrics["PSNR_right(fused,img2)"] = psnr(fused, image2, right_mask)

    metrics["SSIM_full_fused_vs_img1"] = ssim_global(fused, image1)
    metrics["SSIM_full_fused_vs_img2"] = ssim_global(fused, image2)
    metrics["SSIM_left(fused,img1)"] = ssim_global(fused, image1, left_mask)
    metrics["SSIM_right(fused,img2)"] = ssim_global(fused, image2, right_mask)

    # 接缝带梯度与拉普拉斯方差（越小越平滑）
    grad_fused = gradient_magnitude(fused)
    grad_direct = gradient_magnitude(direct)
    seam = seam_band.astype(np.float32)
    metrics["SeamGrad_fused"] = float(np.sum(grad_fused * seam) / (np.sum(seam) + 1e-8))
    metrics["SeamGrad_direct"] = float(np.sum(grad_direct * seam) / (np.sum(seam) + 1e-8))

    # 接缝带色差（RGB L1）
    diff_fused_direct = np.mean(np.abs(fused.astype(np.float32) - direct.astype(np.float32)), axis=2)
    metrics["SeamColorDiff(fused, direct)"] = float(np.sum(diff_fused_direct * seam) / (np.sum(seam) + 1e-8))

    # 全图清晰度（拉普拉斯方差）
    metrics["LaplacianVar_fused"] = laplacian_var(fused)
    metrics["LaplacianVar_direct"] = laplacian_var(direct)

    # 频谱可视化（差分）
    def spectrum(img):
        g = to_gray_float(img)
        F = np.fft.fftshift(np.fft.fft2(g))
        S = np.log(1 + np.abs(F))
        S = S / (S.max() + 1e-8)
        return S

    spec_fused = spectrum(fused)
    spec_direct = spectrum(direct)

    # 可视化面板
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Image 1"); axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)); axes[0, 1].set_title("Image 2"); axes[0, 1].axis('off')
    axes[0, 2].imshow(mask if mask.ndim==2 else mask[:, :, 0], cmap='gray'); axes[0, 2].set_title("Mask"); axes[0, 2].axis('off')
    axes[0, 3].imshow(seam_band, cmap='magma'); axes[0, 3].set_title("Seam Band"); axes[0, 3].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)); axes[1, 0].set_title("Direct Blend"); axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)); axes[1, 1].set_title(f"Laplacian Blend (L={levels})"); axes[1, 1].axis('off')
    axes[1, 2].imshow(spec_direct, cmap='viridis'); axes[1, 2].set_title("Spectrum Direct"); axes[1, 2].axis('off')
    axes[1, 3].imshow(spec_fused, cmap='viridis'); axes[1, 3].set_title("Spectrum Fused"); axes[1, 3].axis('off')

    plt.tight_layout()
    fig.savefig(f"{save_prefix}_panel.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 保存文本报告
    with open(f"{save_prefix}_metrics.txt", "w", encoding="utf-8") as f:
        f.write("融合质量指标\n")
        f.write("="*40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        # 简要判读
        f.write("\n判读：\n")
        if metrics["SeamGrad_fused"] < metrics["SeamGrad_direct"]:
            f.write("- 接缝带梯度能量：拉普拉斯融合更平滑（更小）。\n")
        else:
            f.write("- 接缝带梯度能量：未显著优于直接拼接，检查层数/掩码。\n")
        if metrics["LaplacianVar_fused"] >= metrics["LaplacianVar_direct"]:
            f.write("- 清晰度（LaplacianVar）：融合未降低清晰度。\n")
        else:
            f.write("- 清晰度（LaplacianVar）：融合可能轻微平滑，考虑减少层数或优化掩码。\n")

    return metrics


def _load_images(p1: str, p2: str, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    if img1 is None or img2 is None:
        raise RuntimeError("无法读取输入图像")
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    return img1, img2


def _build_mask(name: str, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if name == 'vertical':
        return create_vertical_split_mask((h, w), 0.5).astype(np.float32)
    if name == 'horizontal':
        return create_horizontal_split_mask((h, w), 0.5).astype(np.float32)
    if name == 'circular':
        return create_circular_mask((h, w), smooth=True).astype(np.float32)
    # 默认垂直
    return create_vertical_split_mask((h, w), 0.5).astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='拉普拉斯金字塔融合质量分析')
    parser.add_argument('img1', help='图像1路径')
    parser.add_argument('img2', help='图像2路径')
    parser.add_argument('--size', type=str, default='512,512', help='目标尺寸: W,H')
    parser.add_argument('--levels', type=int, default=6, help='金字塔层数')
    parser.add_argument('--mask', type=str, default='vertical', choices=['vertical','horizontal','circular'], help='掩码类型')
    parser.add_argument('--out', type=str, default='analysis', help='输出前缀')
    args = parser.parse_args()

    W, H = [int(x) for x in args.size.split(',')]
    img1, img2 = _load_images(args.img1, args.img2, (W, H))
    m = _build_mask(args.mask, (H, W))

    metrics = analyze(img1, img2, m, levels=args.levels, save_prefix=args.out)
    print('分析完成，主要指标：')
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"可视化与指标文件已保存为：{args.out}_panel.png, {args.out}_metrics.txt")


if __name__ == '__main__':
    main()
