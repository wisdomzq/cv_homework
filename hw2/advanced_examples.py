"""
高级示例：自定义掩码和多种融合效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from laplacian_pyramid_fusion import LaplacianPyramid
import os


def create_custom_text_mask(shape: tuple, text: str = "FUSION", 
                            font_scale: float = 3.0) -> np.ndarray:
    """
    创建文字形状的掩码
    
    Args:
        shape: 图像形状 (height, width)
        text: 要显示的文字
        font_scale: 字体大小
        
    Returns:
        掩码数组
    """
    h, w = shape
    mask = np.zeros(shape, dtype=np.uint8)
    
    # 计算文字大小和位置
    font = cv2.FONT_HERSHEY_BOLD
    text_size = cv2.getTextSize(text, font, font_scale, 10)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    
    # 绘制文字
    cv2.putText(mask, text, (text_x, text_y), font, font_scale, 255, 10)
    
    # 平滑处理
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    mask = mask.astype(np.float32) / 255.0
    
    return mask


def create_diagonal_mask(shape: tuple, angle: float = 45, smooth: bool = True) -> np.ndarray:
    """
    创建对角线掩码
    
    Args:
        shape: 图像形状 (height, width)
        angle: 角度（度）
        smooth: 是否平滑
        
    Returns:
        掩码数组
    """
    h, w = shape
    mask = np.zeros(shape, dtype=np.float32)
    
    # 创建坐标网格
    y, x = np.ogrid[:h, :w]
    
    # 计算到对角线的距离
    angle_rad = np.deg2rad(angle)
    # 对角线方程: x*sin(θ) - y*cos(θ) = 0
    distance = x * np.sin(angle_rad) - y * np.cos(angle_rad)
    
    # 归一化到0-1
    distance = (distance - distance.min()) / (distance.max() - distance.min())
    
    if smooth:
        # 创建平滑过渡
        mask = 1 / (1 + np.exp(-20 * (distance - 0.5)))
    else:
        mask = (distance > 0.5).astype(np.float32)
    
    return mask


def create_checkerboard_mask(shape: tuple, squares: int = 8, 
                             smooth: bool = True) -> np.ndarray:
    """
    创建棋盘格掩码
    
    Args:
        shape: 图像形状 (height, width)
        squares: 每行/列的格子数
        smooth: 是否平滑边界
        
    Returns:
        掩码数组
    """
    h, w = shape
    square_h = h // squares
    square_w = w // squares
    
    mask = np.zeros(shape, dtype=np.float32)
    
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                y1, y2 = i * square_h, min((i + 1) * square_h, h)
                x1, x2 = j * square_w, min((j + 1) * square_w, w)
                mask[y1:y2, x1:x2] = 1.0
    
    if smooth:
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
    
    return mask


def create_radial_gradient_mask(shape: tuple, center: tuple = None, 
                                inner_radius: float = 0.2, 
                                outer_radius: float = 0.8) -> np.ndarray:
    """
    创建径向渐变掩码
    
    Args:
        shape: 图像形状 (height, width)
        center: 中心点，默认为图像中心
        inner_radius: 内半径（相对于图像大小）
        outer_radius: 外半径（相对于图像大小）
        
    Returns:
        掩码数组
    """
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    
    max_dist = np.sqrt(h**2 + w**2) / 2
    inner_r = max_dist * inner_radius
    outer_r = max_dist * outer_radius
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # 创建径向渐变
    mask = np.clip((distance - inner_r) / (outer_r - inner_r), 0, 1)
    
    return mask.astype(np.float32)


def create_wave_mask(shape: tuple, frequency: int = 4, 
                     direction: str = 'horizontal', 
                     smooth: bool = True) -> np.ndarray:
    """
    创建波浪形掩码
    
    Args:
        shape: 图像形状 (height, width)
        frequency: 波浪频率
        direction: 波浪方向 'horizontal' 或 'vertical'
        smooth: 是否平滑
        
    Returns:
        掩码数组
    """
    h, w = shape
    
    if direction == 'horizontal':
        x = np.linspace(0, frequency * 2 * np.pi, w)
        wave = np.sin(x)
        wave = (wave + 1) / 2  # 归一化到0-1
        mask = np.tile(wave, (h, 1))
    else:  # vertical
        y = np.linspace(0, frequency * 2 * np.pi, h)
        wave = np.sin(y)
        wave = (wave + 1) / 2
        mask = np.tile(wave.reshape(-1, 1), (1, w))
    
    if smooth:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 15)
    
    return mask.astype(np.float32)


def showcase_all_masks():
    """展示所有掩码类型"""
    print("=== 展示各种掩码类型 ===\n")
    
    shape = (512, 512)
    
    # 创建所有掩码
    masks = {
        'Text Mask': create_custom_text_mask(shape, "CV"),
        'Diagonal (45°)': create_diagonal_mask(shape, 45),
        'Diagonal (135°)': create_diagonal_mask(shape, 135),
        'Checkerboard': create_checkerboard_mask(shape, squares=8),
        'Radial Gradient': create_radial_gradient_mask(shape),
        'Horizontal Wave': create_wave_mask(shape, frequency=4, direction='horizontal'),
        'Vertical Wave': create_wave_mask(shape, frequency=4, direction='vertical'),
    }
    
    # 可视化所有掩码
    n_masks = len(masks)
    cols = 3
    rows = (n_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    
    for i, (name, mask) in enumerate(masks.items()):
        axes[i].imshow(mask, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_masks, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_showcase.png', dpi=150, bbox_inches='tight')
    print("掩码展示已保存到: mask_showcase.png")
    plt.show()


def fusion_gallery():
    """创建融合效果画廊"""
    print("\n=== 创建融合效果画廊 ===\n")
    
    # 创建两个测试图像
    h, w = 512, 512
    
    # 图像1: 渐变背景 + 圆形
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img1[i, :] = [255 * i // h, 100, 255 - 255 * i // h]
    cv2.circle(img1, (w//2, h//2), 120, (255, 255, 0), -1)
    cv2.circle(img1, (w//2, h//2), 100, (255, 200, 0), -1)
    
    # 图像2: 不同的渐变 + 矩形
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    for j in range(w):
        img2[:, j] = [255 - 255 * j // w, 255 * j // w, 100]
    cv2.rectangle(img2, (w//2-100, h//2-100), (w//2+100, h//2+100), (0, 255, 255), -1)
    cv2.rectangle(img2, (w//2-80, h//2-80), (w//2+80, h//2+80), (0, 200, 255), -1)
    
    lp = LaplacianPyramid(levels=6)
    
    # 创建不同掩码的融合结果
    fusion_configs = [
        ('Text Mask', create_custom_text_mask((h, w), "CV")),
        ('Diagonal', create_diagonal_mask((h, w), 45)),
        ('Checkerboard', create_checkerboard_mask((h, w), squares=6)),
        ('Radial', create_radial_gradient_mask((h, w))),
        ('Wave', create_wave_mask((h, w), frequency=3)),
    ]
    
    # 创建画廊
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # 显示原始图像
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Image 1', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Image 2', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    
    # 显示融合结果
    for idx, (name, mask) in enumerate(fusion_configs):
        row = (idx + 3) // 3
        col = (idx + 3) % 3
        
        print(f"正在融合: {name}...")
        result = lp.blend_images(img1, img2, mask)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        
        # 保存单独的结果
        filename = f"fusion_{name.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(filename, result)
    
    plt.tight_layout()
    plt.savefig('fusion_gallery.png', dpi=150, bbox_inches='tight')
    print("\n融合画廊已保存到: fusion_gallery.png")
    plt.show()
    
    print("\n完成！")


def compare_pyramid_levels():
    """比较不同金字塔层数的效果"""
    print("\n=== 比较不同金字塔层数 ===\n")
    
    # 创建测试图像
    h, w = 512, 512
    
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    img1[:, :w//2] = [0, 100, 255]
    img1[:, w//2:] = [0, 150, 200]
    
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    img2[:, :w//2] = [255, 150, 0]
    img2[:, w//2:] = [255, 100, 0]
    
    # 创建垂直分割掩码
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, :w//2] = 1.0
    
    # 测试不同层数
    levels_to_test = [2, 3, 4, 5, 6, 7]
    
    fig, axes = plt.subplots(2, len(levels_to_test), figsize=(18, 6))
    
    for idx, levels in enumerate(levels_to_test):
        print(f"测试 {levels} 层金字塔...")
        
        lp = LaplacianPyramid(levels=levels)
        result = lp.blend_images(img1, img2, mask)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 显示完整结果
        axes[0, idx].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[0, idx].set_title(f'{levels} Levels')
        axes[0, idx].axis('off')
        
        # 显示中心区域的放大图
        center_crop = result[h//2-50:h//2+50, w//2-50:w//2+50]
        axes[1, idx].imshow(cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB))
        axes[1, idx].set_title(f'Center Zoom')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('pyramid_levels_comparison.png', dpi=150, bbox_inches='tight')
    print("层数比较结果已保存到: pyramid_levels_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("高级拉普拉斯金字塔融合示例\n")
    print("=" * 50)
    
    # 1. 展示各种掩码
    showcase_all_masks()
    
    # 2. 创建融合画廊
    fusion_gallery()
    
    # 3. 比较金字塔层数
    compare_pyramid_levels()
    
    print("\n" + "=" * 50)
    print("所有示例完成！")
