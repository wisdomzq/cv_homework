"""
使用真实图像进行拉普拉斯金字塔融合的示例
可以用于融合两张实际的照片
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from laplacian_pyramid_fusion import (
    LaplacianPyramid, 
    create_vertical_split_mask,
    create_horizontal_split_mask,
    create_circular_mask
)
import os
from analyze_fusion import analyze


def load_and_resize_images(path1: str, path2: str, target_size: tuple = None):
    """
    加载并调整两张图像到相同尺寸
    
    Args:
        path1: 第一张图像路径
        path2: 第二张图像路径
        target_size: 目标尺寸 (width, height)，None表示使用第一张图像的尺寸
        
    Returns:
        调整后的两张图像
    """
    # 读取图像
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None:
        raise ValueError(f"无法读取图像: {path1}")
    if img2 is None:
        raise ValueError(f"无法读取图像: {path2}")
    
    # 确定目标尺寸
    if target_size is None:
        target_size = (img1.shape[1], img1.shape[0])
    
    # 调整图像尺寸
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)
    
    return img1, img2


def fusion_with_custom_mask(img1: np.ndarray, img2: np.ndarray, 
                           mask: np.ndarray, levels: int = 6,
                           save_path: str = None,
                           analysis_prefix: str = None):
    """
    使用自定义掩码融合两张图像
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        mask: 融合掩码
        levels: 金字塔层数
        save_path: 保存路径
        
    Returns:
        融合后的图像
    """
    lp = LaplacianPyramid(levels=levels)
    
    # 执行融合
    result = lp.blend_images(img1, img2, mask)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 直接拼接对比（不使用金字塔）
    if len(mask.shape) == 2:
        mask_3d = mask[:, :, np.newaxis]
    else:
        mask_3d = mask
    direct_blend = (img1 * mask_3d + img2 * (1 - mask_3d)).astype(np.uint8)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    if len(mask.shape) == 3:
        axes[0, 2].imshow(mask[:, :, 0], cmap='gray')
    else:
        axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Fusion Mask')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(direct_blend, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Direct Blend (No Pyramid)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Laplacian Pyramid Blend')
    axes[1, 1].axis('off')
    
    # 显示差异
    diff = cv2.absdiff(result, direct_blend)
    axes[1, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Difference (Enhanced)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
        
        # 同时保存融合结果图像
        result_img_path = save_path.replace('.png', '_result.jpg')
        cv2.imwrite(result_img_path, result)
        print(f"融合图像已保存到: {result_img_path}")

    # 质量分析（可选）
    if analysis_prefix is not None:
        metrics = analyze(img1, img2, mask.astype(np.float32), levels=levels, save_prefix=analysis_prefix)
        print("\n质量分析主要指标：")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"分析面板/指标文件已保存为：{analysis_prefix}_panel.png, {analysis_prefix}_metrics.txt")
    
    plt.show()
    
    return result


def create_gradient_mask(shape: tuple, direction: str = 'horizontal',
                        smooth: bool = True) -> np.ndarray:
    """
    创建渐变掩码
    
    Args:
        shape: 图像形状 (height, width)
        direction: 渐变方向 'horizontal' 或 'vertical'
        smooth: 是否平滑
        
    Returns:
        掩码数组
    """
    h, w = shape
    
    if direction == 'horizontal':
        # 从左到右渐变
        mask = np.linspace(1, 0, w)
        mask = np.tile(mask, (h, 1))
    else:  # vertical
        # 从上到下渐变
        mask = np.linspace(1, 0, h)
        mask = np.tile(mask.reshape(-1, 1), (1, w))
    
    if smooth:
        # 应用高斯平滑
        mask = cv2.GaussianBlur(mask.astype(np.float32), (51, 51), 30)
    
    return mask.astype(np.float32)

def demo_with_real_images(img1_path: str, img2_path: str, 
                         target_size: tuple = (512, 512), do_analysis: bool = True):
    """
    使用真实图像进行演示
    
    Args:
        img1_path: 第一张图像路径
        img2_path: 第二张图像路径
        target_size: 目标尺寸
    """
    print(f"=== 使用真实图像演示 ===")
    print(f"图像1: {img1_path}")
    print(f"图像2: {img2_path}\n")
    
    try:
        # 加载图像
        img1, img2 = load_and_resize_images(img1_path, img2_path, target_size)
        h, w = img1.shape[:2]

        print("1. 垂直分割融合...")
        mask_vertical = create_vertical_split_mask((h, w), split_position=0.5)
        fusion_with_custom_mask(
            img1, img2, mask_vertical, levels=6,
            save_path='real_vertical_fusion.png',
            analysis_prefix=('real_vertical' if do_analysis else None)
        )

        print("\n2. 圆形融合...")
        mask_circular = create_circular_mask((h, w), smooth=True)
        fusion_with_custom_mask(
            img1, img2, mask_circular, levels=6,
            save_path='real_circular_fusion.png',
            analysis_prefix=('real_circular' if do_analysis else None)
        )

        print("\n3. 渐变融合...")
        mask_gradient = create_gradient_mask((h, w), direction='horizontal', smooth=True)
        fusion_with_custom_mask(
            img1, img2, mask_gradient, levels=6,
            save_path='real_gradient_fusion.png',
            analysis_prefix=('real_gradient' if do_analysis else None)
        )

        print("\n完成！")

    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    # 检查是否有提供真实图像路径
    import sys
    
    if len(sys.argv) >= 3:
        # 使用命令行参数提供的图像
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        target_size = (512, 512) if len(sys.argv) < 4 else eval(sys.argv[3])
        demo_with_real_images(img1_path, img2_path, target_size)
    else:
        # 尝试查找当前目录下的图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if len(image_files) >= 2:
            print(f"找到图像文件: {image_files[:2]}")
            demo_with_real_images(image_files[0], image_files[1])
        else:
            print("未找到足够的图像文件，使用生成的图像进行演示")
            print("提示: 可以通过命令行参数指定图像:")
            print("  python real_image_fusion.py image1.jpg image2.jpg")

