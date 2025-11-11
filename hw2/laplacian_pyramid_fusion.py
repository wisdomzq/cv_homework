"""
基于拉普拉斯金字塔的图像融合
实现了拉普拉斯金字塔的构建、图像融合和重建
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class LaplacianPyramid:
    """拉普拉斯金字塔类"""
    
    def __init__(self, levels: int = 5):
        """
        初始化拉普拉斯金字塔
        
        Args:
            levels: 金字塔层数
        """
        self.levels = levels
    
    def build_gaussian_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """
        构建高斯金字塔
        
        Args:
            image: 输入图像
            
        Returns:
            高斯金字塔列表，从原图到最小尺寸
        """
        gaussian_pyramid = [image]
        current_level = image.copy()
        
        for i in range(self.levels - 1):
            # 使用高斯模糊并下采样
            current_level = cv2.pyrDown(current_level)
            gaussian_pyramid.append(current_level)
        
        return gaussian_pyramid
    
    def build_laplacian_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """
        构建拉普拉斯金字塔
        
        Args:
            image: 输入图像
            
        Returns:
            拉普拉斯金字塔列表
        """
        # 首先构建高斯金字塔
        gaussian_pyramid = self.build_gaussian_pyramid(image)
        laplacian_pyramid = []
        
        # 对每一层计算拉普拉斯图像
        for i in range(self.levels - 1):
            # 获取当前层和下一层（更小的层）
            current_level = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]
            
            # 上采样下一层
            upsampled = cv2.pyrUp(next_level, dstsize=(current_level.shape[1], current_level.shape[0]))
            
            # 计算拉普拉斯图像（当前层 - 上采样的下一层）
            laplacian = cv2.subtract(current_level, upsampled)
            laplacian_pyramid.append(laplacian)
        
        # 最后一层（最小的高斯层）直接作为拉普拉斯金字塔的最后一层
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def reconstruct_from_laplacian(self, laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
        """
        从拉普拉斯金字塔重建图像
        
        Args:
            laplacian_pyramid: 拉普拉斯金字塔列表
            
        Returns:
            重建的图像
        """
        # 从最顶层（最小的）开始重建
        reconstructed = laplacian_pyramid[-1]
        
        # 从倒数第二层开始往上重建
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            # 上采样当前重建结果
            reconstructed = cv2.pyrUp(reconstructed, 
                                     dstsize=(laplacian_pyramid[i].shape[1], 
                                            laplacian_pyramid[i].shape[0]))
            
            # 加上当前层的拉普拉斯图像
            reconstructed = cv2.add(reconstructed, laplacian_pyramid[i])
        
        return reconstructed
    
    def blend_images(self, image1: np.ndarray, image2: np.ndarray, 
                    mask: np.ndarray) -> np.ndarray:
        """
        使用拉普拉斯金字塔融合两张图像
        
        Args:
            image1: 第一张输入图像
            image2: 第二张输入图像
            mask: 融合掩码（0-1之间的值，或0-255）
            
        Returns:
            融合后的图像
        """
        # 确保mask是浮点型且在0-1之间
        if mask.dtype != np.float32 and mask.dtype != np.float64:
            mask = mask.astype(np.float32) / 255.0
        
        # 确保mask是三通道（如果输入图像是彩色的）
        if len(image1.shape) == 3 and len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=2)
        
        # 构建两张图像的拉普拉斯金字塔
        lap_pyramid1 = self.build_laplacian_pyramid(image1)
        lap_pyramid2 = self.build_laplacian_pyramid(image2)
        
        # 构建掩码的高斯金字塔
        mask_pyramid = self.build_gaussian_pyramid(mask)
        
        # 融合每一层
        blended_pyramid = []
        for lap1, lap2, mask_level in zip(lap_pyramid1, lap_pyramid2, mask_pyramid):
            # 确保mask_level的尺寸与拉普拉斯层匹配
            if mask_level.shape[:2] != lap1.shape[:2]:
                mask_level = cv2.resize(mask_level, (lap1.shape[1], lap1.shape[0]))
            
            # 融合当前层：blended = mask * lap1 + (1 - mask) * lap2
            blended = lap1 * mask_level + lap2 * (1 - mask_level)
            blended_pyramid.append(blended)
        
        # 从融合后的拉普拉斯金字塔重建图像
        result = self.reconstruct_from_laplacian(blended_pyramid)
        
        return result


def create_vertical_split_mask(shape: Tuple[int, int], split_position: float = 0.5) -> np.ndarray:
    """
    创建垂直分割掩码
    
    Args:
        shape: 图像形状 (height, width)
        split_position: 分割位置（0-1之间，表示左侧占比）
        
    Returns:
        掩码数组
    """
    mask = np.zeros(shape, dtype=np.float32)
    split_col = int(shape[1] * split_position)
    mask[:, :split_col] = 1.0
    return mask


def create_horizontal_split_mask(shape: Tuple[int, int], split_position: float = 0.5) -> np.ndarray:
    """
    创建水平分割掩码
    
    Args:
        shape: 图像形状 (height, width)
        split_position: 分割位置（0-1之间，表示上侧占比）
        
    Returns:
        掩码数组
    """
    mask = np.zeros(shape, dtype=np.float32)
    split_row = int(shape[0] * split_position)
    mask[:split_row, :] = 1.0
    return mask


def create_circular_mask(shape: Tuple[int, int], center: Tuple[int, int] = None, 
                        radius: int = None, smooth: bool = True) -> np.ndarray:
    """
    创建圆形掩码
    
    Args:
        shape: 图像形状 (height, width)
        center: 圆心位置，默认为图像中心
        radius: 半径，默认为图像较短边的1/3
        smooth: 是否平滑边缘
        
    Returns:
        掩码数组
    """
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 3
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    if smooth:
        # 创建平滑过渡
        smooth_width = radius * 0.2
        mask = np.clip((radius + smooth_width - distance) / (2 * smooth_width), 0, 1)
    else:
        mask = (distance <= radius).astype(np.float32)
    
    return mask


def visualize_pyramids(image: np.ndarray, levels: int = 5, title: str = "Pyramids"):
    """
    可视化高斯金字塔和拉普拉斯金字塔
    
    Args:
        image: 输入图像
        levels: 金字塔层数
        title: 标题
    """
    lp = LaplacianPyramid(levels=levels)
    
    # 构建金字塔
    gaussian_pyramid = lp.build_gaussian_pyramid(image)
    laplacian_pyramid = lp.build_laplacian_pyramid(image)
    
    # 可视化
    fig, axes = plt.subplots(2, levels, figsize=(15, 6))
    fig.suptitle(title)
    
    for i in range(levels):
        # 高斯金字塔
        if len(gaussian_pyramid[i].shape) == 3:
            axes[0, i].imshow(cv2.cvtColor(gaussian_pyramid[i], cv2.COLOR_BGR2RGB))
        else:
            axes[0, i].imshow(gaussian_pyramid[i], cmap='gray')
        axes[0, i].set_title(f'Gaussian L{i}')
        axes[0, i].axis('off')
        
        # 拉普拉斯金字塔（需要归一化显示）
        lap_img = laplacian_pyramid[i]
        lap_img_norm = cv2.normalize(lap_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(lap_img_norm.shape) == 3:
            axes[1, i].imshow(cv2.cvtColor(lap_img_norm, cv2.COLOR_BGR2RGB))
        else:
            axes[1, i].imshow(lap_img_norm, cmap='gray')
        axes[1, i].set_title(f'Laplacian L{i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


