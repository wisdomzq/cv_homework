import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Inpainter:
    def __init__(self, method: str = "navier-stokes"):
        """
        初始化修复器
        Args:
            method: 'navier-stokes' (cv2.INPAINT_NS) or 'telea' (cv2.INPAINT_TELEA)
        """
        self.method = method
        logger.info(f"Inpainter initialized with method: {method}")

    def process(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        执行背景修复
        Args:
            img: 原始图像 (H, W, 3) BGR
            mask: 前景 Mask (H, W) float32, 1.0 为前景
        Returns:
            bg_filled: 修复后的背景图
        """
        # 1. 准备 Inpainting Mask
        # Mask > 0.1 的区域被视为需要修复的区域
        # 转换为 uint8
        binary_mask = (mask > 0.1).astype(np.uint8) * 255
        
        # 2. 膨胀 Mask
        # 确保覆盖边缘，防止前景颜色渗漏
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)

        # 3. 执行 Inpainting
        radius = 3
        flags = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        
        try:
            bg_filled = cv2.inpaint(img, dilated_mask, radius, flags)
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            # 如果失败，返回原图作为 fallback
            return img
            
        return bg_filled
