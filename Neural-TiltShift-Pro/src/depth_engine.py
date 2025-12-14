import torch
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from torchvision.transforms import Compose
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式实现，确保模型只加载一次"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DepthEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_type: str = "MiDaS_small", device: str = "auto"):
        """
        初始化深度估计引擎
        Args:
            model_type: 模型类型 (MiDaS_small, DPT_Hybrid, DPT_Large)
            device: 运行设备 (cuda, mps, cpu, auto)
        """
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        logger.info(f"Initializing DepthEngine with model: {model_type}")
        
        # 设备选择
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")

        try:
            # 加载 MiDaS 模型
            # trust_repo=True 抑制警告并可能绕过部分检查
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            # 加载对应的 transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            self.initialized = True
            logger.info("DepthEngine initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize DepthEngine: {e}")
            logger.error("提示: 如果遇到 HTTP 403 错误，可能是 GitHub API 速率限制。请尝试等待或设置 GITHUB_TOKEN。")
            raise e

    def estimate_depth(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        估计单帧深度
        Args:
            img_rgb: RGB 图像 (H, W, 3) numpy array
        Returns:
            depth_map: 归一化深度图 (H, W) float32, 0.0-1.0
        """
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # 插值回原图尺寸
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # 鲁棒归一化 (避免极值影响)
        depth_min = np.percentile(depth_map, 2)
        depth_max = np.percentile(depth_map, 98)
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        depth_norm = np.clip(depth_norm, 0, 1)
        
        return depth_norm

    def generate_trimap(self, depth_map: np.ndarray, focus_depth: float, depth_range: float = 0.15) -> np.ndarray:
        """
        根据焦点生成前景 Mask (Trimap)
        Args:
            depth_map: 归一化深度图
            focus_depth: 焦点深度 (0.0 - 1.0)
            depth_range: 景深范围
        Returns:
            mask: 前景 Mask (0.0 - 1.0)
        """
        # 计算距离焦点的距离
        dist = np.abs(depth_map - focus_depth)
        
        # 创建 Soft Mask
        # 使用 Sigmoid 或 Gaussian 函数使边缘更平滑
        # 这里使用简单的线性插值模拟
        mask = np.clip(1.0 - (dist / depth_range), 0, 1)
        
        # 二值化处理用于 Inpainting，但保留 Soft Mask 用于合成
        # 这里返回 Soft Mask
        return mask.astype(np.float32)
