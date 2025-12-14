import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class Renderer:
    """
    沙姆定律 (Scheimpflug Principle) 移轴渲染器
    
    实现功能:
    1. 3D 旋转焦平面 (Virtual Tilt) - 通过 tilt_angle_x/y 控制焦平面倾斜
    2. 透视校正 (Virtual Shift) - 通过 shift_correction 进行梯形校正
    """
    
    def __init__(self):
        pass

    def enhance_color(self, img: np.ndarray, saturation_boost: float = 1.4, value_boost: float = 1.1) -> np.ndarray:
        """色彩增强 (微缩模型感)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        s = np.clip(s * saturation_boost, 0, 255)
        v = np.clip(v * value_boost, 0, 255)
        
        hsv_enhanced = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def generate_linear_mask(self,
                            img_shape: Tuple[int, int],
                            focus_position: float = 0.5,
                            direction: str = "vertical",
                            focus_width: float = 0.2,
                            falloff_power: float = 2.0) -> np.ndarray:
        """
        生成纯几何线性对焦 Mask (不依赖深度估计)
        
        适用场景: 垂直俯拍、建筑摄影等深度估计失效的场景
        
        Args:
            img_shape: 图像形状 (H, W)
            focus_position: 焦点位置 (0.0-1.0), 0=最上/左, 1=最下/右
            direction: 渐变方向
                - "vertical": 垂直渐变 (从上到下)
                - "horizontal": 水平渐变 (从左到右)
                - "radial": 径向渐变 (从中心向外)
                - "diagonal_tlbr": 对角线渐变 (左上到右下)
                - "diagonal_trbl": 对角线渐变 (右上到左下)
            focus_width: 焦点区域宽度 (0.0-1.0), 值越大清晰区域越宽
            falloff_power: 衰减指数 (控制过渡锐度), 值越大过渡越陡峭
            
        Returns:
            mask: 对焦 Mask (H, W), 值范围 0.0-1.0, 1=清晰, 0=模糊
        """
        h, w = img_shape
        
        if direction == "vertical":
            # 垂直渐变: 从上(0)到下(1)
            coord = np.linspace(0, 1, h)[:, np.newaxis].repeat(w, axis=1)
            
        elif direction == "horizontal":
            # 水平渐变: 从左(0)到右(1)
            coord = np.linspace(0, 1, w)[np.newaxis, :].repeat(h, axis=0)
            
        elif direction == "radial":
            # 径向渐变: 从中心(0)向外(1)
            y = np.linspace(-1, 1, h)[:, np.newaxis]
            x = np.linspace(-1, 1, w)[np.newaxis, :]
            coord = np.sqrt(x**2 + y**2) / np.sqrt(2)  # 归一化到 0-1
            focus_position = 1.0 - focus_position  # 反转: 0=边缘清晰, 1=中心清晰
            
        elif direction == "diagonal_tlbr":
            # 对角线渐变: 左上(0)到右下(1)
            y = np.linspace(0, 1, h)[:, np.newaxis]
            x = np.linspace(0, 1, w)[np.newaxis, :]
            coord = (x + y) / 2.0
            
        elif direction == "diagonal_trbl":
            # 对角线渐变: 右上(0)到左下(1)
            y = np.linspace(0, 1, h)[:, np.newaxis]
            x = np.linspace(1, 0, w)[np.newaxis, :]
            coord = (x + y) / 2.0
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        
        # 计算到焦点的归一化距离
        distance = np.abs(coord - focus_position)
        
        # 使用高斯型衰减函数 (比 Sigmoid 更平滑)
        # mask = exp(-(distance / (focus_width/2))^falloff_power)
        sigma = focus_width / 2.0
        mask = np.exp(-np.power(distance / (sigma + 1e-6), falloff_power))
        
        return mask.astype(np.float32)

    def generate_scheimpflug_mask(self, 
                                   depth_map: np.ndarray,
                                   focus_depth: float = 0.5,
                                   tilt_angle_x: float = 0.0,
                                   tilt_angle_y: float = 0.0,
                                   depth_of_field: float = 0.15) -> np.ndarray:
        """
        基于沙姆定律生成焦平面 Mask
        
        原理: 构建像素的伪 3D 坐标 (u, v, depth)，计算每个点到倾斜 3D 平面的垂直距离
        
        Args:
            depth_map: 归一化深度图 (H, W), 值范围 0-1
            focus_depth: 基础焦点深度 (0.0-1.0)
            tilt_angle_x: 俯仰角 (度), 正值=地面清晰/天空模糊, 负值=反向
            tilt_angle_y: 摇摆角 (度), 正值=左侧清晰/右侧模糊, 负值=反向
            depth_of_field: 景深范围，值越大焦点区域越宽
            
        Returns:
            mask: 对焦 Mask (0.0-1.0), 1=清晰, 0=模糊
        """
        h, w = depth_map.shape
        
        # 1. 构建归一化的像素坐标网格 (u, v 范围 -0.5 到 0.5)
        u = np.linspace(-0.5, 0.5, w)
        v = np.linspace(-0.5, 0.5, h)
        uu, vv = np.meshgrid(u, v)
        
        # 2. 将角度转换为弧度
        theta_x = np.radians(tilt_angle_x)  # 俯仰 (绕 X 轴旋转)
        theta_y = np.radians(tilt_angle_y)  # 摇摆 (绕 Y 轴旋转)
        
        # 3. 计算倾斜焦平面的法向量
        # 初始法向量为 (0, 0, 1) - 指向相机
        # 绕 X 轴旋转 theta_x: 影响 y-z 平面
        # 绕 Y 轴旋转 theta_y: 影响 x-z 平面
        nx = np.sin(theta_y)
        ny = -np.sin(theta_x) * np.cos(theta_y)
        nz = np.cos(theta_x) * np.cos(theta_y)
        
        # 归一化法向量
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        # 4. 焦平面过点 (0, 0, focus_depth)
        # 平面方程: nx*(x-0) + ny*(y-0) + nz*(z-focus_depth) = 0
        # 即: nx*x + ny*y + nz*z = nz*focus_depth
        d = nz * focus_depth
        
        # 5. 计算每个像素点 (u, v, depth) 到平面的带符号距离
        # 距离 = (nx*u + ny*v + nz*depth - d) / |n| = nx*u + ny*v + nz*depth - d
        signed_distance = nx * uu + ny * vv + nz * depth_map - d
        
        # 6. 使用绝对距离生成 Mask (距离越小越清晰)
        abs_distance = np.abs(signed_distance)
        
        # 7. 使用 Sigmoid 函数生成平滑过渡的 Mask
        # mask = 1 / (1 + exp(k * (distance - threshold)))
        k = 10.0 / depth_of_field  # 控制过渡锐度
        mask = 1.0 / (1.0 + np.exp(k * (abs_distance - depth_of_field * 0.5)))
        
        return mask.astype(np.float32)

    def warp_perspective(self, 
                         img: np.ndarray, 
                         shift_correction: float = 0.0,
                         direction: str = "vertical") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        透视校正 (Virtual Shift) - 模拟移轴镜头的平移功能
        
        通过梯形校正将建筑线条拉直，消除仰拍时的透视变形
        
        Args:
            img: 输入图像 (H, W, 3)
            shift_correction: 校正强度 (-1.0 到 1.0)
                              正值: 校正仰拍变形 (建筑向上收窄)
                              负值: 校正俯拍变形 (建筑向下收窄)
            direction: 校正方向 "vertical" (垂直) 或 "horizontal" (水平)
            
        Returns:
            warped_img: 校正后的图像
            inverse_matrix: 逆变换矩阵 (用于后续反向映射)
        """
        if abs(shift_correction) < 0.01:
            return img, None
            
        h, w = img.shape[:2]
        
        # 计算偏移量 (最大为图像宽/高的 20%)
        max_offset = 0.2
        offset = int(w * max_offset * abs(shift_correction)) if direction == "vertical" else int(h * max_offset * abs(shift_correction))
        
        if direction == "vertical":
            # 垂直校正 (处理仰拍/俯拍)
            if shift_correction > 0:
                # 仰拍校正: 上窄下宽 -> 矩形
                src_pts = np.float32([
                    [offset, 0],           # 左上
                    [w - offset, 0],       # 右上  
                    [w, h],                # 右下
                    [0, h]                 # 左下
                ])
            else:
                # 俯拍校正: 上宽下窄 -> 矩形
                src_pts = np.float32([
                    [0, 0],                # 左上
                    [w, 0],                # 右上
                    [w - offset, h],       # 右下
                    [offset, h]            # 左下
                ])
        else:
            # 水平校正 (处理侧拍)
            if shift_correction > 0:
                # 左窄右宽 -> 矩形
                src_pts = np.float32([
                    [0, offset],           # 左上
                    [w, 0],                # 右上
                    [w, h],                # 右下
                    [0, h - offset]        # 左下
                ])
            else:
                # 左宽右窄 -> 矩形
                src_pts = np.float32([
                    [0, 0],                # 左上
                    [w, offset],           # 右上
                    [w, h - offset],       # 右下
                    [0, h]                 # 左下
                ])
        
        # 目标矩形
        dst_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        
        # 应用变换
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return warped, M_inv

    def apply_depth_aware_blur(self,
                                img: np.ndarray,
                                mask: np.ndarray,
                                max_blur_radius: int = 25,
                                blur_levels: int = 8) -> np.ndarray:
        """
        基于 Mask 的多级模糊 (模拟真实景深效果)
        
        Args:
            img: 输入图像
            mask: 对焦 Mask (0-1), 1=清晰, 0=最大模糊
            max_blur_radius: 最大模糊半径
            blur_levels: 模糊级数 (越多过渡越平滑)
            
        Returns:
            blurred: 模糊后的图像
        """
        result = img.astype(np.float32)
        
        # 多级模糊叠加
        for i in range(blur_levels):
            # 计算当前级别的模糊半径
            level_ratio = (i + 1) / blur_levels
            blur_radius = int(max_blur_radius * level_ratio)
            if blur_radius < 1:
                continue
            k_size = blur_radius * 2 + 1
            
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(img, (k_size, k_size), 0)
            
            # 计算当前级别的权重 Mask
            # mask < level_ratio 的区域应用此级别模糊
            threshold_low = (i) / blur_levels
            threshold_high = (i + 1) / blur_levels
            
            # 计算混合权重
            level_mask = np.clip((threshold_high - mask) / (1.0 / blur_levels + 1e-6), 0, 1)
            level_mask = level_mask[:, :, np.newaxis]
            
            # 渐进式混合
            result = result * (1 - level_mask) + blurred.astype(np.float32) * level_mask
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def render_frame(self, 
                     original_img: np.ndarray, 
                     filled_bg: np.ndarray, 
                     mask: np.ndarray, 
                     blur_radius: int = 15,
                     transition_smoothness: float = 1.0,
                     color_enhance: bool = True,
                     saturation_boost: float = 1.4,
                     value_boost: float = 1.1) -> np.ndarray:
        """
        渲染单帧 (兼容旧接口)
        """
        # 1. 背景高斯模糊
        k_size = blur_radius * 2 + 1
        bg_blurred = cv2.GaussianBlur(filled_bg, (k_size, k_size), 0)
        
        # 2. 过渡平滑处理
        smooth_k = int(blur_radius * transition_smoothness) * 2 + 1
        smooth_k = max(3, smooth_k)
        mask_smooth = cv2.GaussianBlur(mask, (smooth_k, smooth_k), 0)
        
        gamma = 1.0 + (transition_smoothness - 1.0) * 0.5
        gamma = np.clip(gamma, 0.5, 2.0)
        mask_smooth = np.power(mask_smooth, gamma)
        
        # 3. Alpha Blending
        mask_3c = np.repeat(mask_smooth[:, :, np.newaxis], 3, axis=2)
        final_img = original_img.astype(np.float32) * mask_3c + bg_blurred.astype(np.float32) * (1.0 - mask_3c)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        # 4. 色彩增强
        if color_enhance:
            final_img = self.enhance_color(final_img, saturation_boost, value_boost)
            
        return final_img

    def render_scheimpflug(self,
                           img: np.ndarray,
                           depth_map: np.ndarray,
                           focus_depth: float = 0.5,
                           tilt_angle_x: float = 0.0,
                           tilt_angle_y: float = 0.0,
                           shift_correction: float = 0.0,
                           shift_direction: str = "vertical",
                           depth_of_field: float = 0.15,
                           max_blur_radius: int = 25,
                           color_enhance: bool = True,
                           saturation_boost: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        沙姆定律完整渲染流程
        
        Args:
            img: 输入图像 BGR (H, W, 3)
            depth_map: 深度图 (H, W), 归一化 0-1
            focus_depth: 焦点深度 (0-1)
            tilt_angle_x: 俯仰角 (度), -45 到 45
            tilt_angle_y: 摇摆角 (度), -45 到 45
            shift_correction: 透视校正强度 (-1 到 1)
            shift_direction: 校正方向 "vertical" 或 "horizontal"
            depth_of_field: 景深范围 (0.05 - 0.5)
            max_blur_radius: 最大模糊半径
            color_enhance: 是否色彩增强
            saturation_boost: 饱和度增强系数
            
        Returns:
            result: 渲染结果图像
            mask: 生成的焦平面 Mask (用于可视化)
        """
        # 1. 透视校正 (Virtual Shift)
        if abs(shift_correction) > 0.01:
            img_corrected, M_inv = self.warp_perspective(img, shift_correction, shift_direction)
            # 深度图也需要同样的变换
            depth_corrected, _ = self.warp_perspective(
                (depth_map * 255).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2),
                shift_correction, 
                shift_direction
            )
            depth_corrected = depth_corrected[:, :, 0].astype(np.float32) / 255.0
        else:
            img_corrected = img
            depth_corrected = depth_map
            M_inv = None
        
        # 2. 生成沙姆定律焦平面 Mask (Virtual Tilt)
        mask = self.generate_scheimpflug_mask(
            depth_corrected,
            focus_depth=focus_depth,
            tilt_angle_x=tilt_angle_x,
            tilt_angle_y=tilt_angle_y,
            depth_of_field=depth_of_field
        )
        
        # 3. 基于 Mask 应用多级景深模糊
        result = self.apply_depth_aware_blur(
            img_corrected,
            mask,
            max_blur_radius=max_blur_radius,
            blur_levels=8
        )
        
        # 4. 色彩增强 (微缩模型感)
        if color_enhance:
            result = self.enhance_color(result, saturation_boost=saturation_boost)
        
        return result, mask

    def render_linear(self,
                      img: np.ndarray,
                      focus_position: float = 0.5,
                      direction: str = "vertical",
                      focus_width: float = 0.2,
                      falloff_power: float = 2.0,
                      max_blur_radius: int = 25,
                      color_enhance: bool = True,
                      saturation_boost: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        纯几何线性对焦模式渲染流程 (不依赖深度估计)
        
        适用场景: 垂直俯拍街道、卫星图、建筑立面等深度估计失效的场景
        
        Args:
            img: 输入图像 BGR (H, W, 3)
            focus_position: 焦点位置 (0.0-1.0)
            direction: 渐变方向 (vertical/horizontal/radial/diagonal_tlbr/diagonal_trbl)
            focus_width: 焦点区域宽度 (0.0-1.0)
            falloff_power: 衰减指数 (控制过渡锐度)
            max_blur_radius: 最大模糊半径
            color_enhance: 是否色彩增强
            saturation_boost: 饱和度增强系数
            
        Returns:
            result: 渲染结果图像
            mask: 生成的线性 Mask (用于可视化)
        """
        h, w = img.shape[:2]
        
        # 1. 生成纯几何 Mask
        mask = self.generate_linear_mask(
            img_shape=(h, w),
            focus_position=focus_position,
            direction=direction,
            focus_width=focus_width,
            falloff_power=falloff_power
        )
        
        # 2. 基于 Mask 应用多级景深模糊
        result = self.apply_depth_aware_blur(
            img,
            mask,
            max_blur_radius=max_blur_radius,
            blur_levels=8
        )
        
        # 3. 色彩增强 (微缩模型感)
        if color_enhance:
            result = self.enhance_color(result, saturation_boost=saturation_boost)
        
        return result, mask
