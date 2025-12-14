import torch
import cv2
import numpy as np
import argparse
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
from scipy import signal

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        初始化深度估计器。
        为了保证脚本即插即用，默认使用 torch.hub 加载 MiDaS 模型。
        如果需要 Depth Anything V2，可以替换此处的加载逻辑。
        """
        print(f"正在加载深度估计模型: {model_type}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        # 使用 MiDaS 作为 Depth Anything V2 的可运行替代方案
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # MiDaS 的预处理 transform
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def estimate_depth(self, img_cv2):
        """
        输入 OpenCV BGR 图像，输出归一化深度图 (0.0 - 1.0)
        """
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # 调整大小回原图分辨率
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # 归一化到 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_norm

    def extract_foreground_mask(self, depth_map, focus_depth, depth_range=0.2):
        """
        根据焦点深度和范围提取前景 Mask。
        模拟 Alpha Matting 的粗略效果。
        """
        # 计算距离焦点的距离
        dist = np.abs(depth_map - focus_depth)
        
        # 创建二值 Mask：在焦平面范围内的为前景
        # 这里的逻辑是：深度值接近 focus_depth 的是前景
        # 注意：MiDaS 输出的是逆深度（值越大越近），所以 focus_depth 越大越近
        mask = np.where(dist < depth_range, 1.0, 0.0).astype(np.float32)
        
        # 边缘平滑 (Soft Matting simulation)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask

class Inpainter:
    def __init__(self, method="navier-stokes"):
        self.method = method

    def remove_foreground_and_fill(self, img, mask):
        """
        背景解耦与生成式修复。
        将前景挖空，并修复背景，防止边缘伪影。
        """
        print("正在进行背景修复 (Inpainting)...")
        
        # 1. 准备 Inpainting Mask
        # Mask 中值为 1 的是前景，需要被挖掉并修复
        # 需要稍微膨胀 Mask，确保边缘也被覆盖，防止颜色残留
        inp_mask = (mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_mask = cv2.dilate(inp_mask, kernel, iterations=2)

        # 2. 执行 Inpainting
        # 使用 Navier-Stokes 算法 (cv2.INPAINT_NS) 或 Telea (cv2.INPAINT_TELEA)
        # 这是一个轻量级的替代方案，优于 LaMa 的是它不需要额外权重文件
        if self.method == "telea":
            bg_filled = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)
        else:
            bg_filled = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_NS)
            
        return bg_filled

class Renderer:
    def __init__(self):
        pass

    def generate_bokeh_kernel(self, size=15, shape="disk"):
        """
        生成自定义的 Bokeh 卷积核
        """
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        radius = size // 2

        if shape == "disk":
            cv2.circle(kernel, (center, center), radius, 1, -1)
        elif shape == "hexagon":
            # 简单的六边形模拟
            pts = np.array([
                [center + radius, center],
                [center + radius//2, center + int(radius*0.866)],
                [center - radius//2, center + int(radius*0.866)],
                [center - radius, center],
                [center - radius//2, center - int(radius*0.866)],
                [center + radius//2, center - int(radius*0.866)]
            ], np.int32)
            cv2.fillPoly(kernel, [pts], 1)
            
        # 归一化，保证能量守恒
        kernel /= np.sum(kernel)
        return kernel

    def render_bokeh_highlights(self, img, mask, threshold=220, intensity=1.5, kernel_size=21):
        """
        提取高光并应用 Bokeh 形状卷积
        """
        # 1. 提取高光区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, highlights = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # 仅保留背景区域的高光 (可选，或者全图高光)
        # highlights = cv2.bitwise_and(highlights, highlights, mask=(1-mask).astype(np.uint8))
        
        if cv2.countNonZero(highlights) == 0:
            return np.zeros_like(img, dtype=np.float32)

        # 2. 准备 Bokeh 核
        kernel = self.generate_bokeh_kernel(size=kernel_size, shape="hexagon")
        
        # 3. 对高光进行卷积
        # 分离通道处理
        bokeh_layer = np.zeros_like(img, dtype=np.float32)
        highlights_float = highlights.astype(np.float32) / 255.0
        
        # 增强高光强度
        highlights_float *= intensity

        # 使用 filter2D 模拟散焦
        # 为了性能，可以先缩小再卷积再放大，或者直接卷积
        for i in range(3): # B, G, R
            # 仅对高光部分卷积
            channel_highlight = img[:, :, i].astype(np.float32) * highlights_float
            bokeh_layer[:, :, i] = cv2.filter2D(channel_highlight, -1, kernel)

        return bokeh_layer

    def render(self, original_img, filled_bg, mask, blur_radius=15):
        """
        分层物理渲染
        """
        print("正在进行分层渲染...")
        h, w = original_img.shape[:2]
        
        # 1. 背景层模糊 (Disk Blur)
        # 使用生成的背景图进行模糊，这样边缘处就不会有前景的颜色渗入
        # 模拟大光圈效果
        kernel_size = blur_radius * 2 + 1
        bg_blurred = cv2.GaussianBlur(filled_bg, (kernel_size, kernel_size), 0)
        # 也可以用 disk kernel filter2D 获得更真实的散焦，但 Gaussian 比较快
        
        # 2. 生成 Bokeh 光斑层
        bokeh_layer = self.render_bokeh_highlights(filled_bg, mask, threshold=200, kernel_size=kernel_size)
        
        # 将 Bokeh 叠加到模糊背景上 (Add 模式)
        bg_final = bg_blurred.astype(np.float32) + bokeh_layer
        bg_final = np.clip(bg_final, 0, 255)

        # 3. 合成 (Compositing)
        # Result = Foreground * Alpha + Background * (1 - Alpha)
        # Mask: 1.0 = Foreground, 0.0 = Background
        
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        foreground = original_img.astype(np.float32)
        
        # 核心：无伪影合成
        # 前景保持清晰，背景使用已修复且模糊的版本
        final_image = foreground * mask_3c + bg_final * (1.0 - mask_3c)
        
        return final_image.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="基于MPI和背景修复的无伪影移轴渲染器")
    parser.add_argument("image_path", type=str, help="输入图片路径")
    parser.add_argument("--focus_depth", type=float, default=0.8, help="焦点深度 (0.0-1.0), 越大越近")
    parser.add_argument("--blur_radius", type=int, default=15, help="背景模糊半径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print("错误：找不到输入文件")
        return

    # 1. 加载图像
    img = cv2.imread(args.image_path)
    if img is None:
        print("无法读取图像")
        return
    
    # 2. 深度估计
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate_depth(img)
    
    # 保存深度图用于调试
    cv2.imwrite("debug_depth.jpg", (depth_map * 255).astype(np.uint8))
    print("深度图已生成: debug_depth.jpg")

    # 3. 提取前景 Mask (Trimap / Alpha Matte)
    mask = depth_estimator.extract_foreground_mask(depth_map, args.focus_depth)
    cv2.imwrite("debug_mask.jpg", (mask * 255).astype(np.uint8))
    print("前景 Mask 已生成: debug_mask.jpg")

    # 4. 背景修复 (Inpainting)
    inpainter = Inpainter()
    bg_filled = inpainter.remove_foreground_and_fill(img, mask)
    cv2.imwrite("debug_bg_filled.jpg", bg_filled)
    print("背景修复完成: debug_bg_filled.jpg")

    # 5. 渲染与合成
    renderer = Renderer()
    result = renderer.render(img, bg_filled, mask, blur_radius=args.blur_radius)

    # 6. 保存结果
    if args.output is None:
        name, ext = os.path.splitext(args.image_path)
        output_path = f"{name}_mpi_tiltshift{ext}"
    else:
        output_path = args.output
        
    cv2.imwrite(output_path, result)
    print(f"渲染完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
