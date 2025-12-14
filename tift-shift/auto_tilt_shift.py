import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import argparse
import os

class AutoTiltShift:
    def __init__(self):
        """
        初始化自动移轴摄影生成器
        加载 DeepLabV3 模型用于显著性检测
        """
        print("正在加载 DeepLabV3 模型...")
        # 使用预训练的 DeepLabV3 ResNet50 模型
        # weights='DEFAULT' 会加载最新的预训练权重
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.model.eval()
        
        # 检查是否有 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        
        self.model.to(self.device)
        print(f"模型加载完成，运行设备: {self.device}")

        # 标准 ImageNet 预处理
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_saliency(self, image_pil):
        """
        第一步：智能对焦（显著性检测/分割）
        输入: PIL Image
        输出: 焦点所在的 Y 轴坐标 (int)
        """
        # 预处理并添加 batch 维度
        input_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # 获取每个像素的类别预测 (H, W)
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # 寻找面积最大的感兴趣物体
        # 类别 0 通常是背景，我们忽略它
        unique_classes, counts = np.unique(output_predictions, return_counts=True)
        
        max_area = 0
        target_class = None

        for cls, count in zip(unique_classes, counts):
            if cls == 0: # 忽略背景
                continue
            if count > max_area:
                max_area = count
                target_class = cls
        
        height, width = output_predictions.shape

        if target_class is None:
            print("未检测到显著前景物体，默认焦点在图像中心。")
            return height // 2

        # 生成该类别的二值掩码
        mask = (output_predictions == target_class).astype(np.uint8)

        # 计算质心
        # 使用 OpenCV 的 moments 计算质心
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"检测到最大物体类别ID: {target_class}, 面积: {max_area}, 质心: ({cX}, {cY})")
            return cY
        else:
            return height // 2

    def generate_blur_mask(self, shape, focus_y):
        """
        第二步：生成可变模糊蒙版
        输入: 图像形状 (H, W), 焦点 Y 坐标
        输出: 模糊蒙版 (H, W), 值范围 0.0 (清晰) - 1.0 (模糊)
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        # 创建垂直方向的梯度
        # 每一行的值取决于它距离 focus_y 的距离
        y_indices = np.arange(h).reshape(h, 1)
        # 扩展到宽度
        y_indices = np.repeat(y_indices, w, axis=1)

        # 计算归一化距离
        # 距离越远，值越大。最大距离可能是 focus_y 到 0，或者 focus_y 到 h
        max_dist_top = focus_y
        max_dist_bottom = h - focus_y
        
        # 分别处理上半部分和下半部分，确保边缘处达到 1.0
        # 上半部分 (y < focus_y)
        mask[0:focus_y, :] = (focus_y - y_indices[0:focus_y, :]) / max_dist_top
        # 下半部分 (y >= focus_y)
        mask[focus_y:h, :] = (y_indices[focus_y:h, :] - focus_y) / max_dist_bottom

        # 确保范围在 0-1
        mask = np.clip(mask, 0, 1)

        # 非线性调整：模拟景深衰减 (Gradient ** 2)
        mask = np.power(mask, 2)

        return mask

    def apply_variable_blur(self, image, mask):
        """
        第三步：应用可变高斯模糊 (分层模糊)
        输入: 原始图像 (BGR), 模糊蒙版 (0-1)
        输出: 模糊后的图像
        """
        # 生成不同强度的模糊层
        # 1. 轻微模糊
        blur_light = cv2.GaussianBlur(image, (7, 7), 0)
        # 2. 中度模糊
        blur_medium = cv2.GaussianBlur(image, (15, 15), 0)
        # 3. 强力模糊
        blur_strong = cv2.GaussianBlur(image, (31, 31), 0)

        # 将 mask 扩展为 3 通道以便与图像运算
        mask_3c = cv2.merge([mask, mask, mask])

        # 初始化输出图像
        output = np.zeros_like(image, dtype=np.float32)
        img_float = image.astype(np.float32)
        
        # 分层混合逻辑
        # 定义模糊过渡的阈值
        t1 = 0.33
        t2 = 0.66

        # 区域 1: 清晰 -> 轻微模糊 (Mask: 0 -> t1)
        # 归一化权重 w = mask / t1
        # result = img * (1-w) + blur_light * w
        region1 = (mask_3c < t1)
        w1 = mask_3c / t1
        # 避免除零（虽然 mask < t1 包含 0，但 numpy 处理 float 除法通常没问题，或者加个 epsilon）
        w1 = np.nan_to_num(w1) 
        
        out1 = img_float * (1 - w1) + blur_light.astype(np.float32) * w1
        
        # 区域 2: 轻微模糊 -> 中度模糊 (Mask: t1 -> t2)
        # 归一化权重 w = (mask - t1) / (t2 - t1)
        region2 = (mask_3c >= t1) & (mask_3c < t2)
        w2 = (mask_3c - t1) / (t2 - t1)
        out2 = blur_light.astype(np.float32) * (1 - w2) + blur_medium.astype(np.float32) * w2

        # 区域 3: 中度模糊 -> 强力模糊 (Mask: t2 -> 1.0)
        # 归一化权重 w = (mask - t2) / (1.0 - t2)
        region3 = (mask_3c >= t2)
        w3 = (mask_3c - t2) / (1.0 - t2)
        out3 = blur_medium.astype(np.float32) * (1 - w3) + blur_strong.astype(np.float32) * w3

        # 组合结果
        output = np.where(region1, out1, output)
        output = np.where(region2, out2, output)
        output = np.where(region3, out3, output)

        return output.astype(np.uint8)

    def enhance_color(self, image):
        """
        第四步：微缩模型色彩增强
        输入: BGR 图像
        输出: 增强后的 BGR 图像
        """
        # 转换到 HSV 空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # H, S, V 分量
        h, s, v = cv2.split(hsv)

        # 提升饱和度 (Saturation) * 1.4
        s = s * 1.4
        s = np.clip(s, 0, 255)

        # 适度提升亮度/对比度 (Value) * 1.1
        v = v * 1.1
        v = np.clip(v, 0, 255)

        # 合并回 HSV
        hsv_enhanced = cv2.merge([h, s, v])

        # 转换回 BGR
        output = cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return output

    def run(self, image_path, output_path=None):
        """
        主流程
        """
        if not os.path.exists(image_path):
            print(f"错误: 找不到文件 {image_path}")
            return

        print(f"正在处理图像: {image_path}")
        
        # 1. 读取图像
        # OpenCV 读取用于处理
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            print("无法读取图像")
            return
        
        # PIL 读取用于模型推理 (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        # 2. 智能对焦
        print("Step 1: 正在进行显著性检测...")
        focus_y = self.detect_saliency(img_pil)
        print(f"焦点行坐标: {focus_y} (图像高度: {img_cv2.shape[0]})")

        # 3. 生成模糊蒙版
        print("Step 2: 生成可变模糊蒙版...")
        blur_mask = self.generate_blur_mask(img_cv2.shape, focus_y)
        
        # 保存 mask 用于调试 (可选)
        # cv2.imwrite("debug_mask.jpg", (blur_mask * 255).astype(np.uint8))

        # 4. 应用模糊
        print("Step 3: 应用分层可变模糊...")
        blurred_img = self.apply_variable_blur(img_cv2, blur_mask)

        # 5. 色彩增强
        print("Step 4: 进行微缩模型色彩增强...")
        final_img = self.enhance_color(blurred_img)

        # 6. 保存结果
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_tiltshift{ext}"
        
        cv2.imwrite(output_path, final_img)
        print(f"处理完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于显著性目标检测的自动移轴摄影生成器")
    parser.add_argument("image_path", type=str, help="输入图片的路径")
    parser.add_argument("--output", type=str, default=None, help="输出图片的路径 (可选)")
    
    args = parser.parse_args()
    
    processor = AutoTiltShift()
    processor.run(args.image_path, args.output)
