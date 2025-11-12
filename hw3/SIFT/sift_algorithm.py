"""
SIFT算法核心实现
从零开始实现SIFT特征检测和描述符生成
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MySIFT:
    """自实现的SIFT算法类"""
    
    def __init__(self, num_octaves=4, num_scales=5, sigma=1.6, 
                 contrast_threshold=0.04, edge_threshold=10, 
                 lambda_ori=1.5, lambda_desc=6):
        """
        初始化SIFT检测器
        
        Args:
            num_octaves: 金字塔组数
            num_scales: 每组的尺度数
            sigma: 初始高斯核标准差
            contrast_threshold: 对比度阈值
            edge_threshold: 边缘响应阈值
            lambda_ori: 方向分配时的窗口大小系数
            lambda_desc: 描述符计算时的窗口大小系数
        """
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.lambda_ori = lambda_ori
        self.lambda_desc = lambda_desc
        
        # 计算高斯模糊的sigma值
        self.k = 2 ** (1.0 / (num_scales - 3))
        self.sigmas = self._compute_sigmas()
        
    def _compute_sigmas(self):
        """计算每个尺度的sigma值"""
        sigmas = np.zeros(self.num_scales)
        sigmas[0] = self.sigma
        
        for i in range(1, self.num_scales):
            sigma_prev = self.sigma * (self.k ** (i - 1))
            sigma_total = self.k * sigma_prev
            sigmas[i] = np.sqrt(sigma_total ** 2 - sigma_prev ** 2)
        
        return sigmas
    
    def build_gaussian_pyramid(self, image):
        """
        构建高斯金字塔
        
        Args:
            image: 输入灰度图像
            
        Returns:
            gaussian_pyramid: 高斯金字塔 [octave][scale]
        """
        print("正在构建高斯金字塔...")
        gaussian_pyramid = []
        
        # 对初始图像进行上采样以获得更好的特征检测
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), 
                          interpolation=cv2.INTER_LINEAR)
        
        # 对初始图像进行高斯模糊
        base_image = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        
        for octave in range(self.num_octaves):
            octave_images = []
            
            for scale in range(self.num_scales):
                if octave == 0 and scale == 0:
                    octave_images.append(base_image)
                else:
                    sigma = self.sigmas[scale]
                    blurred = cv2.GaussianBlur(octave_images[-1] if scale > 0 else base_image, 
                                              (0, 0), sigmaX=sigma, sigmaY=sigma)
                    octave_images.append(blurred)
            
            gaussian_pyramid.append(octave_images)
            
            # 下采样准备下一组
            if octave < self.num_octaves - 1:
                base_image = cv2.resize(octave_images[-3], 
                                       (octave_images[-3].shape[1] // 2, 
                                        octave_images[-3].shape[0] // 2),
                                       interpolation=cv2.INTER_NEAREST)
        
        return gaussian_pyramid
    
    def build_dog_pyramid(self, gaussian_pyramid):
        """
        构建DOG（高斯差分）金字塔
        
        Args:
            gaussian_pyramid: 高斯金字塔
            
        Returns:
            dog_pyramid: DOG金字塔
        """
        print("正在构建DOG金字塔...")
        dog_pyramid = []
        
        for octave_images in gaussian_pyramid:
            octave_dogs = []
            for i in range(len(octave_images) - 1):
                dog = cv2.subtract(octave_images[i + 1], octave_images[i])
                octave_dogs.append(dog)
            dog_pyramid.append(octave_dogs)
        
        return dog_pyramid
    
    def find_extrema(self, dog_pyramid):
        """
        在DOG金字塔中寻找极值点
        
        Args:
            dog_pyramid: DOG金字塔
            
        Returns:
            keypoints: 极值点列表 [(octave, scale, y, x, value), ...]
        """
        print("正在检测极值点...")
        keypoints = []
        
        for octave_idx, octave_dogs in enumerate(dog_pyramid):
            for scale_idx in range(1, len(octave_dogs) - 1):
                # 获取当前层和相邻层
                prev_dog = octave_dogs[scale_idx - 1]
                curr_dog = octave_dogs[scale_idx]
                next_dog = octave_dogs[scale_idx + 1]
                
                # 遍历图像（除了边界）
                for i in range(1, curr_dog.shape[0] - 1):
                    for j in range(1, curr_dog.shape[1] - 1):
                        # 获取3x3x3邻域
                        center_value = curr_dog[i, j]
                        
                        # 检查是否为极值（最大值或最小值）
                        neighborhood = np.array([
                            prev_dog[i-1:i+2, j-1:j+2],
                            curr_dog[i-1:i+2, j-1:j+2],
                            next_dog[i-1:i+2, j-1:j+2]
                        ])
                        
                        is_max = center_value == np.max(neighborhood)
                        is_min = center_value == np.min(neighborhood)
                        
                        # 对比度阈值检查
                        if (is_max or is_min) and abs(center_value) > 0.5 * self.contrast_threshold:
                            keypoints.append((octave_idx, scale_idx, i, j, center_value))
        
        print(f"检测到 {len(keypoints)} 个初始极值点")
        return keypoints
    
    def refine_keypoints(self, keypoints, dog_pyramid):
        """
        精确定位关键点（亚像素级）并去除低对比度和边缘响应点
        
        Args:
            keypoints: 初始极值点
            dog_pyramid: DOG金字塔
            
        Returns:
            refined_keypoints: 精确定位后的关键点
        """
        print("正在精确定位关键点...")
        refined_keypoints = []
        
        for octave, scale, y, x, value in keypoints:
            # 获取当前DOG图像
            dog = dog_pyramid[octave][scale]
            
            # 边界检查
            if y <= 1 or y >= dog.shape[0] - 2 or x <= 1 or x >= dog.shape[1] - 2:
                continue
            
            # 计算Hessian矩阵去除边缘响应
            dxx = dog[y, x+1] + dog[y, x-1] - 2 * dog[y, x]
            dyy = dog[y+1, x] + dog[y-1, x] - 2 * dog[y, x]
            dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4.0
            
            trace = dxx + dyy
            det = dxx * dyy - dxy * dxy
            
            # 边缘响应检查
            if det <= 0 or (trace * trace / det) >= ((self.edge_threshold + 1) ** 2 / self.edge_threshold):
                continue
            
            # 对比度阈值检查
            if abs(dog[y, x]) < self.contrast_threshold:
                continue
            
            refined_keypoints.append({
                'octave': octave,
                'scale': scale,
                'y': y,
                'x': x,
                'response': abs(dog[y, x])
            })
        
        print(f"精确定位后保留 {len(refined_keypoints)} 个关键点")
        return refined_keypoints
    
    def compute_orientations(self, keypoints, gaussian_pyramid):
        """
        为每个关键点计算主方向
        
        Args:
            keypoints: 关键点列表
            gaussian_pyramid: 高斯金字塔
            
        Returns:
            keypoints_with_orientations: 带方向的关键点
        """
        print("正在计算关键点方向...")
        keypoints_with_orientations = []
        
        for kp in keypoints:
            octave = kp['octave']
            scale = kp['scale']
            y, x = kp['y'], kp['x']
            
            # 获取对应的高斯图像
            gaussian_image = gaussian_pyramid[octave][scale]
            
            # 边界检查
            radius = int(round(self.lambda_ori * self.sigma * (self.k ** scale)))
            if y - radius < 1 or y + radius >= gaussian_image.shape[0] - 1 or \
               x - radius < 1 or x + radius >= gaussian_image.shape[1] - 1:
                continue
            
            # 计算梯度幅值和方向
            dy = gaussian_image[y+1, x] - gaussian_image[y-1, x]
            dx = gaussian_image[y, x+1] - gaussian_image[y, x-1]
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = np.arctan2(dy, dx) * 180 / np.pi
            
            # 方向直方图（36个bins）
            hist_bins = 36
            hist = np.zeros(hist_bins)
            
            # 在邻域内统计方向
            for i in range(max(1, y - radius), min(gaussian_image.shape[0] - 1, y + radius + 1)):
                for j in range(max(1, x - radius), min(gaussian_image.shape[1] - 1, x + radius + 1)):
                    dy_local = gaussian_image[i+1, j] - gaussian_image[i-1, j]
                    dx_local = gaussian_image[i, j+1] - gaussian_image[i, j-1]
                    mag = np.sqrt(dx_local**2 + dy_local**2)
                    ori = np.arctan2(dy_local, dx_local) * 180 / np.pi
                    
                    # 高斯加权
                    weight = np.exp(-((i - y)**2 + (j - x)**2) / (2 * (1.5 * self.sigma * self.k ** scale)**2))
                    
                    # 添加到直方图
                    bin_idx = int(np.round((ori + 180) / 360 * hist_bins)) % hist_bins
                    hist[bin_idx] += mag * weight
            
            # 找到主方向（直方图峰值）
            max_bin = np.argmax(hist)
            main_orientation = (max_bin * 360 / hist_bins) - 180
            
            # 添加主方向的关键点
            kp_with_ori = kp.copy()
            kp_with_ori['orientation'] = main_orientation
            kp_with_ori['magnitude'] = magnitude
            keypoints_with_orientations.append(kp_with_ori)
            
            # 添加辅助方向（幅值大于主方向80%的）
            for bin_idx in range(hist_bins):
                if bin_idx != max_bin and hist[bin_idx] > 0.8 * hist[max_bin]:
                    orientation = (bin_idx * 360 / hist_bins) - 180
                    kp_aux = kp.copy()
                    kp_aux['orientation'] = orientation
                    kp_aux['magnitude'] = hist[bin_idx]
                    keypoints_with_orientations.append(kp_aux)
        
        print(f"生成 {len(keypoints_with_orientations)} 个带方向的关键点")
        return keypoints_with_orientations
    
    def compute_descriptors(self, keypoints, gaussian_pyramid):
        """
        为每个关键点计算128维SIFT描述符
        
        Args:
            keypoints: 带方向的关键点
            gaussian_pyramid: 高斯金字塔
            
        Returns:
            descriptors: 描述符数组 (N, 128)
        """
        print(f"正在计算SIFT描述符 (共{len(keypoints)}个关键点)...")
        descriptors = []
        valid_keypoints = []
        
        # 添加进度提示
        total = len(keypoints)
        milestone = max(1, total // 10)  # 每10%显示一次
        
        for idx, kp in enumerate(keypoints):
            # 显示进度
            if idx % milestone == 0 and idx > 0:
                print(f"  进度: {idx}/{total} ({100*idx//total}%)")
            
            octave = kp['octave']
            scale = kp['scale']
            y, x = kp['y'], kp['x']
            orientation = kp['orientation']
            
            # 获取对应的高斯图像
            gaussian_image = gaussian_pyramid[octave][scale]
            
            # 描述符窗口大小（优化：减小窗口以加速）
            window_size = 16
            radius = int(round(window_size * self.lambda_desc * self.sigma * (self.k ** scale) / 2))
            # 限制最大半径以加速计算
            radius = min(radius, 20)
            
            # 边界检查
            if y - radius < 1 or y + radius >= gaussian_image.shape[0] - 1 or \
               x - radius < 1 or x + radius >= gaussian_image.shape[1] - 1:
                continue
            
            # 旋转角度（使描述符具有旋转不变性）
            cos_t = np.cos(np.deg2rad(-orientation))
            sin_t = np.sin(np.deg2rad(-orientation))
            
            # 4x4网格，每个子区域8个方向
            descriptor = np.zeros((4, 4, 8), dtype=np.float32)
            
            # 优化：预先计算高斯权重
            gaussian_window = 0.5 * window_size
            
            # 优化：采样步长（可选：降低采样密度以加速）
            step = 1  # 改为2可以大幅加速，但精度略降
            
            # 在窗口内计算梯度（优化版）
            for i in range(-radius, radius, step):
                for j in range(-radius, radius, step):
                    # 旋转坐标
                    row = int(round(y + i * cos_t - j * sin_t))
                    col = int(round(x + i * sin_t + j * cos_t))
                    
                    # 边界检查
                    if row <= 0 or row >= gaussian_image.shape[0] - 1 or \
                       col <= 0 or col >= gaussian_image.shape[1] - 1:
                        continue
                    
                    # 计算梯度
                    dy = gaussian_image[row+1, col] - gaussian_image[row-1, col]
                    dx = gaussian_image[row, col+1] - gaussian_image[row, col-1]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    
                    # 快速跳过小梯度
                    if magnitude < 1e-5:
                        continue
                    
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    
                    # 相对于主方向的角度
                    angle = (angle - orientation) % 360
                    
                    # 高斯加权（优化：减少exp计算）
                    dist_sq = i*i + j*j
                    weight = magnitude * np.exp(-dist_sq / (2 * gaussian_window * gaussian_window))
                    
                    # 确定在哪个子区域（4x4网格）
                    bin_i = int((i + radius) * 4 / (2 * radius))
                    bin_j = int((j + radius) * 4 / (2 * radius))
                    
                    # 限制在有效范围内
                    bin_i = np.clip(bin_i, 0, 3)
                    bin_j = np.clip(bin_j, 0, 3)
                    
                    # 方向bin（8个方向）
                    angle_bin = int(angle * 8 / 360) % 8
                    
                    # 累加到描述符
                    descriptor[bin_i, bin_j, angle_bin] += weight
            
            # 展平为128维向量
            descriptor_vector = descriptor.flatten()
            
            # 归一化
            norm = np.linalg.norm(descriptor_vector)
            if norm > 0:
                descriptor_vector = descriptor_vector / norm
                
                # 限制最大值为0.2并重新归一化（提高鲁棒性）
                descriptor_vector = np.clip(descriptor_vector, 0, 0.2)
                norm = np.linalg.norm(descriptor_vector)
                if norm > 0:
                    descriptor_vector = descriptor_vector / norm
            
            descriptors.append(descriptor_vector)
            valid_keypoints.append(kp)
        
        descriptors = np.array(descriptors, dtype=np.float32)
        print(f"生成 {len(descriptors)} 个SIFT描述符")
        
        return valid_keypoints, descriptors
    
    def detect_and_compute(self, image):
        """
        检测SIFT特征点并计算描述符（主接口）
        
        Args:
            image: 输入图像（灰度或彩色）
            
        Returns:
            keypoints: OpenCV格式的关键点列表
            descriptors: 描述符数组
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 归一化到[0, 1]
        gray = gray.astype(np.float32) / 255.0
        
        # 1. 构建高斯金字塔
        gaussian_pyramid = self.build_gaussian_pyramid(gray)
        
        # 2. 构建DOG金字塔
        dog_pyramid = self.build_dog_pyramid(gaussian_pyramid)
        
        # 3. 寻找极值点
        extrema = self.find_extrema(dog_pyramid)
        
        # 4. 精确定位关键点
        refined_keypoints = self.refine_keypoints(extrema, dog_pyramid)
        
        # 5. 计算方向
        keypoints_with_ori = self.compute_orientations(refined_keypoints, gaussian_pyramid)
        
        # 6. 计算描述符
        final_keypoints, descriptors = self.compute_descriptors(keypoints_with_ori, gaussian_pyramid)
        
        # 转换为OpenCV格式的KeyPoint对象（用于可视化和匹配）
        cv_keypoints = []
        for kp in final_keypoints:
            # 调整坐标（因为我们对初始图像进行了2倍上采样）
            scale_factor = 2 ** (kp['octave'] + 1)
            pt = (kp['x'] / 2.0, kp['y'] / 2.0)  # 除以2是因为初始上采样
            size = self.sigma * (self.k ** kp['scale']) * scale_factor
            angle = kp['orientation']
            response = kp['response']
            
            cv_kp = cv2.KeyPoint(
                x=pt[0], y=pt[1],
                size=size,
                angle=angle,
                response=response,
                octave=kp['octave']
            )
            cv_keypoints.append(cv_kp)
        
        return cv_keypoints, descriptors, gaussian_pyramid, dog_pyramid


def visualize_pyramids(gaussian_pyramid, dog_pyramid, save_path=None):
    """
    可视化高斯金字塔和DOG金字塔
    
    Args:
        gaussian_pyramid: 高斯金字塔
        dog_pyramid: DOG金字塔
        save_path: 保存路径
    """
    num_octaves = len(gaussian_pyramid)
    num_scales = len(gaussian_pyramid[0])
    
    # 可视化高斯金字塔
    fig, axes = plt.subplots(num_octaves, num_scales, figsize=(15, 10))
    fig.suptitle('高斯金字塔', fontsize=16)
    
    for octave in range(num_octaves):
        for scale in range(num_scales):
            ax = axes[octave, scale] if num_octaves > 1 else axes[scale]
            ax.imshow(gaussian_pyramid[octave][scale], cmap='gray')
            ax.set_title(f'O{octave}-S{scale}')
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_gaussian.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 可视化DOG金字塔
    fig, axes = plt.subplots(num_octaves, num_scales-1, figsize=(15, 10))
    fig.suptitle('DOG金字塔（高斯差分）', fontsize=16)
    
    for octave in range(num_octaves):
        for scale in range(num_scales - 1):
            ax = axes[octave, scale] if num_octaves > 1 else axes[scale]
            ax.imshow(dog_pyramid[octave][scale], cmap='gray')
            ax.set_title(f'O{octave}-D{scale}')
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_dog.png", dpi=150, bbox_inches='tight')
    plt.show()


def compare_with_opencv(image_path):
    """
    比较自实现的SIFT和OpenCV的SIFT
    
    Args:
        image_path: 图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    print("=" * 60)
    print("比较自实现SIFT vs OpenCV SIFT")
    print("=" * 60)
    
    # 自实现的SIFT
    print("\n【自实现SIFT】")
    my_sift = MySIFT(num_octaves=4, num_scales=5)
    kp_my, desc_my, gauss_pyr, dog_pyr = my_sift.detect_and_compute(image)
    
    # OpenCV的SIFT
    print("\n【OpenCV SIFT】")
    opencv_sift = cv2.SIFT_create()
    kp_cv, desc_cv = opencv_sift.detectAndCompute(image, None)
    print(f"检测到 {len(kp_cv)} 个特征点")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果：")
    print(f"自实现SIFT: {len(kp_my)} 个特征点, 描述符维度: {desc_my.shape}")
    print(f"OpenCV SIFT: {len(kp_cv)} 个特征点, 描述符维度: {desc_cv.shape}")
    print("=" * 60)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 自实现SIFT
    img_my = cv2.drawKeypoints(image, kp_my, None, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[0].imshow(cv2.cvtColor(img_my, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'自实现SIFT ({len(kp_my)}个特征点)')
    axes[0].axis('off')
    
    # OpenCV SIFT
    img_cv = cv2.drawKeypoints(image, kp_cv, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[1].imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'OpenCV SIFT ({len(kp_cv)}个特征点)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sift_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return kp_my, desc_my, kp_cv, desc_cv, gauss_pyr, dog_pyr


if __name__ == "__main__":
    # 测试代码
    test_image = "/Users/zqli/Desktop/大三上/计算机视觉/code/hw3/SIFT/image1.jpg"
    
    if not Path(test_image).exists():
        print(f"找不到测试图像: {test_image}")
        print("请提供有效的图像路径")
    else:
        kp_my, desc_my, kp_cv, desc_cv, gauss_pyr, dog_pyr = compare_with_opencv(test_image)
        
        # 可视化金字塔
        print("\n正在可视化金字塔...")
        visualize_pyramids(gauss_pyr, dog_pyr, save_path='pyramid')
