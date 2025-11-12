"""
SIFT图像特征匹配实现
使用OpenCV的SIFT算法进行两幅图像的特征点检测和匹配
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


class SIFTMatcher:
    """SIFT特征匹配器类"""
    
    def __init__(self, ratio_threshold=0.75):
        """
        初始化SIFT匹配器
        
        Args:
            ratio_threshold: Lowe's ratio test的阈值，默认0.75
        """
        self.ratio_threshold = ratio_threshold
        self.sift = cv2.SIFT_create()
        
    def detect_and_compute(self, image):
        """
        检测图像中的SIFT特征点并计算描述符
        
        Args:
            image: 输入图像（灰度或彩色）
            
        Returns:
            keypoints: 特征点列表
            descriptors: 特征描述符
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 检测特征点并计算描述符
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, use_flann=True):
        """
        使用BFMatcher或FLANN匹配特征描述符
        
        Args:
            desc1: 第一幅图像的描述符
            desc2: 第二幅图像的描述符
            use_flann: 是否使用FLANN匹配器（快速），默认True
            
        Returns:
            good_matches: 通过ratio test的优质匹配点对
        """
        if use_flann:
            # FLANN匹配器参数
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher（暴力匹配）
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # KNN匹配，k=2表示为每个特征点找到两个最佳匹配
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # 应用Lowe's ratio test筛选优质匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def draw_matches(self, img1, kp1, img2, kp2, matches, max_matches=100):
        """
        绘制匹配结果
        
        Args:
            img1: 第一幅图像
            kp1: 第一幅图像的特征点
            img2: 第二幅图像
            kp2: 第二幅图像的特征点
            matches: 匹配点对
            max_matches: 最多显示的匹配数量
            
        Returns:
            result_img: 绘制匹配结果的图像
        """
        # 只显示前max_matches个匹配
        matches_to_draw = matches[:max_matches]
        
        # 绘制匹配
        result_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches_to_draw, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result_img
    
    def find_homography(self, kp1, kp2, matches, min_match_count=10):
        """
        使用RANSAC算法计算单应性矩阵
        
        Args:
            kp1: 第一幅图像的特征点
            kp2: 第二幅图像的特征点
            matches: 匹配点对
            min_match_count: 最少匹配点数量
            
        Returns:
            H: 单应性矩阵
            mask: 内点标记
        """
        if len(matches) < min_match_count:
            print(f"匹配点数量不足：{len(matches)} < {min_match_count}")
            return None, None
        
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC算法计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H, mask
    
    def draw_homography_matches(self, img1, kp1, img2, kp2, matches, mask):
        """
        绘制经过单应性矩阵筛选后的匹配结果
        
        Args:
            img1: 第一幅图像
            kp1: 第一幅图像的特征点
            img2: 第二幅图像
            kp2: 第二幅图像的特征点
            matches: 匹配点对
            mask: 内点标记
            
        Returns:
            result_img: 绘制结果的图像
        """
        # 筛选内点
        inlier_matches = [m for m, msk in zip(matches, mask) if msk]
        
        # 绘制内点匹配
        result_img = cv2.drawMatches(
            img1, kp1, img2, kp2, inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result_img


def visualize_keypoints(img, keypoints, title="Keypoints"):
    """
    可视化图像的特征点
    
    Args:
        img: 输入图像
        keypoints: 特征点列表
        title: 图像标题
    """
    img_with_kp = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} (总计: {len(keypoints)}个特征点)")
    plt.axis('off')
    plt.tight_layout()
    return img_with_kp


def match_images(img1_path, img2_path, ratio_threshold=0.75, use_flann=True, 
                 save_results=True, output_dir="output"):
    """
    对两幅图像进行SIFT特征匹配的完整流程
    
    Args:
        img1_path: 第一幅图像路径
        img2_path: 第二幅图像路径
        ratio_threshold: Lowe's ratio test阈值
        use_flann: 是否使用FLANN匹配器
        save_results: 是否保存结果
        output_dir: 输出目录
        
    Returns:
        results: 包含匹配信息的字典
    """
    # 读取图像
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像文件")
    
    print(f"图像1尺寸: {img1.shape}")
    print(f"图像2尺寸: {img2.shape}")
    
    # 创建SIFT匹配器
    matcher = SIFTMatcher(ratio_threshold=ratio_threshold)
    
    # 检测特征点和计算描述符
    print("\n正在检测特征点...")
    kp1, desc1 = matcher.detect_and_compute(img1)
    kp2, desc2 = matcher.detect_and_compute(img2)
    
    print(f"图像1检测到 {len(kp1)} 个特征点")
    print(f"图像2检测到 {len(kp2)} 个特征点")
    
    # 匹配特征
    print("\n正在匹配特征...")
    matches = matcher.match_features(desc1, desc2, use_flann=use_flann)
    print(f"找到 {len(matches)} 个优质匹配")
    
    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 计算单应性矩阵
    print("\n正在计算单应性矩阵...")
    H, mask = matcher.find_homography(kp1, kp2, matches)
    
    if H is not None:
        inliers = np.sum(mask)
        print(f"RANSAC找到 {inliers} 个内点")
    
    # 创建输出目录
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # 可视化特征点
    print("\n正在生成可视化结果...")
    fig = plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    img1_kp = cv2.drawKeypoints(img1, kp1, None, 
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"图像1特征点 ({len(kp1)}个)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    img2_kp = cv2.drawKeypoints(img2, kp2, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"图像2特征点 ({len(kp2)}个)")
    plt.axis('off')
    
    plt.tight_layout()
    if save_results:
        plt.savefig(output_path / "keypoints.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 绘制匹配结果
    match_img = matcher.draw_matches(img1, kp1, img2, kp2, matches, max_matches=100)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT特征匹配结果 (显示前100个匹配，总计{len(matches)}个)")
    plt.axis('off')
    plt.tight_layout()
    if save_results:
        plt.savefig(output_path / "matches.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 如果找到单应性矩阵，绘制RANSAC筛选后的匹配
    if H is not None:
        ransac_img = matcher.draw_homography_matches(img1, kp1, img2, kp2, matches, mask)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB))
        plt.title(f"RANSAC筛选后的匹配 ({inliers}个内点)")
        plt.axis('off')
        plt.tight_layout()
        if save_results:
            plt.savefig(output_path / "ransac_matches.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    # 返回结果
    results = {
        'keypoints1': kp1,
        'keypoints2': kp2,
        'descriptors1': desc1,
        'descriptors2': desc2,
        'matches': matches,
        'homography': H,
        'inlier_mask': mask
    }
    
    print("\n匹配完成！")
    if save_results:
        print(f"结果已保存到 {output_path}")
    
    return results


