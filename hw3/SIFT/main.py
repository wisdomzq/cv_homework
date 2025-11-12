"""
完整的SIFT图像匹配程序
整合自实现的SIFT算法和匹配功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sift_algorithm import MySIFT, visualize_pyramids

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SIFTMatcher:
    """SIFT特征匹配器（使用自实现的SIFT）"""
    
    def __init__(self, ratio_threshold=0.75, use_my_sift=True, fast_mode=False):
        """
        初始化匹配器
        
        Args:
            ratio_threshold: Lowe's ratio test的阈值
            use_my_sift: 是否使用自实现的SIFT（True）或OpenCV的SIFT（False）
            fast_mode: 快速模式（减少特征点以加速计算）
        """
        self.ratio_threshold = ratio_threshold
        self.use_my_sift = use_my_sift
        
        if use_my_sift:
            self.sift = MySIFT(num_octaves=4, num_scales=5, sigma=1.6, fast_mode=fast_mode)
            print(f"使用自实现的SIFT算法 {'[快速模式]' if fast_mode else ''}")
        else:
            self.sift = cv2.SIFT_create()
            print("使用OpenCV的SIFT算法")
    
    def detect_and_compute(self, image):
        """检测特征点并计算描述符"""
        if self.use_my_sift:
            kp, desc, gauss_pyr, dog_pyr = self.sift.detect_and_compute(image)
            return kp, desc, gauss_pyr, dog_pyr
        else:
            kp, desc = self.sift.detectAndCompute(image, None)
            return kp, desc, None, None
    
    def match_features(self, desc1, desc2, use_flann=True):
        """
        匹配特征描述符
        
        Args:
            desc1: 第一幅图像的描述符
            desc2: 第二幅图像的描述符
            use_flann: 是否使用FLANN匹配器
            
        Returns:
            good_matches: 优质匹配点对
        """
        if use_flann:
            # FLANN匹配器
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # KNN匹配
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography(self, kp1, kp2, matches, min_match_count=10):
        """使用RANSAC计算单应性矩阵"""
        if len(matches) < min_match_count:
            print(f"匹配点数量不足：{len(matches)} < {min_match_count}")
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H, mask


def analyze_keypoint_distribution(keypoints, image_shape, title="关键点分布分析"):
    """
    分析关键点的分布特性
    
    Args:
        keypoints: 关键点列表
        image_shape: 图像形状
        title: 标题
    """
    if len(keypoints) == 0:
        print("没有关键点可分析")
        return
    
    # 提取关键点信息
    positions = np.array([kp.pt for kp in keypoints])
    sizes = np.array([kp.size for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    angles = np.array([kp.angle for kp in keypoints])
    
    # 创建分析图
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 空间分布热力图
    ax1 = plt.subplot(2, 3, 1)
    h, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=20)
    im1 = ax1.imshow(h.T, origin='lower', cmap='hot', interpolation='nearest',
                     extent=[0, image_shape[1], 0, image_shape[0]], aspect='auto')
    ax1.set_title('关键点空间分布热力图')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 尺度分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(sizes, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_title(f'尺度分布 (均值: {np.mean(sizes):.2f})')
    ax2.set_xlabel('特征点尺度')
    ax2.set_ylabel('数量')
    ax2.grid(True, alpha=0.3)
    
    # 3. 响应强度分布
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(responses, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.set_title(f'响应强度分布 (均值: {np.mean(responses):.4f})')
    ax3.set_xlabel('响应强度')
    ax3.set_ylabel('数量')
    ax3.grid(True, alpha=0.3)
    
    # 4. 方向分布（极坐标）
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    angles_rad = np.deg2rad(angles)
    ax4.hist(angles_rad, bins=36, color='red', alpha=0.7, edgecolor='black')
    ax4.set_title('主方向分布')
    
    # 5. 尺度-响应散点图
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(sizes, responses, c=responses, cmap='viridis', 
                         alpha=0.6, s=20, edgecolor='black', linewidth=0.5)
    ax5.set_title('尺度 vs 响应强度')
    ax5.set_xlabel('尺度')
    ax5.set_ylabel('响应强度')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5)
    
    # 6. 统计信息文本
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    统计信息：
    
    特征点总数: {len(keypoints)}
    
    位置:
      X范围: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]
      Y范围: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]
    
    尺度:
      最小: {sizes.min():.2f}
      最大: {sizes.max():.2f}
      均值: {sizes.mean():.2f}
      标准差: {sizes.std():.2f}
    
    响应强度:
      最小: {responses.min():.4f}
      最大: {responses.max():.4f}
      均值: {responses.mean():.4f}
      标准差: {responses.std():.4f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def analyze_matching_quality(matches, kp1, kp2):
    """
    分析匹配质量
    
    Args:
        matches: 匹配点对
        kp1: 图像1的关键点
        kp2: 图像2的关键点
    """
    if len(matches) == 0:
        print("没有匹配可分析")
        return
    
    # 提取匹配信息
    distances = np.array([m.distance for m in matches])
    
    # 提取匹配点对的位置
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
    
    # 计算位移
    displacements = pts2 - pts1
    displacement_mags = np.linalg.norm(displacements, axis=1)
    displacement_angles = np.arctan2(displacements[:, 1], displacements[:, 0])
    
    # 创建分析图
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 匹配距离分布
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(distances, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(distances.mean(), color='red', linestyle='--', 
                label=f'均值: {distances.mean():.2f}')
    ax1.axvline(np.median(distances), color='green', linestyle='--',
                label=f'中位数: {np.median(distances):.2f}')
    ax1.set_title('匹配距离分布')
    ax1.set_xlabel('描述符距离')
    ax1.set_ylabel('数量')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 位移大小分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(displacement_mags, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(displacement_mags.mean(), color='red', linestyle='--',
                label=f'均值: {displacement_mags.mean():.1f}')
    ax2.set_title('匹配点位移大小分布')
    ax2.set_xlabel('位移大小(像素)')
    ax2.set_ylabel('数量')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 位移方向分布
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    ax3.hist(displacement_angles, bins=36, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_title('匹配点位移方向分布')
    
    # 4. 位移向量场
    ax4 = plt.subplot(2, 3, 4)
    # 随机采样显示部分向量（避免过于密集）
    sample_indices = np.random.choice(len(pts1), min(100, len(pts1)), replace=False)
    ax4.quiver(pts1[sample_indices, 0], pts1[sample_indices, 1],
               displacements[sample_indices, 0], displacements[sample_indices, 1],
               angles='xy', scale_units='xy', scale=1, alpha=0.6)
    ax4.set_title('匹配点位移向量场(采样100个)')
    ax4.set_xlabel('X坐标')
    ax4.set_ylabel('Y坐标')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # 5. 距离排名曲线
    ax5 = plt.subplot(2, 3, 5)
    sorted_distances = np.sort(distances)
    ax5.plot(range(len(sorted_distances)), sorted_distances, 'b-', linewidth=2)
    ax5.set_title('匹配距离排名曲线')
    ax5.set_xlabel('匹配排名')
    ax5.set_ylabel('描述符距离')
    ax5.grid(True, alpha=0.3)
    
    # 6. 统计信息
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    匹配质量统计：
    
    匹配总数: {len(matches)}
    
    描述符距离:
      最小: {distances.min():.2f}
      最大: {distances.max():.2f}
      均值: {distances.mean():.2f}
      中位数: {np.median(distances):.2f}
      标准差: {distances.std():.2f}
    
    位移大小:
      最小: {displacement_mags.min():.1f} 像素
      最大: {displacement_mags.max():.1f} 像素
      均值: {displacement_mags.mean():.1f} 像素
      标准差: {displacement_mags.std():.1f} 像素
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('匹配质量分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def comprehensive_matching(img1_path, img2_path, use_my_sift=True, 
                          ratio_threshold=0.75, output_dir="output_comprehensive",
                          fast_mode=False, visualize_pyramids_flag=True):
    """
    完整的SIFT匹配分析流程
    
    Args:
        img1_path: 图像1路径
        img2_path: 图像2路径
        use_my_sift: 是否使用自实现的SIFT
        ratio_threshold: ratio test阈值
        output_dir: 输出目录
        fast_mode: 快速模式（减少特征点以加速）
        visualize_pyramids_flag: 是否可视化金字塔（耗时操作）
    """
    print("=" * 70)
    print("SIFT图像特征匹配 - 完整分析")
    print(f"模式: {'快速模式' if fast_mode else '标准模式'}")
    print("=" * 70)
    
    # 读取图像
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像文件")
    
    print(f"\n图像1尺寸: {img1.shape}")
    print(f"图像2尺寸: {img2.shape}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 创建匹配器
    matcher = SIFTMatcher(ratio_threshold=ratio_threshold, use_my_sift=use_my_sift, fast_mode=fast_mode)
    
    # 检测特征点
    print("\n" + "-" * 70)
    print("步骤1: 特征点检测")
    print("-" * 70)
    
    import time
    start_time = time.time()
    kp1, desc1, gauss_pyr1, dog_pyr1 = matcher.detect_and_compute(img1)
    time1 = time.time() - start_time
    print(f"图像1: {len(kp1)} 个特征点 (耗时: {time1:.2f}秒)")
    
    start_time = time.time()
    kp2, desc2, gauss_pyr2, dog_pyr2 = matcher.detect_and_compute(img2)
    time2 = time.time() - start_time
    print(f"图像2: {len(kp2)} 个特征点 (耗时: {time2:.2f}秒)")
    
    # 如果使用自实现的SIFT，可视化金字塔
    if use_my_sift and gauss_pyr1 is not None and visualize_pyramids_flag:
        print("\n正在可视化金字塔结构...")
        visualize_pyramids(gauss_pyr1, dog_pyr1, save_path=output_path / "pyramid_img1")
        visualize_pyramids(gauss_pyr2, dog_pyr2, save_path=output_path / "pyramid_img2")
    elif not visualize_pyramids_flag:
        print("\n跳过金字塔可视化以节省时间")
    
    # 分析关键点分布
    print("\n" + "-" * 70)
    print("步骤2: 关键点分布分析")
    print("-" * 70)
    fig1 = analyze_keypoint_distribution(kp1, img1.shape, "图像1 - 关键点分布分析")
    fig1.savefig(output_path / "keypoint_analysis_img1.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    fig2 = analyze_keypoint_distribution(kp2, img2.shape, "图像2 - 关键点分布分析")
    fig2.savefig(output_path / "keypoint_analysis_img2.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 可视化特征点
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
    plt.savefig(output_path / "keypoints.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 特征匹配
    print("\n" + "-" * 70)
    print("步骤3: 特征匹配")
    print("-" * 70)
    matches = matcher.match_features(desc1, desc2, use_flann=True)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"找到 {len(matches)} 个优质匹配")
    
    # 分析匹配质量
    print("\n" + "-" * 70)
    print("步骤4: 匹配质量分析")
    print("-" * 70)
    fig3 = analyze_matching_quality(matches, kp1, kp2)
    fig3.savefig(output_path / "matching_quality_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 绘制匹配结果
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"特征匹配结果 (显示前100个，总计{len(matches)}个)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path / "matches.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # RANSAC筛选
    print("\n" + "-" * 70)
    print("步骤5: RANSAC单应性矩阵估计")
    print("-" * 70)
    H, mask = matcher.find_homography(kp1, kp2, matches)
    
    if H is not None:
        inliers = int(np.sum(mask))
        outliers = len(matches) - inliers
        inlier_ratio = inliers / len(matches) * 100
        
        print(f"内点数: {inliers}")
        print(f"外点数: {outliers}")
        print(f"内点比例: {inlier_ratio:.2f}%")
        
        # 绘制RANSAC结果
        inlier_matches = [m for m, msk in zip(matches, mask) if msk]
        ransac_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB))
        plt.title(f"RANSAC筛选后的匹配 ({inliers}个内点, 内点率{inlier_ratio:.1f}%)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / "ransac_matches.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    # 最终统计
    print("\n" + "=" * 70)
    print("最终统计结果")
    print("=" * 70)
    print(f"图像1特征点数: {len(kp1)}")
    print(f"图像2特征点数: {len(kp2)}")
    print(f"匹配点对数: {len(matches)}")
    if H is not None:
        print(f"RANSAC内点数: {inliers}")
        print(f"内点比例: {inlier_ratio:.2f}%")
    print(f"\n结果已保存到: {output_path}")
    print("=" * 70)
    
    return {
        'keypoints1': kp1,
        'keypoints2': kp2,
        'descriptors1': desc1,
        'descriptors2': desc2,
        'matches': matches,
        'homography': H,
        'inlier_mask': mask
    }


def main():
    """主函数"""
    img1_path = "/Users/zqli/Desktop/大三上/计算机视觉/code/hw3/SIFT/image1.jpg"
    img2_path = "/Users/zqli/Desktop/大三上/计算机视觉/code/hw3/SIFT/image2.jpg"
    
    if not Path(img1_path).exists() or not Path(img2_path).exists():
        print("错误：找不到图像文件")
        print("请将图像命名为 image1.jpg 和 image2.jpg")
        return
    
    print("\n请选择运行模式：")
    print("1. 标准模式（完整分析，耗时较长）")
    print("2. 快速模式（减少特征点，加快速度）")
    print("3. 极速模式（快速+跳过金字塔可视化）")
    
    choice = input("\n请输入选择 (1/2/3，默认2): ").strip() or "2"
    
    if choice == "1":
        fast_mode = False
        visualize_pyramids_flag = True
        output_dir = "output_my_sift_standard"
    elif choice == "2":
        fast_mode = True
        visualize_pyramids_flag = True
        output_dir = "output_my_sift_fast"
    else:  # choice == "3"
        fast_mode = True
        visualize_pyramids_flag = False
        output_dir = "output_my_sift_ultra_fast"
    
    # 使用自实现的SIFT进行完整分析
    results = comprehensive_matching(
        img1_path=img1_path,
        img2_path=img2_path,
        use_my_sift=True,  # 使用自实现的SIFT
        ratio_threshold=0.75,
        output_dir=output_dir,
        fast_mode=fast_mode,
        visualize_pyramids_flag=visualize_pyramids_flag
    )


if __name__ == "__main__":
    main()
