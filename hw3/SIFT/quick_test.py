"""
快速SIFT匹配测试脚本
针对速度优化的简化版本
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sift_algorithm import MySIFT
import time

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def quick_match(img1_path, img2_path, fast_mode=True):
    """
    快速匹配测试（最小可视化）
    
    Args:
        img1_path: 图像1路径
        img2_path: 图像2路径
        fast_mode: 快速模式
    """
    print("=" * 60)
    print("SIFT快速匹配测试")
    print("=" * 60)
    
    # 读取图像
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像文件")
    
    print(f"\n图像1尺寸: {img1.shape}")
    print(f"图像2尺寸: {img2.shape}")
    
    # 创建SIFT检测器
    my_sift = MySIFT(
        num_octaves=3,  # 减少到3组加速
        num_scales=4,   # 减少到4个尺度
        sigma=1.6,
        contrast_threshold=0.06 if fast_mode else 0.04,  # 提高阈值减少特征点
        edge_threshold=10,
        fast_mode=fast_mode
    )
    
    # 检测图像1
    print("\n" + "-" * 60)
    print("检测图像1...")
    start = time.time()
    kp1, desc1, _, _ = my_sift.detect_and_compute(img1)
    time1 = time.time() - start
    print(f"完成! 特征点数: {len(kp1)}, 耗时: {time1:.2f}秒")
    
    # 检测图像2
    print("\n" + "-" * 60)
    print("检测图像2...")
    start = time.time()
    kp2, desc2, _, _ = my_sift.detect_and_compute(img2)
    time2 = time.time() - start
    print(f"完成! 特征点数: {len(kp2)}, 耗时: {time2:.2f}秒")
    
    # 匹配
    print("\n" + "-" * 60)
    print("特征匹配...")
    start = time.time()
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    time_match = time.time() - start
    print(f"完成! 匹配数: {len(good_matches)}, 耗时: {time_match:.2f}秒")
    
    # RANSAC
    print("\n" + "-" * 60)
    print("RANSAC单应性矩阵估计...")
    if len(good_matches) >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(np.sum(mask))
        print(f"内点数: {inliers}/{len(good_matches)}, 内点率: {100*inliers/len(good_matches):.1f}%")
    else:
        print("匹配点不足，跳过RANSAC")
        H, mask = None, None
        inliers = 0
    
    # 总时间
    total_time = time1 + time2 + time_match
    print("\n" + "=" * 60)
    print("总耗时统计:")
    print(f"  图像1检测: {time1:.2f}秒")
    print(f"  图像2检测: {time2:.2f}秒")
    print(f"  特征匹配: {time_match:.2f}秒")
    print(f"  总计: {total_time:.2f}秒")
    print("=" * 60)
    
    # 简单可视化
    print("\n生成可视化结果...")
    
    # 只显示匹配结果
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT匹配结果 (显示前50个，总计{len(good_matches)}个) - 耗时{total_time:.1f}秒")
    plt.axis('off')
    plt.tight_layout()
    
    output_dir = Path("output_quick")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "quick_match.png", dpi=120, bbox_inches='tight')
    print(f"结果已保存到: {output_dir / 'quick_match.png'}")
    plt.show()
    
    return {
        'keypoints1': kp1,
        'keypoints2': kp2,
        'matches': good_matches,
        'time': total_time,
        'inliers': inliers
    }


def compare_modes(img1_path, img2_path):
    """
    比较不同模式的速度
    """
    print("\n" + "=" * 60)
    print("模式对比测试")
    print("=" * 60)
    
    modes = [
        ("标准模式", False),
        ("快速模式", True),
    ]
    
    results = []
    
    for mode_name, fast_mode in modes:
        print(f"\n>>> 测试 {mode_name} <<<")
        result = quick_match(img1_path, img2_path, fast_mode=fast_mode)
        results.append((mode_name, result))
        print()
    
    # 对比总结
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'模式':<15} {'特征点1':<10} {'特征点2':<10} {'匹配数':<10} {'内点数':<10} {'总耗时':<10}")
    print("-" * 60)
    
    for mode_name, result in results:
        print(f"{mode_name:<15} {len(result['keypoints1']):<10} {len(result['keypoints2']):<10} "
              f"{len(result['matches']):<10} {result['inliers']:<10} {result['time']:.2f}秒")
    
    print("=" * 60)


def main():
    """主函数"""
    img1_path = "/Users/zqli/Desktop/大三上/计算机视觉/code/hw3/SIFT/image1.jpg"
    img2_path = "/Users/zqli/Desktop/大三上/计算机视觉/code/hw3/SIFT/image2.jpg"
    
    if not Path(img1_path).exists() or not Path(img2_path).exists():
        print("错误：找不到图像文件")
        print("请将图像命名为 image1.jpg 和 image2.jpg")
        return
    
    print("\n请选择测试模式：")
    print("1. 快速匹配（推荐）")
    print("2. 模式对比（标准vs快速）")
    
    choice = input("\n请输入选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "1":
        quick_match(img1_path, img2_path, fast_mode=True)
    else:
        compare_modes(img1_path, img2_path)


if __name__ == "__main__":
    main()
