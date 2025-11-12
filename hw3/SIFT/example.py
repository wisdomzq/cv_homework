"""
SIFT特征匹配示例 - 使用生成的测试图像
"""

import cv2
import numpy as np
from pathlib import Path
from sift_matching import match_images


def create_test_images():
    """
    创建测试图像对
    包含一些简单的几何形状和纹理
    """
    # 创建第一幅图像
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 添加一些形状
    cv2.rectangle(img1, (50, 50), (200, 150), (0, 0, 255), -1)
    cv2.circle(img1, (400, 100), 50, (0, 255, 0), -1)
    cv2.rectangle(img1, (100, 200), (300, 350), (255, 0, 0), 3)
    
    # 添加一些文字（作为纹理特征）
    cv2.putText(img1, "SIFT TEST", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 添加一些随机噪声增加纹理
    noise = np.random.randint(0, 50, img1.shape, dtype=np.uint8)
    img1 = cv2.add(img1, noise)
    
    # 创建第二幅图像（第一幅的变换版本）
    # 应用旋转和缩放
    rows, cols = img1.shape[:2]
    
    # 旋转15度
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 0.9)
    img2 = cv2.warpAffine(img1, M_rot, (cols, rows), 
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=(255, 255, 255))
    
    # 添加一些不同的噪声
    noise2 = np.random.randint(0, 50, img2.shape, dtype=np.uint8)
    img2 = cv2.add(img2, noise2)
    
    return img1, img2


def example_with_test_images():
    """使用生成的测试图像进行演示"""
    print("=== 使用生成的测试图像 ===\n")
    
    # 创建测试图像
    print("正在生成测试图像...")
    img1, img2 = create_test_images()
    
    # 保存测试图像
    cv2.imwrite("test_image1.jpg", img1)
    cv2.imwrite("test_image2.jpg", img2)
    print("测试图像已保存: test_image1.jpg, test_image2.jpg\n")
    
    # 执行匹配
    results = match_images(
        img1_path="test_image1.jpg",
        img2_path="test_image2.jpg",
        ratio_threshold=0.75,
        use_flann=True,
        save_results=True,
        output_dir="output_test"
    )
    
    return results


def example_with_real_images(img1_path, img2_path):
    """使用真实图像进行演示"""
    print("=== 使用真实图像 ===\n")
    
    # 检查文件是否存在
    if not Path(img1_path).exists():
        print(f"错误：找不到图像文件 {img1_path}")
        return None
    if not Path(img2_path).exists():
        print(f"错误：找不到图像文件 {img2_path}")
        return None
    
    # 执行匹配
    results = match_images(
        img1_path=img1_path,
        img2_path=img2_path,
        ratio_threshold=0.75,
        use_flann=True,
        save_results=True,
        output_dir="output_real"
    )
    
    return results


def compare_matching_methods():
    """比较不同匹配方法和参数的效果"""
    print("=== 比较不同匹配方法 ===\n")
    
    # 创建测试图像
    img1, img2 = create_test_images()
    cv2.imwrite("test_img1.jpg", img1)
    cv2.imwrite("test_img2.jpg", img2)
    
    # 测试不同的ratio_threshold
    thresholds = [0.6, 0.7, 0.75, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\n--- Ratio Threshold = {threshold} ---")
        results = match_images(
            img1_path="test_img1.jpg",
            img2_path="test_img2.jpg",
            ratio_threshold=threshold,
            use_flann=True,
            save_results=True,
            output_dir=f"output_threshold_{threshold}"
        )
        
        if results['homography'] is not None:
            inliers = np.sum(results['inlier_mask'])
            print(f"匹配数: {len(results['matches'])}, 内点数: {inliers}")


def main():
    """主函数"""
    print("SIFT特征匹配示例程序\n")
    print("请选择运行模式：")
    print("1. 使用生成的测试图像（推荐用于快速测试）")
    print("2. 使用自己的图像")
    print("3. 比较不同参数效果")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        example_with_test_images()
    elif choice == "2":
        img1_path = input("请输入第一张图像路径: ").strip()
        img2_path = input("请输入第二张图像路径: ").strip()
        example_with_real_images(img1_path, img2_path)
    elif choice == "3":
        compare_matching_methods()
    else:
        print("无效的选择，使用默认模式（生成测试图像）")
        example_with_test_images()


if __name__ == "__main__":
    main()
