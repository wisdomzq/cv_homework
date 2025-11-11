"""
快速开始示例 - 最简单的使用方式
"""

import cv2
import numpy as np
from laplacian_pyramid_fusion import LaplacianPyramid, create_vertical_split_mask
import matplotlib.pyplot as plt


def quick_start():
    """最简单的使用示例"""
    
    print("快速开始：基于拉普拉斯金字塔的图像融合\n")
    
    # 步骤1: 创建或读取两张图像
    print("步骤 1: 创建测试图像...")
    
    # 图像1 - 蓝色渐变
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        img1[:, i] = [255 - i*255//400, 100, i*255//400]
    cv2.circle(img1, (200, 200), 80, (255, 255, 0), -1)
    
    # 图像2 - 红色渐变
    img2 = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        img2[i, :] = [i*255//400, 255 - i*255//400, 100]
    cv2.rectangle(img2, (150, 150), (250, 250), (0, 255, 255), -1)
    
    # 步骤2: 创建拉普拉斯金字塔对象
    print("步骤 2: 创建拉普拉斯金字塔对象...")
    lp = LaplacianPyramid(levels=5)  # 5层金字塔
    
    # 步骤3: 创建融合掩码
    print("步骤 3: 创建融合掩码...")
    mask = create_vertical_split_mask(
        shape=(400, 400),
        split_position=0.5  # 0.5 表示中间分割
    )
    
    # 步骤4: 执行融合
    print("步骤 4: 执行图像融合...")
    result = lp.blend_images(img1, img2, mask)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 步骤5: 显示和保存结果
    print("步骤 5: 显示结果...")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Input Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Input Image 2')
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Fusion Mask')
    axes[2].axis('off')
    
    axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Fused Result')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_start_result.png', dpi=150, bbox_inches='tight')
    cv2.imwrite('quick_start_result.jpg', result)
    
    print("\n完成！结果已保存:")
    print("  - quick_start_result.png (可视化)")
    print("  - quick_start_result.jpg (融合结果)")
    
    plt.show()


if __name__ == "__main__":
    quick_start()
