"""
测试拉普拉斯金字塔实现的正确性
"""

import cv2
import numpy as np
from laplacian_pyramid_fusion import LaplacianPyramid
import time
from laplacian_pyramid_fusion import create_vertical_split_mask


def test_gaussian_pyramid():
    """测试高斯金字塔构建"""
    print("测试 1: 高斯金字塔构建")
    print("-" * 50)
    
    # 创建测试图像
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    lp = LaplacianPyramid(levels=5)
    pyramid = lp.build_gaussian_pyramid(img)
    
    # 验证层数
    assert len(pyramid) == 5, f"期望5层，得到{len(pyramid)}层"
    print(f"✓ 金字塔层数正确: {len(pyramid)}")
    
    # 验证每层的尺寸递减
    for i in range(len(pyramid) - 1):
        h1, w1 = pyramid[i].shape[:2]
        h2, w2 = pyramid[i+1].shape[:2]
        expected_h = (h1 + 1) // 2
        expected_w = (w1 + 1) // 2
        
        # pyrDown可能产生稍微不同的尺寸
        assert abs(h2 - expected_h) <= 1, f"第{i+1}层高度不正确"
        assert abs(w2 - expected_w) <= 1, f"第{i+1}层宽度不正确"
    
    print(f"✓ 各层尺寸递减正确")
    
    # 打印各层尺寸
    for i, layer in enumerate(pyramid):
        print(f"  层 {i}: {layer.shape}")
    
    print("✓ 高斯金字塔测试通过\n")
    return True


def test_laplacian_pyramid():
    """测试拉普拉斯金字塔构建"""
    print("测试 2: 拉普拉斯金字塔构建")
    print("-" * 50)
    
    # 创建测试图像
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    lp = LaplacianPyramid(levels=4)
    lap_pyramid = lp.build_laplacian_pyramid(img)
    
    # 验证层数
    assert len(lap_pyramid) == 4, f"期望4层，得到{len(lap_pyramid)}层"
    print(f"✓ 拉普拉斯金字塔层数正确: {len(lap_pyramid)}")
    
    # 打印各层尺寸
    for i, layer in enumerate(lap_pyramid):
        print(f"  层 {i}: {layer.shape}, dtype: {layer.dtype}")
    
    print("✓ 拉普拉斯金字塔测试通过\n")
    return True


def test_reconstruction():
    """测试图像重建"""
    print("测试 3: 图像重建")
    print("-" * 50)
    
    # 创建测试图像
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    lp = LaplacianPyramid(levels=5)
    
    # 构建拉普拉斯金字塔
    lap_pyramid = lp.build_laplacian_pyramid(img)
    
    # 重建图像
    reconstructed = lp.reconstruct_from_laplacian(lap_pyramid)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # 计算重建误差
    error = np.abs(img.astype(np.float32) - reconstructed.astype(np.float32))
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"重建误差统计:")
    print(f"  最大误差: {max_error:.4f}")
    print(f"  平均误差: {mean_error:.4f}")
    print(f"  标准差: {np.std(error):.4f}")
    
    # 验证误差在合理范围内（由于浮点运算和pyrUp/pyrDown的舍入）
    assert max_error < 10, f"最大误差过大: {max_error}"
    assert mean_error < 2, f"平均误差过大: {mean_error}"
    
    print("✓ 图像重建测试通过\n")
    return True


def test_image_fusion():
    """测试图像融合"""
    print("测试 4: 图像融合")
    print("-" * 50)
    
    # 创建两个测试图像
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img1[:, :] = [255, 0, 0]  # 红色
    
    img2 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2[:, :] = [0, 0, 255]  # 蓝色
    
    # 创建简单的垂直分割掩码
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[:, :128] = 1.0  # 左半边为1，右半边为0
    
    lp = LaplacianPyramid(levels=4)
    
    # 执行融合
    result = lp.blend_images(img1, img2, mask)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 验证结果
    # 左侧应该接近红色，右侧应该接近蓝色
    left_color = result[128, 64]  # 左侧中心点
    right_color = result[128, 192]  # 右侧中心点
    
    print(f"左侧颜色 (应接近红色): {left_color}")
    print(f"右侧颜色 (应接近蓝色): {right_color}")
    
    # 验证左侧偏红
    assert left_color[2] > 200, "左侧应该是红色"
    # 验证右侧偏蓝
    assert right_color[0] > 200, "右侧应该是蓝色"
    
    print("✓ 图像融合测试通过\n")
    return True


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    return g.astype(np.float32) / 255.0


def _grad_mag(img: np.ndarray) -> np.ndarray:
    g = _to_gray_float(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(g * 0 + gx * gx + gy * gy)


def _seam_band(mask: np.ndarray, band_width: int = 12) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    m = (mask > 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (band_width, band_width))
    dil = cv2.dilate(m, k)
    ero = cv2.erode(m, k)
    band = cv2.subtract(dil, ero)
    return (band > 0).astype(np.float32)


def test_seam_energy_reduction():
    """硬边掩码下，接缝带梯度能量：拉普拉斯融合优于直接硬拼接。"""
    print("测试 4b: 接缝带能量对比（Laplacian vs Direct）")
    h, w = 256, 256

    # 构造高反差图像，显式制造接缝挑战
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    img1[:, :] = [0, 0, 255]  # 红色通道在BGR的最后一位，保持示例一致
    cv2.putText(img1, 'A', (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    img2[:, :] = [255, 255, 0]  # 青色
    cv2.putText(img2, 'B', (140, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    mask = create_vertical_split_mask((h, w), split_position=0.5).astype(np.float32)

    # 融合
    lp = LaplacianPyramid(levels=5)
    fused = lp.blend_images(img1, img2, mask)
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    # 直接硬拼接（等价于直拼）
    direct = (img1 * mask[:, :, None] + img2 * (1 - mask[:, :, None])).astype(np.uint8)

    # 接缝带
    band = _seam_band(mask, band_width=14)

    # 梯度能量
    g_f = _grad_mag(fused)
    g_d = _grad_mag(direct)
    seam_energy_fused = float(np.sum(g_f * band) / (np.sum(band) + 1e-8))
    seam_energy_direct = float(np.sum(g_d * band) / (np.sum(band) + 1e-8))

    print(f"  Seam energy (fused):  {seam_energy_fused:.4f}")
    print(f"  Seam energy (direct): {seam_energy_direct:.4f}")

    # 断言：融合的接缝能量不高于直接硬拼接（通常更小）
    assert seam_energy_fused <= seam_energy_direct * 1.05, \
        "接缝带梯度能量未优于直接拼接，请检查层数或实现"
    print("✓ 接缝能量对比测试通过\n")
    return True


def test_edge_cases():
    """测试边界情况"""
    print("测试 5: 边界情况")
    print("-" * 50)
    
    lp = LaplacianPyramid(levels=3)
    
    # 测试小图像
    small_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    try:
        pyramid = lp.build_gaussian_pyramid(small_img)
        print(f"✓ 小图像测试通过 (32x32)")
    except Exception as e:
        print(f"✗ 小图像测试失败: {e}")
        return False
    
    # 测试单通道图像
    gray_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    try:
        pyramid = lp.build_gaussian_pyramid(gray_img)
        print(f"✓ 灰度图像测试通过")
    except Exception as e:
        print(f"✗ 灰度图像测试失败: {e}")
        return False
    
    # 测试非方形图像
    rect_img = np.random.randint(0, 256, (256, 512, 3), dtype=np.uint8)
    try:
        pyramid = lp.build_gaussian_pyramid(rect_img)
        print(f"✓ 长方形图像测试通过 (256x512)")
    except Exception as e:
        print(f"✗ 长方形图像测试失败: {e}")
        return False
    
    print("✓ 边界情况测试通过\n")
    return True


def test_performance():
    """测试性能"""
    print("测试 6: 性能测试")
    print("-" * 50)
    
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    lp = LaplacianPyramid(levels=6)
    
    # 测试高斯金字塔构建速度
    start = time.time()
    for _ in range(10):
        pyramid = lp.build_gaussian_pyramid(img)
    gauss_time = (time.time() - start) / 10
    print(f"高斯金字塔构建平均时间: {gauss_time*1000:.2f} ms")
    
    # 测试拉普拉斯金字塔构建速度
    start = time.time()
    for _ in range(10):
        lap_pyramid = lp.build_laplacian_pyramid(img)
    lap_time = (time.time() - start) / 10
    print(f"拉普拉斯金字塔构建平均时间: {lap_time*1000:.2f} ms")
    
    # 测试重建速度
    lap_pyramid = lp.build_laplacian_pyramid(img)
    start = time.time()
    for _ in range(10):
        reconstructed = lp.reconstruct_from_laplacian(lap_pyramid)
    recon_time = (time.time() - start) / 10
    print(f"图像重建平均时间: {recon_time*1000:.2f} ms")
    
    # 测试融合速度
    img2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.random.rand(512, 512).astype(np.float32)
    start = time.time()
    for _ in range(10):
        result = lp.blend_images(img, img2, mask)
    fusion_time = (time.time() - start) / 10
    print(f"图像融合平均时间: {fusion_time*1000:.2f} ms")
    
    print("✓ 性能测试完成\n")
    return True


def test_different_levels():
    """测试不同的金字塔层数"""
    print("测试 7: 不同金字塔层数")
    print("-" * 50)
    
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    for levels in [2, 3, 4, 5, 6, 7, 8]:
        try:
            lp = LaplacianPyramid(levels=levels)
            lap_pyramid = lp.build_laplacian_pyramid(img)
            reconstructed = lp.reconstruct_from_laplacian(lap_pyramid)
            
            error = np.mean(np.abs(img.astype(np.float32) - 
                                  np.clip(reconstructed, 0, 255).astype(np.float32)))
            
            print(f"  {levels} 层: 平均误差 = {error:.4f}")
            
        except Exception as e:
            print(f"  {levels} 层: 失败 - {e}")
    
    print("✓ 不同层数测试完成\n")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("拉普拉斯金字塔实现正确性测试")
    print("=" * 50)
    print()
    
    tests = [
        ("高斯金字塔构建", test_gaussian_pyramid),
        ("拉普拉斯金字塔构建", test_laplacian_pyramid),
        ("图像重建", test_reconstruction),
        ("图像融合", test_image_fusion),
        ("接缝能量对比", test_seam_energy_reduction),
        ("边界情况", test_edge_cases),
        ("性能测试", test_performance),
        ("不同金字塔层数", test_different_levels),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} 失败\n")
        except Exception as e:
            failed += 1
            print(f"✗ {name} 异常: {e}\n")
    
    print("=" * 50)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✓ 所有测试通过！实现正确。")
    else:
        print("\n✗ 部分测试失败，请检查实现。")
