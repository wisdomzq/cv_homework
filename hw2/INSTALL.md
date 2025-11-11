# 安装和运行指南

## 第一步：安装依赖

打开 PowerShell 或命令提示符，导航到项目目录：

```powershell
cd "e:\Study\大三上\计算机视觉\hw2"
```

然后安装所需的 Python 包：

```powershell
pip install -r requirements.txt
```

或者单独安装：

```powershell
pip install opencv-python numpy matplotlib
```

## 第二步：运行示例

### 方式1：快速开始（推荐新手）

```powershell
python quick_start.py
```

这将运行一个简单的演示，展示基本的图像融合效果。

### 方式2：完整演示

```powershell
python laplacian_pyramid_fusion.py
```

这将展示：
- 拉普拉斯金字塔的构建
- 垂直和圆形融合效果
- 重建精度测试

### 方式3：高级示例

```powershell
python advanced_examples.py
```

这将展示：
- 多种自定义掩码（文字、对角线、棋盘格、波浪等）
- 融合效果画廊
- 不同金字塔层数的比较

### 方式4：使用真实图像

如果你有自己的图像文件：

```powershell
# 方式 A: 将两张图片放在 hw2 目录下，自动检测
python real_image_fusion.py

# 方式 B: 指定图片路径
python real_image_fusion.py image1.jpg image2.jpg

# 方式 C: 指定图片和目标尺寸
python real_image_fusion.py image1.jpg image2.jpg "(800, 600)"
```

### 方式5：运行测试

验证实现的正确性：

```powershell
python test_laplacian_pyramid.py
```

## 第三步：查看结果

运行后，会在当前目录生成多个图像文件：

- `laplacian_pyramid_fusion_demo.png` - 基本融合演示
- `reconstruction_test.png` - 重建测试
- `quick_start_result.png` - 快速开始结果
- `fusion_gallery.png` - 融合效果画廊
- `mask_showcase.png` - 各种掩码展示
- 以及其他 `.jpg` 格式的融合结果

## 常见问题

### 问题1：import cv2 报错

**解决方案**：
```powershell
pip install opencv-python
```

### 问题2：import matplotlib 报错

**解决方案**：
```powershell
pip install matplotlib
```

### 问题3：图像显示窗口不出现

**解决方案**：检查 matplotlib 后端设置，或直接查看保存的图像文件。

### 问题4：内存不足

**解决方案**：减少图像尺寸或金字塔层数。

## 代码示例

### 最简单的使用方式

```python
from laplacian_pyramid_fusion import LaplacianPyramid, create_vertical_split_mask
import cv2

# 读取图像
img1 = cv2.imread('photo1.jpg')
img2 = cv2.imread('photo2.jpg')

# 调整到相同尺寸
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 创建金字塔和掩码
lp = LaplacianPyramid(levels=6)
mask = create_vertical_split_mask((img1.shape[0], img1.shape[1]))

# 融合
result = lp.blend_images(img1, img2, mask)

# 保存
cv2.imwrite('result.jpg', result)
```

### 自定义掩码

```python
import numpy as np

# 创建自定义掩码（例如：左上角为1，右下角为0）
h, w = img1.shape[:2]
mask = np.zeros((h, w), dtype=np.float32)
for i in range(h):
    for j in range(w):
        mask[i, j] = 1 - (i + j) / (h + w)

# 使用自定义掩码融合
result = lp.blend_images(img1, img2, mask)
```

## 推荐运行顺序

1. `python quick_start.py` - 先运行快速示例了解基本功能
2. `python test_laplacian_pyramid.py` - 验证实现正确性
3. `python laplacian_pyramid_fusion.py` - 查看完整演示
4. `python advanced_examples.py` - 探索高级功能
5. `python real_image_fusion.py` - 使用自己的图像

## 性能参考

在普通电脑上（Intel i5, 8GB RAM）：

- 512x512 图像，6层金字塔
  - 构建拉普拉斯金字塔：~10-30 ms
  - 图像融合：~50-150 ms
  - 内存占用：~50-100 MB

## 支持

如有问题，请检查：

1. Python 版本 >= 3.7
2. 所有依赖包已正确安装
3. 图像文件路径正确
4. 图像格式支持（.jpg, .png, .bmp 等）
