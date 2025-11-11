# 基于拉普拉斯金字塔的图像融合

这是一个完整的拉普拉斯金字塔图像融合实现，不使用现成的拉普拉斯金字塔函数，但使用OpenCV提供的高斯处理函数（`pyrDown`和`pyrUp`）。

## 功能特点

- ✅ **自实现拉普拉斯金字塔构建**：手动实现金字塔的构建和重建过程
- ✅ **多种融合模式**：支持垂直、水平、圆形、渐变等多种掩码
- ✅ **完整的可视化**：展示融合过程和结果对比
- ✅ **支持真实图像**：可用于融合实际照片

## 文件说明

1. **laplacian_pyramid_fusion.py** - 核心实现
   - `LaplacianPyramid` 类：实现拉普拉斯金字塔的构建、融合和重建
   - 掩码创建函数：垂直、水平、圆形掩码
   - 演示函数：使用生成的图像进行演示

2. **real_image_fusion.py** - 真实图像融合示例
   - 支持加载和融合真实照片
   - 多种融合效果展示
   - 自动图像尺寸调整

## 原理说明

### 拉普拉斯金字塔构建

1. **高斯金字塔构建**：
   - 对原图进行高斯模糊和下采样，得到多层不同分辨率的图像
   - 使用 `cv2.pyrDown()` 实现

2. **拉普拉斯金字塔构建**：
   - 对高斯金字塔的每一层，计算该层与上采样的下一层的差值
   - 拉普拉斯层 = 当前层 - pyrUp(下一层)
   - 手动实现，不使用 `cv2.buildPyramid()` 等现成函数

3. **金字塔重建**：
   - 从最顶层开始，逐层上采样并加上拉普拉斯层
   - 重建图像 = pyrUp(上一层重建) + 当前拉普拉斯层

### 图像融合

1. 构建两张输入图像的拉普拉斯金字塔
2. 构建掩码的高斯金字塔
3. 在每一层进行加权融合：`融合层 = 掩码 * 图像1 + (1-掩码) * 图像2`
4. 从融合后的拉普拉斯金字塔重建最终图像

## 安装依赖

```bash
pip install opencv-python numpy matplotlib
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本演示（使用生成的图像）

```bash
python laplacian_pyramid_fusion.py
```

这将：
- 展示拉普拉斯金字塔的构建过程
- 演示垂直和圆形融合效果
- 测试金字塔重建精度
- 保存结果图像

### 2. 使用真实图像

**方法1：自动检测**
```bash
# 将两张图像放在同一目录下，然后运行：
python real_image_fusion.py
```

**方法2：指定图像路径**
```bash
python real_image_fusion.py image1.jpg image2.jpg
```

**方法3：指定图像和尺寸**
```bash
python real_image_fusion.py image1.jpg image2.jpg "(800, 600)"
```

### 3. 在代码中使用

```python
from laplacian_pyramid_fusion import LaplacianPyramid, create_vertical_split_mask
import cv2

# 读取图像
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 确保两张图像尺寸相同
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 创建拉普拉斯金字塔对象
lp = LaplacianPyramid(levels=6)

# 创建掩码（左半边为图像1，右半边为图像2）
mask = create_vertical_split_mask((img1.shape[0], img1.shape[1]), split_position=0.5)

# 执行融合
result = lp.blend_images(img1, img2, mask)

# 保存结果
cv2.imwrite('result.jpg', result)
```

## 掩码类型

### 1. 垂直分割掩码
```python
mask = create_vertical_split_mask(shape, split_position=0.5)
```
左侧显示图像1，右侧显示图像2

### 2. 水平分割掩码
```python
mask = create_horizontal_split_mask(shape, split_position=0.5)
```
上侧显示图像1，下侧显示图像2

### 3. 圆形掩码
```python
mask = create_circular_mask(shape, center=None, radius=None, smooth=True)
```
中心显示图像1，外围显示图像2

### 4. 渐变掩码
```python
from real_image_fusion import create_gradient_mask
mask = create_gradient_mask(shape, direction='horizontal', smooth=True)
```
从图像1平滑过渡到图像2

## 输出结果

运行示例代码后，会生成以下文件：

- `laplacian_pyramid_fusion_demo.png` - 基本融合演示
- `reconstruction_test.png` - 重建精度测试
- `vertical_fusion.png` - 垂直分割融合
- `horizontal_fusion.png` - 水平分割融合
- `circular_fusion.png` - 圆形融合
- `gradient_fusion.png` - 渐变融合
- `*_result.jpg` - 各种融合的结果图像

## 代码结构

### LaplacianPyramid 类

主要方法：

1. `build_gaussian_pyramid(image)` - 构建高斯金字塔
2. `build_laplacian_pyramid(image)` - 构建拉普拉斯金字塔
3. `reconstruct_from_laplacian(laplacian_pyramid)` - 从拉普拉斯金字塔重建图像
4. `blend_images(image1, image2, mask)` - 融合两张图像

### 关键实现细节

**拉普拉斯层计算**：
```python
# 上采样下一层
upsampled = cv2.pyrUp(next_level, dstsize=(current_level.shape[1], current_level.shape[0]))

# 计算拉普拉斯图像
laplacian = cv2.subtract(current_level, upsampled)
```

**图像重建**：
```python
# 上采样
reconstructed = cv2.pyrUp(reconstructed, dstsize=target_size)

# 加上拉普拉斯层
reconstructed = cv2.add(reconstructed, laplacian_pyramid[i])
```

**多层融合**：
```python
# 对每一层进行加权融合
blended = lap1 * mask_level + lap2 * (1 - mask_level)
```

## 参数调整

- `levels`：金字塔层数（默认5-6层）
  - 层数越多，融合越平滑，但计算时间越长
  - 建议根据图像尺寸选择：512x512 用 5-6 层

- `split_position`：分割位置（0-1之间）
  - 控制两张图像的比例

- `smooth`：是否平滑掩码边缘
  - True：产生更自然的过渡效果

## 优势对比

与直接拼接相比，拉普拉斯金字塔融合的优势：

1. **平滑过渡**：在不同频率上进行融合，避免明显的接缝
2. **保留细节**：高频细节得到更好的保留
3. **自然效果**：融合结果看起来更自然

## 应用场景

- 全景图拼接
- 多焦点图像融合
- HDR图像合成
- 艺术照片创作
- 医学图像融合

## 注意事项

1. 输入图像必须具有相同的尺寸
2. 掩码尺寸必须与图像匹配
3. 金字塔层数不宜过多（避免图像尺寸小于金字塔层数要求）
4. 重建时可能有微小的数值误差（通常小于1个像素值）

## 性能

- 512x512 图像，6层金字塔：约 0.1-0.3 秒（取决于硬件）
- 内存占用：约为原图的 1.5-2 倍

## 扩展功能

可以进一步扩展：

1. **自动掩码生成**：基于图像内容自动生成最佳融合掩码
2. **多图像融合**：支持融合3张或更多图像
3. **实时融合**：优化代码用于视频处理
4. **GPU加速**：使用CUDA加速计算

## 参考资料

- Burt, P. J., & Adelson, E. H. (1983). "The Laplacian Pyramid as a Compact Image Code"
- OpenCV Documentation: Image Pyramids

## 许可

MIT License

## 作者

计算机视觉课程作业 - 2025年
