# SIFT算法自实现 - 完整项目

本项目从零开始实现SIFT（Scale-Invariant Feature Transform）算法的核心流程，并进行详细的特征匹配和质量分析。

## 🎯 项目特点

### ✅ 核心算法自实现
- **高斯金字塔构建**：多尺度空间表示
- **DOG金字塔**：高斯差分检测
- **极值点检测**：3D空间极值搜索
- **关键点精确定位**：亚像素级定位 + 低对比度过滤 + 边缘响应消除
- **主方向分配**：36bins方向直方图，支持多主方向
- **描述符生成**：128维SIFT描述符（4×4×8结构）

### 📊 详细分析功能
- **金字塔可视化**：高斯金字塔和DOG金字塔的完整展示
- **关键点分布分析**：空间分布、尺度分布、响应强度、方向统计
- **匹配质量分析**：距离分布、位移向量场、内点率评估
- **对比实验**：自实现vs OpenCV SIFT

### 🔧 实用工具
- 支持切换使用自实现或OpenCV的SIFT
- FLANN/BFMatcher两种匹配方式
- Lowe's ratio test筛选
- RANSAC单应性矩阵估计
- 完整的可视化报告

## 📁 项目结构

```
SIFT/
├── sift_algorithm.py    # SIFT算法核心实现
├── main.py              # 完整的匹配和分析主程序
├── sift_matching.py     # 原始匹配程序（使用OpenCV）
├── example.py           # 示例程序
├── requirements.txt     # 依赖包
└── README_FULL.md       # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备图像

将两张要匹配的图像放在SIFT目录下，命名为：
- `image1.jpg`
- `image2.jpg`

### 3. 运行完整分析

```bash
python main.py
```

这将执行完整的SIFT检测、匹配和分析流程，生成详细的可视化报告。

### 4. 测试单个图像

```bash
python sift_algorithm.py
```

对比自实现SIFT和OpenCV SIFT的效果。

## 📖 详细使用

### 使用自实现的SIFT

```python
from sift_algorithm import MySIFT
import cv2

# 读取图像
image = cv2.imread('image.jpg')

# 创建SIFT检测器
my_sift = MySIFT(
    num_octaves=4,          # 金字塔组数
    num_scales=5,           # 每组尺度数
    sigma=1.6,              # 初始sigma
    contrast_threshold=0.04, # 对比度阈值
    edge_threshold=10       # 边缘响应阈值
)

# 检测特征点和计算描述符
keypoints, descriptors, gaussian_pyr, dog_pyr = my_sift.detect_and_compute(image)

print(f"检测到 {len(keypoints)} 个特征点")
```

### 完整匹配流程

```python
from main import comprehensive_matching

# 执行完整分析
results = comprehensive_matching(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    use_my_sift=True,        # True: 自实现, False: OpenCV
    ratio_threshold=0.75,    # Lowe's ratio test阈值
    output_dir="output"      # 输出目录
)

# 访问结果
keypoints1 = results['keypoints1']
keypoints2 = results['keypoints2']
matches = results['matches']
homography = results['homography']
```

### 参数调优建议

#### SIFT检测参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `num_octaves` | 4 | 金字塔组数 | 增加可检测更多尺度，但速度变慢 |
| `num_scales` | 5 | 每组尺度数 | 通常3-5即可 |
| `sigma` | 1.6 | 初始高斯sigma | 影响特征点的尺度 |
| `contrast_threshold` | 0.04 | 对比度阈值 | 降低可获得更多特征点 |
| `edge_threshold` | 10 | 边缘响应阈值 | 增大可过滤更多边缘点 |

#### 匹配参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `ratio_threshold` | 0.75 | Lowe's ratio | 0.7-0.8，越小越严格 |
| `use_flann` | True | 使用FLANN匹配 | 大数据集建议True |

## 📊 输出结果

程序会在输出目录生成以下文件：

### 金字塔可视化
- `pyramid_img1_gaussian.png` - 图像1的高斯金字塔
- `pyramid_img1_dog.png` - 图像1的DOG金字塔
- `pyramid_img2_gaussian.png` - 图像2的高斯金字塔
- `pyramid_img2_dog.png` - 图像2的DOG金字塔

### 关键点分析
- `keypoint_analysis_img1.png` - 图像1关键点分析（6个子图）
  - 空间分布热力图
  - 尺度分布
  - 响应强度分布
  - 主方向分布（极坐标）
  - 尺度-响应散点图
  - 统计信息
- `keypoint_analysis_img2.png` - 图像2关键点分析
- `keypoints.png` - 两幅图像的特征点可视化

### 匹配分析
- `matching_quality_analysis.png` - 匹配质量分析（6个子图）
  - 匹配距离分布
  - 位移大小分布
  - 位移方向分布（极坐标）
  - 位移向量场
  - 距离排名曲线
  - 统计信息
- `matches.png` - 匹配连线图（前100个）
- `ransac_matches.png` - RANSAC筛选后的内点

## 🔬 算法详解

### 1. 高斯金字塔构建

```
组0: [σ, k·σ, k²·σ, k³·σ, k⁴·σ]
组1: [σ, k·σ, k²·σ, k³·σ, k⁴·σ] (下采样)
组2: [σ, k·σ, k²·σ, k³·σ, k⁴·σ] (下采样)
组3: [σ, k·σ, k²·σ, k³·σ, k⁴·σ] (下采样)
```

其中 k = 2^(1/(s-3))，s为每组尺度数

### 2. DOG金字塔

```
DOG(x,y,σ) = L(x,y,k·σ) - L(x,y,σ)
```

每组生成s-1个DOG图像

### 3. 极值点检测

在DOG空间中，检查每个像素的3×3×3邻域（26个邻居），判断是否为极值点。

### 4. 关键点精确定位

- **泰勒展开**：亚像素级定位
- **对比度检查**：|DOG(x,y,σ)| > threshold
- **边缘响应消除**：Tr(H)²/Det(H) < (r+1)²/r

### 5. 主方向分配

- 统计邻域内梯度方向（36bins）
- 高斯加权
- 找到峰值方向
- 支持多个主方向（>80%峰值）

### 6. 描述符生成

- 16×16邻域，分为4×4子区域
- 每个子区域统计8个方向的梯度
- 生成128维向量（4×4×8）
- 归一化并限制最大值为0.2

## 🎓 理论基础

### SIFT的不变性

1. **尺度不变性**：通过尺度空间理论和DOG实现
2. **旋转不变性**：通过主方向归一化实现
3. **亮度不变性**：通过梯度归一化实现
4. **部分仿射不变性**：通过局部邻域描述实现

### Lowe's Ratio Test

```python
if dist(match1) < ratio × dist(match2):
    accept match1
```

典型ratio值：0.7-0.8，用于过滤模糊匹配。

### RANSAC算法

迭代寻找最优单应性矩阵：
1. 随机选择4对点
2. 计算单应性矩阵H
3. 统计内点数量
4. 保留最佳H

## 📈 性能对比

### 自实现 vs OpenCV

| 指标 | 自实现 | OpenCV | 备注 |
|------|--------|--------|------|
| 准确性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OpenCV更优化 |
| 速度 | 较慢 | 快 | OpenCV有C++优化 |
| 可定制性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 自实现更灵活 |
| 可读性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 自实现便于学习 |

## 🐛 常见问题

### Q1: 检测到的特征点太少？
A: 尝试降低`contrast_threshold`（如0.03或0.02）

### Q2: 有很多边缘点？
A: 增大`edge_threshold`（如15或20）

### Q3: 匹配效果不好？
A: 调整`ratio_threshold`，尝试0.6-0.8之间的值

### Q4: 处理速度太慢？
A: 减少`num_octaves`或降低图像分辨率

## 📚 参考文献

1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints". International Journal of Computer Vision, 60(2), 91-110.

2. Lowe, D. G. (1999). "Object recognition from local scale-invariant features". ICCV.

3. OpenCV SIFT Tutorial: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html

## 📝 作业建议

本实现适合作为计算机视觉课程作业，包含：

✅ **算法核心自实现**（不使用cv2.SIFT_create）
✅ **完整的实验分析**（多角度可视化）
✅ **参数对比实验**（不同阈值的影响）
✅ **性能评估**（与OpenCV对比）
✅ **详细文档**（算法原理和使用说明）

生成的可视化图表可直接用于报告。

## 🎨 可视化示例

程序会生成丰富的可视化结果：

1. **金字塔结构**：理解多尺度表示
2. **关键点分布**：分析特征的空间、尺度、方向特性
3. **匹配质量**：评估匹配的准确性和一致性
4. **RANSAC效果**：展示几何约束的作用

## 📧 联系方式

如有问题，欢迎提Issue或Pull Request。

## 📄 License

MIT License
