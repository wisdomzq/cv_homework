# SIFT图像特征匹配

本项目使用SIFT（Scale-Invariant Feature Transform）算法实现两幅图像的特征点检测和匹配。

## 功能特性

- ✨ SIFT特征点检测和描述符计算
- 🔍 支持BFMatcher和FLANN两种匹配算法
- 📊 Lowe's ratio test筛选优质匹配
- 🎯 RANSAC算法计算单应性矩阵
- 📈 完整的可视化结果（特征点、匹配、RANSAC筛选）
- 💾 自动保存结果图像

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

1. 准备两张要匹配的图像（例如：`image1.jpg` 和 `image2.jpg`）

2. 运行主程序：

```bash
python sift_matching.py
```

### 自定义参数

在代码中修改参数：

```python
from sift_matching import match_images

results = match_images(
    img1_path="path/to/image1.jpg",
    img2_path="path/to/image2.jpg",
    ratio_threshold=0.75,  # Lowe's ratio test阈值 (0-1)
    use_flann=True,        # True: FLANN匹配器, False: BFMatcher
    save_results=True,     # 是否保存结果
    output_dir="output"    # 输出目录
)
```

### 高级用法

使用SIFTMatcher类进行更细粒度的控制：

```python
from sift_matching import SIFTMatcher
import cv2

# 读取图像
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# 创建匹配器
matcher = SIFTMatcher(ratio_threshold=0.75)

# 检测特征点
kp1, desc1 = matcher.detect_and_compute(img1)
kp2, desc2 = matcher.detect_and_compute(img2)

# 匹配特征
matches = matcher.match_features(desc1, desc2, use_flann=True)

# 计算单应性矩阵
H, mask = matcher.find_homography(kp1, kp2, matches)

# 绘制匹配结果
match_img = matcher.draw_matches(img1, kp1, img2, kp2, matches)
```

## 算法说明

### SIFT算法

SIFT（Scale-Invariant Feature Transform）是一种检测和描述图像局部特征的算法，具有以下特点：

- **尺度不变性**：对图像缩放保持不变
- **旋转不变性**：对图像旋转保持不变
- **亮度不变性**：对光照变化具有鲁棒性
- **仿射变换不变性**：对视角变化具有一定鲁棒性

### 匹配流程

1. **特征检测**：使用SIFT算法检测两幅图像的特征点
2. **描述符计算**：为每个特征点计算128维描述符
3. **特征匹配**：使用KNN算法找到最相似的特征点对
4. **Lowe's Ratio Test**：筛选优质匹配（最近邻距离 < 0.75 × 次近邻距离）
5. **RANSAC筛选**：使用RANSAC算法去除外点，计算单应性矩阵

### 参数说明

- **ratio_threshold** (0-1)：Lowe's ratio test阈值，越小匹配越严格
  - 推荐值：0.7-0.8
  - 较小值（0.6-0.7）：更严格，匹配更准确但数量较少
  - 较大值（0.8-0.9）：更宽松，匹配数量多但可能有误匹配

- **use_flann** (True/False)：选择匹配算法
  - True：使用FLANN（快速近似最近邻搜索），速度快
  - False：使用BFMatcher（暴力匹配），精度高但速度慢

## 输出结果

程序会生成以下可视化结果：

1. **keypoints.png**：两幅图像的特征点可视化
2. **matches.png**：特征点匹配结果（前50个最佳匹配）
3. **ransac_matches.png**：RANSAC筛选后的内点匹配

所有结果保存在`output/`目录中。

## 示例场景

SIFT特征匹配适用于：

- 📷 图像拼接/全景图生成
- 🔎 物体识别与检测
- 📍 图像配准
- 🎯 3D重建
- 🏞️ 场景识别
- 🤖 视觉SLAM

## 注意事项

- 确保输入图像具有足够的纹理信息（特征点丰富）
- 两幅图像应该有一定的重叠区域或相同物体
- 对于纯色或重复纹理的图像，匹配效果可能较差
- 图像尺寸过大时处理速度会较慢，可以考虑先缩放

## 参考资料

- Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
- OpenCV SIFT Documentation: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html

## License

MIT License
