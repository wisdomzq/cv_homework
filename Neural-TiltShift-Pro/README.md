# 🏙️ Neural-TiltShift-Pro: 神经渲染移轴生成器

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

**一个结合深度学习与计算机图形学的专业级移轴摄影生成器**

[功能特性](#✨-核心功能) • [技术原理](#🛠️-技术原理) • [快速开始](#📦-安装与运行) • [演示效果](#🎬-应用场景)

</div>

---

## 📖 项目简介

Neural-TiltShift-Pro 是一个基于 AI 深度估计和计算机视觉技术的**专业级移轴效果生成器**。它不仅能模拟传统的微缩模型效果，还通过引入 **沙姆定律 (Scheimpflug Principle)** 的数学模型，实现了专业移轴镜头的 **Virtual Tilt (虚拟俯仰/摇摆)** 和 **Virtual Shift (虚拟平移/透视校正)** 功能。

### 🎯 项目亮点

- ✅ **三种对焦模式**: 几何线性、沙姆定律、AI 深度感知，适配不同场景
- ✅ **智能对焦系统**: YOLOv8 语义识别 + 区域选择 + 点击对焦
- ✅ **专业光学模拟**: 3D 焦平面旋转 + 透视校正，媲美真实移轴镜头
- ✅ **Web 交互界面**: Gradio 实时预览，无需编程即可使用
- ✅ **完整渲染管线**: 多级景深模糊 + 色彩增强，输出专业品质

---

## ✨ 核心功能

### 1. 📏 几何线性模式 (Geometric Linear Mode) ⭐ 新增
**专为垂直俯拍、卫星图等深度估计失效场景设计**：
- **纯几何渐变**: 完全不依赖深度估计模型，基于像素的几何位置生成线性渐变 Mask
- **多种渐变方向**:
    - **垂直渐变 (Vertical)**: 从上到下或从下到上的线性模糊，适合俯拍街道
    - **水平渐变 (Horizontal)**: 从左到右的线性模糊
    - **径向渐变 (Radial)**: 从中心向外扩散的环形模糊（类似经典移轴效果）
    - **对角线渐变 (Diagonal)**: 左上到右下或右上到左下的斜向模糊
- **解决问题**: 当 MiDaS 深度估计在平面场景（如地图、建筑立面）产生噪点时，使用几何模式可获得干净、均匀的渐变效果

### 2. 📐 沙姆定律模式 (Scheimpflug Mode)
这是本系统的核心创新点，模拟了专业移轴镜头的光学特性：
- **3D 焦平面旋转 (Virtual Tilt)**: 不再局限于基于距离的简单模糊，而是可以在 3D 空间中旋转焦平面。
    - **Tilt-X (俯仰)**: 控制焦平面绕 X 轴旋转，实现"地面清晰、天空模糊"或反之的效果。
    - **Tilt-Y (摇摆)**: 控制焦平面绕 Y 轴旋转，实现"左侧清晰、右侧模糊"的效果。
- **透视校正 (Virtual Shift)**: 模拟移轴镜头的平移功能，通过梯形变换校正建筑摄影中的透视变形（如仰拍时建筑线条汇聚的问题）。

### 3. 🧠 AI 深度感知 (Depth Awareness)
- 集成 **MiDaS** (Monocular Depth Estimation) 模型，从单张 RGB 图像生成高质量的相对深度图。
- 相比传统的线性梯度模糊，基于深度的模糊能更真实地处理物体遮挡和复杂的场景几何。

### 4. 🎯 智能对焦系统
- **语义自动对焦 (Semantic Auto-Focus)**: 集成 **YOLOv8** 目标检测模型，自动识别画面中的人、车等主体，并自动将焦点设置在最显著的目标上。
- **区域选择对焦**: 支持手动框选感兴趣区域，系统会自动计算该区域的平均深度并进行对焦。
- **点击对焦**: 支持交互式点击图片任意位置进行对焦。

### 5. 🎨 渲染与增强
- **多级景深模糊**: 使用基于 Mask 的多级高斯模糊算法，模拟真实镜头的光圈虚化效果，过渡自然平滑。
- **微缩色彩增强**: 自动提升饱和度和对比度，增强"玩具模型"的视觉质感。

---

## 🛠️ 技术原理

### 1. 几何线性对焦 (Geometric Linear Focus)
对于垂直俯拍、卫星图等缺乏透视关系的场景，深度估计模型（MiDaS）往往会产生大量噪点。几何线性模式通过构建基于像素坐标的归一化坐标系来生成 Mask。

以**垂直渐变**为例，生成从上到下的归一化坐标 $c \in [0, 1]$：
$$
c(y) = \frac{y}{H}
$$

焦点位置 $p_{focus}$ 处的模糊度最低，使用高斯型衰减函数计算 Mask：
$$
M(y) = \exp\left(-\left(\frac{|c(y) - p_{focus}|}{\sigma}\right)^k\right)
$$

其中 $\sigma$ 为焦点宽度，$k$ 为衰减指数（控制过渡锐度）。

**优势**: 完全避免深度估计噪点，产生干净、均匀的渐变模糊效果。

### 2. 沙姆定律 (Scheimpflug Principle) 实现
传统的移轴滤镜通常只使用简单的线性梯度 Mask。本系统构建了一个像素级的伪 3D 坐标系 $(u, v, d)$，其中 $d$ 为深度值。

焦平面的法向量 $\vec{n}$ 由用户输入的俯仰角 $\theta_x$ 和摇摆角 $\theta_y$ 计算得出：
$$
\vec{n} = (\sin\theta_y, -\sin\theta_x \cos\theta_y, \cos\theta_x \cos\theta_y)
$$

每个像素点 $P(u, v, d)$ 到焦平面的带符号距离 $D$ 为：
$$
D = \vec{n} \cdot P - d_{focus}
$$

最后通过 Sigmoid 函数将距离映射为模糊权重 Mask，从而实现精确的 3D 焦平面控制。

### 3. 透视校正 (Perspective Correction)
利用 OpenCV 的透视变换 (Perspective Transform) 技术。根据用户输入的 Shift 强度，计算源图像四个角点的偏移量，构建变换矩阵 $M$：
$$
dst = M \cdot src
$$
从而将梯形变形的图像校正为矩形，拉直垂直线条。

### 4. 深度估计与渲染
- **深度估计**: 使用 `MiDaS` 模型提取相对深度。
- **渲染管线**: 
    1. **Shift**: 应用透视校正。
    2. **Mask**: 计算沙姆定律 Mask 或深度 Mask。
    3. **Blur**: 根据 Mask 权重，将原图与多级模糊图进行 Alpha Blending 混合。

---

## 🎬 应用场景

### 适用场景

| 场景类型 | 推荐模式 | 效果说明 |
|---------|---------|----------|
| 🏙️ **城市街景** | 沙姆定律模式 | 利用深度信息精确控制焦平面，突出建筑主体 |
| 🛰️ **垂直俯拍/卫星图** | 几何线性模式 | 避免深度估计噪点，产生干净的渐变效果 |
| 🚗 **车辆/人物摄影** | 智能对焦 | 自动识别主体，一键生成专业景深效果 |
| 🏗️ **建筑摄影** | 透视校正 + 沙姆定律 | 拉直线条 + 控制焦平面，专业建筑摄影效果 |
| 🎨 **创意设计** | 径向/对角线渐变 | 独特的艺术化模糊效果 |

### 典型用例

1. **微缩世界效果**: 将真实街景转换为玩具模型般的微缩景观
2. **突出主体**: 通过选择性模糊弱化背景，引导视觉焦点
3. **建筑矫正**: 校正仰拍/俯拍导致的透视变形
4. **创意后期**: 为摄影作品添加专业的景深和色彩增强

---

## 📦 安装与运行

### 环境要求
- Python 3.8+
- PyTorch >= 2.0
- OpenCV, NumPy, Gradio, Ultralytics

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/your-repo/Neural-TiltShift-Pro.git
cd Neural-TiltShift-Pro
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
python app.py
```
启动后，浏览器访问显示的本地地址 (通常是 `http://127.0.0.1:7860`)。

---

## 📂 文件结构

```
Neural-TiltShift-Pro/
├── app.py                  # Gradio Web UI 主程序
├── requirements.txt        # 项目依赖
├── configs/
│   └── settings.yaml       # 配置文件
├── src/
│   ├── renderer.py         # 核心渲染器 (沙姆定律、透视校正、模糊算法)
│   ├── depth_engine.py     # MiDaS 深度估计引擎
│   ├── auto_focus.py       # YOLOv8 语义自动对焦模块
│   ├── inpainting.py       # 图像修复模块 (用于处理遮挡区域)
│   └── video_processor.py  # 视频处理模块
└── README.md               # 项目文档
```

## 🚀 技术栈

### 核心框架
- **深度学习**: PyTorch 2.0+, TorchVision
- **计算机视觉**: OpenCV, NumPy, Pillow
- **Web 界面**: Gradio 4.0+
- **配置管理**: PyYAML

### AI 模型
- **深度估计**: MiDaS (Intel ISL) - DPT_Large / MiDaS_small
- **目标检测**: YOLOv8 (Ultralytics) - yolov8n.pt

### 算法技术
- 沙姆定律 (Scheimpflug Principle) 3D 几何建模
- 透视变换 (Perspective Transform)
- 多级高斯模糊 (Multi-level Gaussian Blur)
- Alpha Blending 图像合成
- HSV 色彩空间增强

---

## 📊 性能指标

### 处理速度 (在 Apple M1 Pro 上测试)

| 图像尺寸 | 几何线性模式 | 沙姆定律模式 | 智能对焦模式 |
|---------|-------------|-------------|-------------|
| 1920×1080 | ~0.5s | ~2.5s | ~3.0s |
| 3840×2160 (4K) | ~1.2s | ~5.8s | ~6.5s |

**注**: 首次运行包含模型加载时间（约 3-5 秒），后续处理速度显著提升。

### 模型大小
- **MiDaS_small**: ~40 MB (推荐，速度快)
- **DPT_Large**: ~350 MB (精度高)
- **YOLOv8n**: ~6 MB (轻量级检测)

---

## 📝 注意事项

- 首次运行时会自动下载 MiDaS 和 YOLOv8 模型权重，请保持网络连接。
- 沙姆定律模式下的计算量较大，建议在有 GPU 的环境下运行以获得最佳体验。
- 对于垂直俯拍图片，强烈推荐使用**几何线性模式**以避免深度估计噪点。
- 支持 CPU 运行，但 GPU 加速可提升 3-5 倍处理速度。

---

## 🙏 致谢

本项目使用了以下开源项目和模型：

- [MiDaS](https://github.com/isl-org/MiDaS) - Intel ISL 深度估计模型
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics 目标检测框架
- [Gradio](https://gradio.app/) - 快速构建 ML Web 界面
- [OpenCV](https://opencv.org/) - 计算机视觉基础库

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 👤 作者

**计算机视觉课程项目** - 大三上学期

如有问题或建议，欢迎提 Issue 或 PR！
