# 基于拉普拉斯金字塔的图像融合（课程项目报告）

作者：——
日期：2025-11-05
环境：Windows + Python 3.x + OpenCV/NumPy/Matplotlib

## 1. 背景介绍

图像融合旨在将多源图像的信息在视觉上自然、在统计上互补地合成为一幅图像，常见于全景拼接、多焦点融合、HDR 合成、计算摄影与医学影像等。理想的融合应：
- 在接缝处不过度突兀（无明显“硬边”）；
- 保留各图像的细节和结构；
- 在亮度/色彩/纹理统计上保持自然一致。

问题定义：给定两张空间已对齐的输入图像 $I_1, I_2$ 以及融合掩码 $M\in[0,1]^{H\times W}$（0 表示更偏向 $I_2$，1 表示更偏向 $I_1$），输出融合图 $F$。

相关方法简述：
- 直接（硬）拼接 / 线性 Alpha 融合：实现简单，但硬边/模糊/重影明显。
- 多带/金字塔融合（Burt & Adelson 1983）：在不同频带分别融合，兼顾低频过渡与高频细节保留，是经典且有效的方法。
- 梯度域/Poisson 融合：从梯度重建，擅长光照/颜色过渡，但计算量较大、边界条件敏感。
- 引导滤波/GF 融合、拉普拉斯引导等：更“结构保持”，但需更多参数与实现细节。

现有方法问题：
- 直接拼接：接缝伪影严重；
- 简单 Alpha：全频带统一加权，易出现模糊或幽灵影；
- 梯度域：高计算/实现复杂度、边界处理棘手；
- 多带融合：参数（层数、掩码带宽）影响较大，需要实验选型。

本项目选择“拉普拉斯金字塔融合”，在实验效果、复杂度、可解释性之间取得良好平衡。

## 2. 动机与方法

动机：
- 针对硬边/直拼接缝明显的问题，引入多尺度频带融合：在低频上做平滑过渡以避免亮度/色调突变，在高频上选择性保留各自的纹理细节，使过渡自然且锐度可控。
- 相比梯度域方法，本项目强调简单、稳定、工程友好且可复现；同时通过多种掩码与层数消融，提供可解释的可调节性。

方法概述：
- 高斯金字塔 $G_i$：$G_0=I$，$G_{i+1}=\operatorname{pyrDown}(G_i)$；
- 拉普拉斯金字塔 $L_i$：$L_i = G_i - \operatorname{pyrUp}(G_{i+1})$，顶层为 $G_{L}$；
- 掩码金字塔 $M_i$：对 $M$ 构建高斯金字塔以形成尺度一致的软权重；
- 层级融合：$B_i = M_i\odot L_i^{(1)} + (1-M_i)\odot L_i^{(2)}$；
- 重建：自顶层逐层上采样并相加 $\hat F = B_0 + \sum_{i\ge1} \operatorname{pyrUp}^i(B_i)$。

本实现特点：
- 不使用现成“拉普拉斯金字塔”黑箱函数，全部手写（允许使用 OpenCV 的高斯金字塔算子 `pyrDown/pyrUp`）。
- 提供多种掩码：垂直/水平、圆形、渐变、文字/棋盘等，便于场景化选择。
- 配套“质量分析”与“消融评测”工具，定量度量接缝与保真度：`analyze_fusion.py`、`benchmark_levels.py`。

流程图（伪）：
1) 构建 $L^{(1)}, L^{(2)}$ 与 $M$ 的金字塔；
2) 对每层进行 $B_i = M_i\cdot L^{(1)}_i + (1-M_i)\cdot L^{(2)}_i$；
3) 自顶向下重建得到融合图 $F$；
4) 计算指标（PSNR/SSIM、SeamGrad、LaplacianVar）并可视化面板与频谱差异。

## 3. 实验与结果分析

实现与环境：
- 主要代码（均为本项目自编）：
  - `laplacian_pyramid_fusion.py`（核心实现与可视化）
  - `analyze_fusion.py`（指标与面板）
  - `real_image_fusion.py`（真实图像融合入口，自动保存指标与图）
  - `test_laplacian_pyramid.py`（正确性与接缝能量对比单测）
  - `benchmark_levels.py`（层数/掩码消融与时间统计）
  - 依赖库：OpenCV、NumPy、Matplotlib（外部库，不含高阶融合黑箱函数）

可复现实验：
```powershell
# 安装依赖
pip install -r requirements.txt

# 真实图像融合 + 自动分析
python real_image_fusion.py image1.jpg image2.jpg

# 单次分析（可选）
python analyze_fusion.py image1.jpg image2.jpg --size 512,512 --levels 6 --mask vertical --out case1

# 层数/掩码消融（耗时与指标曲线）
python benchmark_levels.py
```

实验数据：两张真实图像（分辨率统一为 512×512），掩码取：垂直硬边（vertical）、圆形（circular）、渐变（gradient），金字塔层数 $L=6$。

评价指标：
- 局部保真度：PSNR/SSIM（左区域对 $I_1$，右区域对 $I_2$）。
- 接缝平滑：SeamGrad（接缝带内梯度能量，↓更好）、接缝色差（与直拼差异）。
- 清晰度：全图拉普拉斯方差 LaplacianVar（↑更锐，但直拼伪边也会抬升）。

定量结果（三种掩码）：

| 掩码 | SeamGrad_fused | SeamGrad_direct | 下降幅度 | PSNR_left (dB) | SSIM_left | PSNR_right (dB) | SSIM_right | LapVar_fused | LapVar_direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 垂直 Vertical | 0.0920 | 0.2067 | −55.5% | 29.99 | 0.9738 | 21.47 | 0.8926 | 0.0369 | 0.1058 |
| 圆形 Circular | 0.0626 | 0.1081 | −42.1% | 28.46 | 0.8776 | 21.99 | 0.8819 | 0.0452 | 0.1240 |
| 渐变 Gradient | 0.0658 | 0.1114 | −40.9% | 24.59 | 0.8937 | 20.87 | 0.8373 | 0.0237 | 0.0671 |

（对应图：`real_vertical_panel.png`、`real_circular_panel.png`、`real_gradient_panel.png`）

分析：
- 接缝：三种掩码下，Laplacian 融合的 SeamGrad 均显著低于直拼（约 41%–56% 降幅）；圆形/渐变掩码最平滑，垂直硬边次之。
- 保真：各自区域 PSNR/SSIM 较高，说明融合在“属于自己的区域”能保真；渐变掩码因过渡带更宽，局部保真略降，换来更柔的过渡。
- 清晰度：融合相对直拼的 LapVar 更低，表明接缝伪边被抑制；渐变掩码整体最“柔”，圆形在“平滑 vs 清晰”间更平衡。

补充：接缝能量对比单测（`test_laplacian_pyramid.py`）验证在硬边掩码条件下，Laplacian 融合的接缝梯度能量不高于直接硬拼接（通常更小），支持上述结论的稳健性。

## 4. 总结与展望

贡献与收获：
- 手写实现了拉普拉斯金字塔融合（不依赖黑箱），并系统评估其在真实数据上的接缝抑制与保真能力；
- 设计了多类掩码并完成消融，给出“圆形/平滑掩码 + L=5–6”的推荐配置；
- 构建了可复用的评测工具（SeamGrad、局部 PSNR/SSIM、LaplacianVar、频谱），提升分析深度与可复现性；
- 完成单测与性能对比，为工程落地提供基础保障。

不足与改进方向：
- 对曝光/色温差异大的输入，建议融合前做亮度/色彩对齐（如直方图匹配、线性增益）；
- 自动掩码：可尝试基于梯度/显著性/语义分割生成更优的过渡区域；
- 多图像/视频融合：扩展到时序一致性的多帧融合，加入运动估计与时域平滑；
- 更先进的多尺度框架：如引导滤波金字塔、拉普拉斯-引导混合、神经金字塔等，以进一步提升边界质量与细节保真；
- GPU 加速与实时：在高分辨率/多帧场景下进行优化。

---

### 附：公式
- 拉普拉斯层：$$ L_i = G_i - \operatorname{pyrUp}(G_{i+1}) $$
- 重建：$$ G_i \approx L_i + \operatorname{pyrUp}(G_{i+1}),\quad F = \sum_i B_i^{\uparrow i} $$
- 融合层：$$ B_i = M_i\cdot L_i^{(1)} + (1-M_i)\cdot L_i^{(2)} $$
- PSNR：$$ \operatorname{PSNR}=20\log_{10}(\mathrm{MAX})-10\log_{10}(\mathrm{MSE}) $$
- SSIM（简化全局版）：$$ \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)} $$

### 附：代码结构（路径）
- `laplacian_pyramid_fusion.py`：核心实现与可视化
- `real_image_fusion.py`：真实图像实验入口
- `analyze_fusion.py`：指标与面板输出
- `benchmark_levels.py`：层数/掩码消融与时间曲线
- `test_laplacian_pyramid.py`：正确性与接缝能量对比单测
- `requirements.txt`、`README.md`、`INSTALL.md`
