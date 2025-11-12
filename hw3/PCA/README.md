# 基于PCA降维的表情识别系统

本项目实现了一个完整的基于PCA(主成分分析)降维的表情识别系统,包含自实现的PCA算法核心和多种实验分析。

## 项目特点

### 1. 核心算法自实现
- ✅ **PCA算法完整实现**:手动实现数据中心化、协方差矩阵计算、特征值分解、降维和重构
- ✅ **KNN分类器实现**:支持欧氏距离、曼哈顿距离和余弦距离
- ✅ **完整的实验分析**:多维度对比实验,生成详细的可视化结果

### 2. 实验内容丰富
- 基本表情识别实验
- 不同主成分数量对比(10~200维)
- 不同距离度量对比(欧氏、曼哈顿、余弦)
- 不同K值对比(K=1~20)
- 重构效果分析

## 项目结构

```
PCA/
├── pca.py                  # PCA核心算法实现
├── classifier.py           # KNN分类器和表情识别器
├── data_utils.py          # 数据处理和可视化工具
├── main_experiment.py     # 主实验脚本
├── requirements.txt       # 依赖库
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- numpy: 数值计算
- matplotlib: 数据可视化
- scikit-learn: 用于验证自实现的PCA算法正确性
- Pillow: 图像处理

## 使用方法

### 1. 运行完整实验

```bash
python main_experiment.py
```

这将运行所有实验并生成以下可视化结果:
- `sample_images.png`: 数据集样本展示
- `principal_components.png`: 主成分可视化
- `variance_explained.png`: 方差解释比例分析
- `confusion_matrix.png`: 混淆矩阵
- `component_comparison.png`: 不同主成分数量的性能对比
- `distance_metrics_comparison.png`: 不同距离度量的对比
- `k_value_comparison.png`: 不同K值的对比
- `reconstruction_comparison.png`: 图像重构效果对比

### 2. 测试PCA算法

```bash
python pca.py
```

功能:
- 测试PCA的基本功能
- 验证与sklearn实现的一致性
- 展示降维和重构效果

### 3. 测试分类器

```bash
python classifier.py
```

功能:
- 测试KNN分类器
- 测试表情识别器
- 展示评估指标

### 4. 测试数据处理

```bash
python data_utils.py
```

功能:
- 生成合成数据集
- 测试可视化功能

## PCA算法实现细节

### 核心流程

1. **数据中心化**
   ```python
   X_centered = X - np.mean(X, axis=0)
   ```

2. **计算协方差矩阵**
   ```python
   cov_matrix = X_centered.T @ X_centered / (n_samples - 1)
   ```

3. **特征值分解**
   ```python
   eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
   ```

4. **选择主成分并降维**
   ```python
   X_transformed = X_centered @ eigenvectors[:, :n_components]
   ```

5. **重构**
   ```python
   X_reconstructed = X_transformed @ eigenvectors[:, :n_components].T + mean
   ```

### 关键特性

- **方差比例选择**: 支持根据保留的方差比例自动确定主成分数量
- **累积方差分析**: 提供累积方差解释比例的计算
- **重构误差计算**: 评估降维的信息损失
- **与sklearn对比验证**: 确保算法实现的正确性

## 实验设置

### 数据集
- 使用合成的表情数据集进行测试
- 7个类别: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 每类100个样本,图像大小48×48
- 训练集:测试集 = 8:2

### 实验参数
- **主成分数量**: 10, 20, 30, 50, 80, 100, 150, 200
- **距离度量**: 欧氏距离, 曼哈顿距离, 余弦距离
- **K值**: 1, 3, 5, 7, 9, 11, 15, 20

## 实验结果说明

### 1. 主成分数量影响
- **准确率**: 随主成分数量增加而提升,在一定数量后趋于平稳
- **训练/测试时间**: 与主成分数量正相关
- **重构误差**: 与主成分数量负相关

### 2. 距离度量影响
- **欧氏距离**: 最常用,适用于大多数情况
- **曼哈顿距离**: 对异常值更鲁棒
- **余弦距离**: 适用于高维空间

### 3. K值影响
- K值过小: 容易过拟合,对噪声敏感
- K值过大: 可能欠拟合,决策边界过于平滑
- 通常K=3~7效果较好

## 使用真实数据集

如果你有真实的表情数据集,可以修改`main_experiment.py`中的数据加载部分:

```python
# 将这一行:
X, y, label_names = create_synthetic_expression_dataset(...)

# 替换为:
from data_utils import load_facial_expression_dataset
X, y, label_names = load_facial_expression_dataset('path/to/your/dataset')
```

数据集目录结构应该是:
```
dataset/
├── class_0/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class_1/
│   └── ...
└── ...
```

## 扩展功能

### 1. 添加更多分类器

可以在`classifier.py`中添加其他分类器,如SVM、随机森林等:

```python
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf')
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.model.predict(X_test)
```

### 2. 添加更多评估指标

可以添加ROC曲线、PR曲线等评估指标。

### 3. 使用其他降维方法

可以实现LDA、t-SNE等其他降维方法进行对比。

## 算法复杂度分析

### PCA算法
- **时间复杂度**: O(n²·m + m³)
  - n: 样本数量
  - m: 特征维度
  - 协方差矩阵计算: O(n²·m)
  - 特征值分解: O(m³)

- **空间复杂度**: O(n·m + m²)
  - 数据存储: O(n·m)
  - 协方差矩阵: O(m²)

### KNN分类器
- **时间复杂度**: O(n·m·k)
  - n: 训练样本数
  - m: 特征维度
  - k: 近邻数量

- **空间复杂度**: O(n·m)
  - 存储所有训练样本

## 优化建议

1. **大规模数据**: 使用增量PCA或随机PCA
2. **加速KNN**: 使用KD树或Ball树
3. **并行计算**: 使用多进程加速距离计算
4. **GPU加速**: 使用CuPy替换NumPy

## 常见问题

### Q1: 为什么自实现的PCA结果与sklearn略有不同?
A: 特征向量可能存在符号差异,但方向是一致的。这是正常现象,不影响降维效果。

### Q2: 如何选择合适的主成分数量?
A: 可以根据累积方差解释比例选择,通常保留90%~95%的方差即可。

### Q3: 为什么KNN在高维空间表现不好?
A: 这是"维度灾难"问题。PCA降维可以有效缓解这个问题。

## 参考资料

- Jolliffe, I. T. (2002). Principal Component Analysis. Springer.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86.

## 作者

计算机视觉课程作业 - PCA表情识别

## 许可

本项目仅用于学习和研究目的。
