"""
数据处理和可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def load_facial_expression_dataset(data_dir, image_size=(48, 48)):
    """
    加载表情识别数据集
    
    数据集目录结构应该是:
    data_dir/
        class_0/
            img1.jpg
            img2.jpg
            ...
        class_1/
            img1.jpg
            ...
    
    参数:
        data_dir: str, 数据集根目录
        image_size: tuple, 图像大小
        
    返回:
        X: ndarray, 图像数据 (n_samples, height*width)
        y: ndarray, 标签
        label_names: list, 标签名称
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    X = []
    y = []
    label_names = []
    
    # 获取所有类别目录
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    if len(class_dirs) == 0:
        raise ValueError(f"在 {data_dir} 中没有找到类别目录")
    
    label_names = class_dirs
    
    print(f"找到 {len(class_dirs)} 个类别: {class_dirs}")
    
    for label_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"加载类别 '{class_name}': {len(image_files)} 张图像")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # 读取图像
                img = Image.open(img_path).convert('L')  # 转为灰度图
                img = img.resize(image_size)
                img_array = np.array(img).flatten()  # 展平为一维向量
                
                X.append(img_array)
                y.append(label_idx)
            except Exception as e:
                print(f"加载图像 {img_path} 失败: {e}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\n数据集加载完成:")
    print(f"  样本数量: {X.shape[0]}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别数量: {len(label_names)}")
    
    return X, y, label_names


def create_synthetic_expression_dataset(n_samples_per_class=100, n_classes=7, 
                                        image_size=(48, 48), noise_level=0.3):
    """
    创建合成的表情数据集用于测试
    
    参数:
        n_samples_per_class: int, 每个类别的样本数
        n_classes: int, 类别数
        image_size: tuple, 图像大小
        noise_level: float, 噪声水平
        
    返回:
        X: ndarray, 图像数据
        y: ndarray, 标签
        label_names: list, 标签名称
    """
    np.random.seed(42)
    
    n_features = image_size[0] * image_size[1]
    X = []
    y = []
    
    # 表情标签
    label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    label_names = label_names[:n_classes]
    
    print(f"生成合成数据集:")
    print(f"  每类样本数: {n_samples_per_class}")
    print(f"  类别数: {n_classes}")
    print(f"  图像大小: {image_size}")
    
    for class_idx in range(n_classes):
        # 为每个类别创建一个基础模板
        template = np.random.randn(n_features) * 0.5 + class_idx
        
        for _ in range(n_samples_per_class):
            # 添加噪声
            noise = np.random.randn(n_features) * noise_level
            sample = template + noise
            
            # 归一化到 [0, 255]
            sample = (sample - sample.min()) / (sample.max() - sample.min()) * 255
            
            X.append(sample)
            y.append(class_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"数据集生成完成: {X.shape}")
    
    return X, y, label_names


def normalize_data(X_train, X_test=None):
    """
    归一化数据到 [0, 1]
    
    参数:
        X_train: ndarray, 训练数据
        X_test: ndarray, 测试数据(可选)
        
    返回:
        X_train_normalized: ndarray
        X_test_normalized: ndarray (如果提供)
    """
    X_train_normalized = X_train / 255.0
    
    if X_test is not None:
        X_test_normalized = X_test / 255.0
        return X_train_normalized, X_test_normalized
    
    return X_train_normalized


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    
    参数:
        X: ndarray, 数据
        y: ndarray, 标签
        test_size: float, 测试集比例
        random_state: int, 随机种子
        
    返回:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    return X_train, X_test, y_train, y_test


def visualize_samples(X, y, label_names, image_size=(48, 48), n_samples=10):
    """
    可视化样本图像
    
    参数:
        X: ndarray, 图像数据
        y: ndarray, 标签
        label_names: list, 标签名称
        image_size: tuple, 图像大小
        n_samples: int, 显示的样本数
    """
    n_classes = len(label_names)
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 1.5, n_classes * 1.5))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx in range(n_classes):
        class_samples = X[y == class_idx][:n_samples]
        
        for sample_idx in range(min(n_samples, len(class_samples))):
            img = class_samples[sample_idx].reshape(image_size)
            
            ax = axes[class_idx, sample_idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            if sample_idx == 0:
                ax.set_title(label_names[class_idx], fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print("\n样本图像已保存至 sample_images.png")
    plt.close()


def visualize_principal_components(pca, image_size=(48, 48), n_components=16):
    """
    可视化主成分
    
    参数:
        pca: PCA对象
        image_size: tuple, 图像大小
        n_components: int, 显示的主成分数
    """
    n_components = min(n_components, pca.components_.shape[0])
    
    n_rows = int(np.sqrt(n_components))
    n_cols = int(np.ceil(n_components / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()
    
    for i in range(n_components):
        component = pca.components_[i].reshape(image_size)
        
        # 归一化以便可视化
        component = (component - component.min()) / (component.max() - component.min())
        
        axes[i].imshow(component, cmap='gray')
        axes[i].set_title(f'PC{i+1}\n({pca.explained_variance_ratio_[i]:.2%})', fontsize=8)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('principal_components.png', dpi=150, bbox_inches='tight')
    print("主成分可视化已保存至 principal_components.png")
    plt.close()


def visualize_variance_explained(pca, save_path='variance_explained.png'):
    """
    可视化方差解释比例
    
    参数:
        pca: PCA对象
        save_path: str, 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 单个主成分的方差解释比例
    n_components = len(pca.explained_variance_ratio_)
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained Ratio')
    ax1.set_title('Variance Explained by Each Principal Component')
    ax1.grid(True, alpha=0.3)
    
    # 累积方差解释比例
    cumsum = pca.get_cumulative_variance_ratio()
    ax2.plot(range(1, n_components + 1), cumsum, 'b-', marker='o')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Variance Explained Ratio')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"方差解释比例图已保存至 {save_path}")
    plt.close()


def visualize_reconstruction(X_original, X_reconstructed, image_size=(48, 48), n_samples=5):
    """
    可视化重构效果
    
    参数:
        X_original: ndarray, 原始图像
        X_reconstructed: ndarray, 重构图像
        image_size: tuple, 图像大小
        n_samples: int, 显示的样本数
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    for i in range(n_samples):
        # 原始图像
        img_orig = X_original[i].reshape(image_size)
        axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # 重构图像
        img_recon = X_reconstructed[i].reshape(image_size)
        axes[1, i].imshow(img_recon, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
    print("重构效果图已保存至 reconstruction.png")
    plt.close()


def plot_confusion_matrix(confusion_matrix, label_names, save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    参数:
        confusion_matrix: ndarray, 混淆矩阵
        label_names: list, 标签名称
        save_path: str, 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # 设置标签
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white")
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存至 {save_path}")
    plt.close()


if __name__ == "__main__":
    # 测试数据处理功能
    print("测试数据处理模块\n")
    
    # 创建合成数据集
    X, y, label_names = create_synthetic_expression_dataset(
        n_samples_per_class=50,
        n_classes=7,
        image_size=(48, 48)
    )
    
    # 划分数据集
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    
    # 归一化
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # 可视化样本
    visualize_samples(X_train, y_train, label_names, n_samples=5)
    
    print("\n数据处理模块测试完成!")
