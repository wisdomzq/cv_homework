"""
主实验脚本: 基于PCA降维的表情识别
包含多种实验和分析
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pca import PCA
from classifier import KNNClassifier, PCAExpressionRecognizer
from data_utils import (
    create_synthetic_expression_dataset,
    split_dataset,
    normalize_data,
    visualize_samples,
    visualize_principal_components,
    visualize_variance_explained,
    visualize_reconstruction,
    plot_confusion_matrix
)


def experiment_1_basic_recognition(X_train, X_test, y_train, y_test, label_names):
    """
    实验1: 基本的表情识别实验
    """
    print("\n" + "=" * 70)
    print("实验1: 基本的表情识别")
    print("=" * 70)
    
    # 使用PCA降维到50维
    n_components = 50
    print(f"\n使用PCA降维到 {n_components} 维")
    
    pca = PCA(n_components=n_components)
    knn = KNNClassifier(k=5, distance_metric='euclidean')
    recognizer = PCAExpressionRecognizer(pca, knn)
    
    # 训练
    start_time = time.time()
    recognizer.fit(X_train, y_train, label_names)
    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.2f} 秒")
    
    # 评估
    start_time = time.time()
    results = recognizer.evaluate(X_test, y_test)
    test_time = time.time() - start_time
    print(f"测试时间: {test_time:.2f} 秒")
    
    recognizer.print_evaluation_results(results)
    
    # 可视化主成分
    visualize_principal_components(pca, n_components=16)
    
    # 可视化方差解释
    visualize_variance_explained(pca)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'], label_names)
    
    return results


def experiment_2_component_comparison(X_train, X_test, y_train, y_test, label_names):
    """
    实验2: 不同主成分数量的对比实验
    """
    print("\n" + "=" * 70)
    print("实验2: 不同主成分数量的对比")
    print("=" * 70)
    
    # 测试不同的主成分数量
    component_numbers = [10, 20, 30, 50, 80, 100, 150, 200]
    
    accuracies = []
    train_times = []
    test_times = []
    reconstruction_errors = []
    
    for n_components in component_numbers:
        print(f"\n测试 n_components = {n_components}")
        
        pca = PCA(n_components=n_components)
        knn = KNNClassifier(k=5, distance_metric='euclidean')
        recognizer = PCAExpressionRecognizer(pca, knn)
        
        # 训练
        start_time = time.time()
        recognizer.fit(X_train, y_train, label_names)
        train_time = time.time() - start_time
        train_times.append(train_time)
        
        # 测试
        start_time = time.time()
        results = recognizer.evaluate(X_test, y_test)
        test_time = time.time() - start_time
        test_times.append(test_time)
        
        accuracies.append(results['accuracy'])
        
        # 计算重构误差
        recon_error = pca.reconstruction_error(X_test)
        reconstruction_errors.append(recon_error)
        
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"训练时间: {train_time:.2f}s, 测试时间: {test_time:.2f}s")
        print(f"重构误差: {recon_error:.6f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 准确率 vs 主成分数量
    axes[0, 0].plot(component_numbers, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Principal Components')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Number of Components')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练时间 vs 主成分数量
    axes[0, 1].plot(component_numbers, train_times, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Principal Components')
    axes[0, 1].set_ylabel('Training Time (s)')
    axes[0, 1].set_title('Training Time vs Number of Components')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试时间 vs 主成分数量
    axes[1, 0].plot(component_numbers, test_times, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Principal Components')
    axes[1, 0].set_ylabel('Testing Time (s)')
    axes[1, 0].set_title('Testing Time vs Number of Components')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 重构误差 vs 主成分数量
    axes[1, 1].plot(component_numbers, reconstruction_errors, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Principal Components')
    axes[1, 1].set_ylabel('Reconstruction Error (MSE)')
    axes[1, 1].set_title('Reconstruction Error vs Number of Components')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('component_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比结果已保存至 component_comparison.png")
    plt.close()
    
    # 打印总结
    print("\n" + "=" * 70)
    print("主成分数量对比总结")
    print("=" * 70)
    print(f"{'Components':<15} {'Accuracy':<15} {'Train Time(s)':<15} {'Test Time(s)':<15} {'Recon Error':<15}")
    print("-" * 70)
    for i, n_comp in enumerate(component_numbers):
        print(f"{n_comp:<15} {accuracies[i]:<15.4f} {train_times[i]:<15.2f} "
              f"{test_times[i]:<15.2f} {reconstruction_errors[i]:<15.6f}")
    
    # 找到最佳主成分数量
    best_idx = np.argmax(accuracies)
    print(f"\n最佳主成分数量: {component_numbers[best_idx]} (准确率: {accuracies[best_idx]:.4f})")
    
    return component_numbers, accuracies, train_times, test_times, reconstruction_errors


def experiment_3_distance_metrics(X_train, X_test, y_train, y_test, label_names):
    """
    实验3: 不同距离度量的对比
    """
    print("\n" + "=" * 70)
    print("实验3: 不同距离度量的对比")
    print("=" * 70)
    
    distance_metrics = ['euclidean', 'manhattan', 'cosine']
    n_components = 50
    
    results_dict = {}
    
    for metric in distance_metrics:
        print(f"\n测试距离度量: {metric}")
        
        pca = PCA(n_components=n_components)
        knn = KNNClassifier(k=5, distance_metric=metric)
        recognizer = PCAExpressionRecognizer(pca, knn)
        
        # 训练和评估
        recognizer.fit(X_train, y_train, label_names)
        results = recognizer.evaluate(X_test, y_test)
        
        results_dict[metric] = results
        print(f"准确率: {results['accuracy']:.4f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (metric, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = results['confusion_matrix']
        im = ax.imshow(cm, cmap='Blues')
        
        ax.set_title(f'{metric}\nAccuracy: {results["accuracy"]:.4f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # 简化标签显示
        n_classes = len(label_names)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels([name[:3] for name in label_names], fontsize=8)
        ax.set_yticklabels([name[:3] for name in label_names], fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('distance_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("\n距离度量对比结果已保存至 distance_metrics_comparison.png")
    plt.close()
    
    return results_dict


def experiment_4_k_value_comparison(X_train, X_test, y_train, y_test, label_names):
    """
    实验4: 不同K值的对比
    """
    print("\n" + "=" * 70)
    print("实验4: 不同K值的对比")
    print("=" * 70)
    
    k_values = [1, 3, 5, 7, 9, 11, 15, 20]
    n_components = 50
    
    accuracies = []
    
    for k in k_values:
        print(f"\n测试 K = {k}")
        
        pca = PCA(n_components=n_components)
        knn = KNNClassifier(k=k, distance_metric='euclidean')
        recognizer = PCAExpressionRecognizer(pca, knn)
        
        recognizer.fit(X_train, y_train, label_names)
        results = recognizer.evaluate(X_test, y_test)
        
        accuracies.append(results['accuracy'])
        print(f"准确率: {results['accuracy']:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs K Value in KNN', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # 标记最佳K值
    best_idx = np.argmax(accuracies)
    plt.plot(k_values[best_idx], accuracies[best_idx], 'r*', markersize=20, 
             label=f'Best K={k_values[best_idx]} (Acc={accuracies[best_idx]:.4f})')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('k_value_comparison.png', dpi=150, bbox_inches='tight')
    print("\nK值对比结果已保存至 k_value_comparison.png")
    plt.close()
    
    print(f"\n最佳K值: {k_values[best_idx]} (准确率: {accuracies[best_idx]:.4f})")
    
    return k_values, accuracies


def experiment_5_reconstruction(X_train, X_test, y_train, y_test):
    """
    实验5: 重构效果分析
    """
    print("\n" + "=" * 70)
    print("实验5: 重构效果分析")
    print("=" * 70)
    
    component_numbers = [10, 30, 50, 100, 200]
    
    fig, axes = plt.subplots(len(component_numbers) + 1, 5, figsize=(10, 2 * (len(component_numbers) + 1)))
    
    # 显示原始图像
    for i in range(5):
        img = X_test[i].reshape(48, 48)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)
    
    # 不同主成分数量的重构
    for idx, n_components in enumerate(component_numbers):
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        
        X_test_transformed = pca.transform(X_test[:5])
        X_test_reconstructed = pca.inverse_transform(X_test_transformed)
        
        recon_error = np.mean((X_test[:5] - X_test_reconstructed) ** 2)
        
        for i in range(5):
            img = X_test_reconstructed[i].reshape(48, 48)
            axes[idx + 1, i].imshow(img, cmap='gray')
            axes[idx + 1, i].axis('off')
            
            if i == 0:
                axes[idx + 1, i].set_ylabel(f'n={n_components}\nMSE={recon_error:.4f}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    print("重构对比结果已保存至 reconstruction_comparison.png")
    plt.close()


def main():
    """
    主函数: 运行所有实验
    """
    print("=" * 70)
    print("基于PCA降维的表情识别实验")
    print("=" * 70)
    
    # 创建合成数据集
    print("\n步骤1: 准备数据")
    print("-" * 70)
    X, y, label_names = create_synthetic_expression_dataset(
        n_samples_per_class=100,
        n_classes=7,
        image_size=(48, 48),
        noise_level=0.3
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    
    # 归一化
    X_train, X_test = normalize_data(X_train, X_test)
    
    # 可视化样本
    visualize_samples(X_train, y_train, label_names, n_samples=5)
    
    # 运行实验
    print("\n步骤2: 运行实验")
    print("-" * 70)
    
    # 实验1: 基本识别
    experiment_1_basic_recognition(X_train, X_test, y_train, y_test, label_names)
    
    # 实验2: 主成分数量对比
    experiment_2_component_comparison(X_train, X_test, y_train, y_test, label_names)
    
    # 实验3: 距离度量对比
    experiment_3_distance_metrics(X_train, X_test, y_train, y_test, label_names)
    
    # 实验4: K值对比
    experiment_4_k_value_comparison(X_train, X_test, y_train, y_test, label_names)
    
    # 实验5: 重构效果
    experiment_5_reconstruction(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 70)
    print("所有实验完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - sample_images.png: 样本图像")
    print("  - principal_components.png: 主成分可视化")
    print("  - variance_explained.png: 方差解释比例")
    print("  - confusion_matrix.png: 混淆矩阵")
    print("  - component_comparison.png: 主成分数量对比")
    print("  - distance_metrics_comparison.png: 距离度量对比")
    print("  - k_value_comparison.png: K值对比")
    print("  - reconstruction_comparison.png: 重构效果对比")


if __name__ == "__main__":
    main()
