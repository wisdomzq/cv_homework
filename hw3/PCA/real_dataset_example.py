"""
使用真实数据集的表情识别示例
如果你有真实的表情数据集,可以使用这个脚本
"""

import numpy as np
from pca import PCA
from classifier import KNNClassifier, PCAExpressionRecognizer
from data_utils import (
    load_facial_expression_dataset,
    split_dataset,
    normalize_data,
    visualize_samples,
    visualize_principal_components,
    visualize_variance_explained,
    plot_confusion_matrix
)


def run_with_real_dataset(data_dir, n_components=50, k=5):
    """
    使用真实数据集运行表情识别
    
    参数:
        data_dir: str, 数据集目录路径
        n_components: int, 主成分数量
        k: int, KNN的K值
    """
    print("=" * 70)
    print("基于真实数据集的表情识别")
    print("=" * 70)
    
    # 加载数据
    print("\n步骤1: 加载数据")
    print("-" * 70)
    try:
        X, y, label_names = load_facial_expression_dataset(data_dir)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("\n请确保数据集目录结构正确:")
        print("data_dir/")
        print("  ├── class_0/")
        print("  │   ├── img1.jpg")
        print("  │   └── ...")
        print("  ├── class_1/")
        print("  │   └── ...")
        print("  └── ...")
        return
    
    # 划分数据集
    print("\n步骤2: 划分训练集和测试集")
    print("-" * 70)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    
    # 归一化
    print("\n步骤3: 数据归一化")
    print("-" * 70)
    X_train, X_test = normalize_data(X_train, X_test)
    print("数据已归一化到 [0, 1] 范围")
    
    # 可视化样本
    print("\n步骤4: 可视化样本")
    print("-" * 70)
    visualize_samples(X_train, y_train, label_names, n_samples=8)
    
    # 创建模型
    print("\n步骤5: 训练模型")
    print("-" * 70)
    pca = PCA(n_components=n_components)
    knn = KNNClassifier(k=k, distance_metric='euclidean')
    recognizer = PCAExpressionRecognizer(pca, knn)
    
    # 训练
    recognizer.fit(X_train, y_train, label_names)
    
    # 可视化主成分
    print("\n步骤6: 可视化分析")
    print("-" * 70)
    visualize_principal_components(pca, n_components=min(16, n_components))
    visualize_variance_explained(pca, save_path='real_variance_explained.png')
    
    # 评估
    print("\n步骤7: 模型评估")
    print("-" * 70)
    results = recognizer.evaluate(X_test, y_test)
    recognizer.print_evaluation_results(results)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(results['confusion_matrix'], label_names, 
                         save_path='real_confusion_matrix.png')
    
    # 分析结果
    print("\n步骤8: 结果分析")
    print("-" * 70)
    print(f"总体准确率: {results['accuracy']:.4f}")
    print(f"\n各类别F1分数:")
    for i, name in enumerate(label_names):
        print(f"  {name}: {results['f1_score'][i]:.4f}")
    
    # 计算平均指标
    avg_precision = np.mean(results['precision'])
    avg_recall = np.mean(results['recall'])
    avg_f1 = np.mean(results['f1_score'])
    
    print(f"\n平均指标:")
    print(f"  精确率: {avg_precision:.4f}")
    print(f"  召回率: {avg_recall:.4f}")
    print(f"  F1分数: {avg_f1:.4f}")
    
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    
    return results


def optimize_hyperparameters(data_dir):
    """
    优化超参数
    
    参数:
        data_dir: str, 数据集目录路径
    """
    print("=" * 70)
    print("超参数优化")
    print("=" * 70)
    
    # 加载数据
    try:
        X, y, label_names = load_facial_expression_dataset(data_dir)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
    X_train, X_test = normalize_data(X_train, X_test)
    
    # 测试不同的参数组合
    n_components_list = [20, 30, 50, 80, 100]
    k_values_list = [3, 5, 7, 9]
    
    best_accuracy = 0
    best_params = {}
    
    print("\n开始网格搜索...")
    print("-" * 70)
    
    for n_comp in n_components_list:
        for k_val in k_values_list:
            print(f"测试 n_components={n_comp}, k={k_val}...", end=" ")
            
            pca = PCA(n_components=n_comp)
            knn = KNNClassifier(k=k_val, distance_metric='euclidean')
            recognizer = PCAExpressionRecognizer(pca, knn)
            
            recognizer.fit(X_train, y_train, label_names)
            results = recognizer.evaluate(X_test, y_test)
            
            accuracy = results['accuracy']
            print(f"准确率: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'n_components': n_comp, 'k': k_val}
    
    print("\n" + "=" * 70)
    print("优化结果")
    print("=" * 70)
    print(f"最佳参数: n_components={best_params['n_components']}, k={best_params['k']}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    
    return best_params, best_accuracy


if __name__ == "__main__":
    import sys
    
    # 使用示例
    print("表情识别 - 真实数据集使用示例\n")
    
    if len(sys.argv) > 1:
        # 从命令行参数获取数据集路径
        data_dir = sys.argv[1]
        
        # 运行基本实验
        run_with_real_dataset(data_dir, n_components=50, k=5)
        
        # 如果需要优化超参数,取消下面的注释
        # optimize_hyperparameters(data_dir)
        
    else:
        print("使用方法:")
        print("  python real_dataset_example.py <数据集路径>")
        print("\n例如:")
        print("  python real_dataset_example.py ./fer2013_data")
        print("\n数据集目录结构应该是:")
        print("  dataset/")
        print("    ├── angry/")
        print("    │   ├── img1.jpg")
        print("    │   └── ...")
        print("    ├── happy/")
        print("    │   └── ...")
        print("    └── ...")
        print("\n或者你可以修改此脚本,直接指定数据集路径。")
