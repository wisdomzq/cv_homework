"""
基于PCA降维的表情识别分类器
使用K-NN算法进行分类
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K近邻分类器
    """
    
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        初始化KNN分类器
        
        参数:
            k: int, 近邻数量
            distance_metric: str, 距离度量方式 ('euclidean', 'manhattan', 'cosine')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """
        训练KNN分类器(实际上只是存储训练数据)
        
        参数:
            X_train: ndarray, 训练数据特征
            y_train: ndarray, 训练数据标签
        """
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def _compute_distance(self, x1, x2):
        """
        计算两个样本之间的距离
        
        参数:
            x1, x2: ndarray, 两个样本
            
        返回:
            distance: float, 距离值
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            dot_product = np.dot(x1, x2)
            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return 1 - dot_product / (norm1 * norm2)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
    
    def predict_single(self, x):
        """
        预测单个样本
        
        参数:
            x: ndarray, 单个样本
            
        返回:
            prediction: 预测标签
        """
        # 计算与所有训练样本的距离
        distances = []
        for x_train in self.X_train:
            dist = self._compute_distance(x, x_train)
            distances.append(dist)
        
        # 找到k个最近的邻居
        distances = np.array(distances)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # 投票决定类别
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X_test):
        """
        预测多个样本
        
        参数:
            X_test: ndarray, 测试数据
            
        返回:
            predictions: ndarray, 预测标签
        """
        predictions = []
        for x in X_test:
            pred = self.predict_single(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        计算准确率
        
        参数:
            X_test: ndarray, 测试数据
            y_test: ndarray, 测试标签
            
        返回:
            accuracy: float, 准确率
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy


class PCAExpressionRecognizer:
    """
    基于PCA降维的表情识别器
    """
    
    def __init__(self, pca, classifier):
        """
        初始化表情识别器
        
        参数:
            pca: PCA对象
            classifier: 分类器对象
        """
        self.pca = pca
        self.classifier = classifier
        self.label_names = None
        
    def fit(self, X_train, y_train, label_names=None):
        """
        训练表情识别器
        
        参数:
            X_train: ndarray, 训练数据
            y_train: ndarray, 训练标签
            label_names: list, 标签名称
        """
        # PCA降维
        print(f"使用PCA降维: {X_train.shape[1]} -> {self.pca.n_components}")
        X_train_pca = self.pca.fit_transform(X_train)
        print(f"降维后数据形状: {X_train_pca.shape}")
        print(f"保留方差比例: {self.pca.get_cumulative_variance_ratio()[-1]:.4f}")
        
        # 训练分类器
        self.classifier.fit(X_train_pca, y_train)
        self.label_names = label_names
        
        return self
    
    def predict(self, X_test):
        """
        预测表情
        
        参数:
            X_test: ndarray, 测试数据
            
        返回:
            predictions: ndarray, 预测标签
        """
        # PCA降维
        X_test_pca = self.pca.transform(X_test)
        
        # 分类
        predictions = self.classifier.predict(X_test_pca)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        参数:
            X_test: ndarray, 测试数据
            y_test: ndarray, 测试标签
            
        返回:
            results: dict, 评估结果
        """
        # 预测
        predictions = self.predict(X_test)
        
        # 计算准确率
        accuracy = np.mean(predictions == y_test)
        
        # 计算混淆矩阵
        n_classes = len(np.unique(y_test))
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for true_label, pred_label in zip(y_test, predictions):
            confusion_matrix[true_label, pred_label] += 1
        
        # 计算每个类别的精确率、召回率和F1分数
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_score = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
                          if (precision[i] + recall[i]) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'predictions': predictions
        }
        
        return results
    
    def print_evaluation_results(self, results):
        """
        打印评估结果
        
        参数:
            results: dict, 评估结果
        """
        print("\n" + "=" * 50)
        print("评估结果")
        print("=" * 50)
        print(f"总体准确率: {results['accuracy']:.4f}")
        
        if self.label_names is not None:
            print("\n各类别性能:")
            print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
            print("-" * 50)
            for i, name in enumerate(self.label_names):
                print(f"{name:<15} {results['precision'][i]:<10.4f} "
                      f"{results['recall'][i]:<10.4f} {results['f1_score'][i]:<10.4f}")
        
        print("\n混淆矩阵:")
        if self.label_names is not None:
            header = "实际\\预测"
            print(f"{header:<15}", end="")
            for name in self.label_names:
                print(f"{name[:10]:<12}", end="")
            print()
        print(results['confusion_matrix'])


def test_classifier():
    """
    测试分类器
    """
    from pca import PCA
    
    # 生成测试数据
    np.random.seed(42)
    n_samples_per_class = 50
    n_classes = 3
    n_features = 100
    
    X_train = []
    y_train = []
    for i in range(n_classes):
        # 每个类别有不同的均值
        X_class = np.random.randn(n_samples_per_class, n_features) + i * 2
        X_train.append(X_class)
        y_train.extend([i] * n_samples_per_class)
    
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    
    # 生成测试数据
    X_test = []
    y_test = []
    for i in range(n_classes):
        X_class = np.random.randn(20, n_features) + i * 2
        X_test.append(X_class)
        y_test.extend([i] * 20)
    
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)
    
    # 创建识别器
    print("测试表情识别器")
    print("=" * 50)
    pca = PCA(n_components=20)
    classifier = KNNClassifier(k=5, distance_metric='euclidean')
    recognizer = PCAExpressionRecognizer(pca, classifier)
    
    # 训练
    label_names = ['Class 0', 'Class 1', 'Class 2']
    recognizer.fit(X_train, y_train, label_names)
    
    # 评估
    results = recognizer.evaluate(X_test, y_test)
    recognizer.print_evaluation_results(results)


if __name__ == "__main__":
    test_classifier()
