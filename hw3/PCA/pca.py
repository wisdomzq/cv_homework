"""
PCA (主成分分析) 核心算法实现
不使用sklearn等现成的PCA实现，手动实现算法流程
"""

import numpy as np


class PCA:
    """
    主成分分析(PCA)类
    
    手动实现PCA的核心算法流程:
    1. 数据中心化
    2. 计算协方差矩阵
    3. 特征值分解
    4. 选择主成分
    5. 数据降维和重构
    """
    
    def __init__(self, n_components=None, variance_ratio=None):
        """
        初始化PCA
        
        参数:
            n_components: int, 保留的主成分数量
            variance_ratio: float, 保留的方差比例(如0.95表示保留95%的方差)
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.mean_ = None  # 数据均值
        self.components_ = None  # 主成分(特征向量)
        self.explained_variance_ = None  # 每个主成分解释的方差
        self.explained_variance_ratio_ = None  # 每个主成分解释的方差比例
        self.singular_values_ = None  # 奇异值
        
    def fit(self, X):
        """
        拟合PCA模型
        
        参数:
            X: ndarray, shape (n_samples, n_features)
               训练数据
        """
        n_samples, n_features = X.shape
        
        # 步骤1: 数据中心化 - 减去均值
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 步骤2: 计算协方差矩阵
        # Cov = (X^T * X) / (n-1)
        # 这里手动实现协方差矩阵计算
        cov_matrix = self._compute_covariance_matrix(X_centered)
        
        # 步骤3: 特征值分解
        # 手动使用numpy的eig函数进行特征值分解(这是线性代数的基础操作)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 步骤4: 对特征值和特征向量进行排序(从大到小)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 转换为实数(有时会有很小的虚部)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 步骤5: 计算方差解释比例
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        
        # 步骤6: 确定保留的主成分数量
        if self.n_components is None:
            if self.variance_ratio is not None:
                # 根据方差比例确定主成分数量
                cumsum_variance = np.cumsum(explained_variance_ratio)
                self.n_components = np.searchsorted(cumsum_variance, self.variance_ratio) + 1
            else:
                # 默认保留所有主成分
                self.n_components = n_features
        
        # 限制主成分数量不超过特征数
        self.n_components = min(self.n_components, n_features)
        
        # 步骤7: 保存结果
        self.components_ = eigenvectors[:, :self.n_components].T  # shape: (n_components, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components]
        self.singular_values_ = np.sqrt(eigenvalues[:self.n_components] * (n_samples - 1))
        
        return self
    
    def _compute_covariance_matrix(self, X_centered):
        """
        手动计算协方差矩阵
        
        参数:
            X_centered: ndarray, 中心化后的数据
            
        返回:
            cov_matrix: ndarray, 协方差矩阵
        """
        n_samples = X_centered.shape[0]
        # 协方差矩阵 = X^T @ X / (n-1)
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        return cov_matrix
    
    def transform(self, X):
        """
        将数据投影到主成分空间
        
        参数:
            X: ndarray, shape (n_samples, n_features)
               待转换的数据
               
        返回:
            X_transformed: ndarray, shape (n_samples, n_components)
                          降维后的数据
        """
        if self.mean_ is None:
            raise ValueError("模型还未拟合，请先调用fit方法")
        
        # 中心化
        X_centered = X - self.mean_
        
        # 投影到主成分空间
        # X_transformed = X_centered @ components.T
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        拟合模型并转换数据
        
        参数:
            X: ndarray, shape (n_samples, n_features)
               训练数据
               
        返回:
            X_transformed: ndarray, shape (n_samples, n_components)
                          降维后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        将降维后的数据重构回原始空间
        
        参数:
            X_transformed: ndarray, shape (n_samples, n_components)
                          降维后的数据
                          
        返回:
            X_reconstructed: ndarray, shape (n_samples, n_features)
                            重构后的数据
        """
        if self.mean_ is None:
            raise ValueError("模型还未拟合，请先调用fit方法")
        
        # 重构: X_reconstructed = X_transformed @ components + mean
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        
        return X_reconstructed
    
    def get_cumulative_variance_ratio(self):
        """
        获取累积方差解释比例
        
        返回:
            cumulative_variance_ratio: ndarray, 累积方差解释比例
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("模型还未拟合，请先调用fit方法")
        
        return np.cumsum(self.explained_variance_ratio_)
    
    def reconstruction_error(self, X):
        """
        计算重构误差
        
        参数:
            X: ndarray, 原始数据
            
        返回:
            error: float, 平均重构误差(MSE)
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        error = np.mean((X - X_reconstructed) ** 2)
        return error


def compare_pca_implementations(X, n_components=10):
    """
    比较自实现的PCA和sklearn的PCA结果
    用于验证算法正确性
    """
    from sklearn.decomposition import PCA as SklearnPCA
    
    # 自实现的PCA
    pca_custom = PCA(n_components=n_components)
    X_custom = pca_custom.fit_transform(X)
    
    # sklearn的PCA
    pca_sklearn = SklearnPCA(n_components=n_components)
    X_sklearn = pca_sklearn.fit_transform(X)
    
    # 比较结果
    print("=" * 50)
    print("PCA实现对比验证")
    print("=" * 50)
    print(f"自实现PCA - 转换后数据形状: {X_custom.shape}")
    print(f"sklearn PCA - 转换后数据形状: {X_sklearn.shape}")
    print(f"\n方差解释比例对比:")
    print(f"自实现: {pca_custom.explained_variance_ratio_[:5]}")
    print(f"sklearn: {pca_sklearn.explained_variance_ratio_[:5]}")
    
    # 注意: 特征向量可能有符号差异,但方向是一致的
    # 所以比较时使用绝对值
    similarity = np.mean(np.abs(np.abs(X_custom) - np.abs(X_sklearn)))
    print(f"\n降维后数据的平均差异: {similarity:.6f}")
    
    if similarity < 0.01:
        print("✓ 自实现的PCA与sklearn的PCA结果一致!")
    else:
        print("⚠ 结果存在差异，需要检查实现")
    
    return pca_custom, pca_sklearn


if __name__ == "__main__":
    # 测试代码
    print("PCA算法测试\n")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    X_test = np.random.randn(n_samples, n_features)
    
    # 测试PCA
    print("1. 基本功能测试")
    print("-" * 50)
    pca = PCA(n_components=10)
    pca.fit(X_test)
    X_transformed = pca.transform(X_test)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    print(f"原始数据形状: {X_test.shape}")
    print(f"降维后形状: {X_transformed.shape}")
    print(f"重构后形状: {X_reconstructed.shape}")
    print(f"重构误差(MSE): {pca.reconstruction_error(X_test):.6f}")
    print(f"前10个主成分解释的方差比例: {pca.explained_variance_ratio_}")
    print(f"累积方差比例: {pca.get_cumulative_variance_ratio()[-1]:.4f}")
    
    # 测试方差比例模式
    print("\n2. 方差比例模式测试")
    print("-" * 50)
    pca_ratio = PCA(variance_ratio=0.95)
    X_transformed_ratio = pca_ratio.fit_transform(X_test)
    print(f"保留95%方差需要的主成分数: {pca_ratio.n_components}")
    print(f"实际累积方差比例: {pca_ratio.get_cumulative_variance_ratio()[-1]:.4f}")
    
    # 与sklearn对比
    print("\n3. 与sklearn实现对比")
    print("-" * 50)
    compare_pca_implementations(X_test, n_components=10)
