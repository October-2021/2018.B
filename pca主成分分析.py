import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 创建包含滞后值的数据集
n_lags = 5  # 滞后值数量
data = time_series_df['Speakers'].values  # 从时间序列数据中提取Speakers列

lagged_data = np.zeros((len(data), n_lags))

for i in range(n_lags):
    lagged_data[i+1:, i] = data[:-i-1]

lagged_df = pd.DataFrame(lagged_data, columns=[f'Lag_{i+1}' for i in range(n_lags)])

# 执行PCA
pca = PCA()
pca.fit(lagged_df)

# 解释方差比例
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 绘制解释方差比例和累积解释方差比例
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', label='Explained Variance Ratio')
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', label='Cumulative Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance Ratio and Cumulative Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.show()
