import pandas as pd
import numpy as np
import statsmodels.api as sm

# 读取数据（假设您已经有了相关的数据集）
# 请将下面的文件路径替换为您实际的数据文件路径
data = pd.read_csv('your_data.csv')

# 定义自变量和因变量
# 假设 'Population', 'Education', 'Economic_Factors' 是自变量，'Language_Usage' 是因变量
X = data[['Population', 'Education', 'Economic_Factors']]
y = data['Language_Usage']

# 添加截距（常数项）
X = sm.add_constant(X)

# 拟合多元线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归结果摘要
print(model.summary())
