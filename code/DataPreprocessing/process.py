import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy.stats import zscore
import numpy as np



file_path = '../data/star_to_fill.csv'
# 加载数据
df = pd.read_csv(file_path, na_values=["\\N"])

# 使用随机森林来填充一个具有缺失值的列，例如，total_tran_amt
def impute_with_rf(df, column):
    # 将待预测列和其它包含 NaN 或无穷大的列排除在外，仅使用其它完整的列作为特征
    columns = [col for col in df.columns
               if df[col].notna().all() and (df[col].dtypes in ['int64', 'float64'] and np.isfinite(df[col]).all())
               and col != column and col != 'uid']

    df_notnull = df.loc[df[column].notnull()]
    df_isnull = df.loc[df[column].isnull()]

    X = df_notnull[columns]
    y = df_notnull[column]

    rf = RandomForestRegressor(random_state=0, n_estimators=100)
    rf.fit(X, y)

    predicted_values = rf.predict(df_isnull[columns])
    df_isnull[column] = predicted_values

    df_new = pd.concat([df_notnull, df_isnull])

    return df_new

not_to_process=[]
# 先保存这两列，以备后续使用
uid = df['uid']
level = df['star_level'] if 'star_level' in df.columns else df['credit_level']
# 创建 StandardScaler 实例
scaler = StandardScaler()

# 选择要标准化的数值列，排除 'uid' 和 'star_level' 或 'credit_level'
numeric_columns = [col for col in df.columns if (df[col].dtypes in ['int64', 'float64']) and col not in ['uid', 'star_level', 'credit_level']]

# 将 Z-score 标准化应用到数据
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
# 根据文件名称决定处理方式
if 'credit' in file_path:
    # 将credit_level放置到最后一列
    credit_level = df['credit_level']
    df = df.drop(['credit_level'], axis=1)
    df['credit_level'] = credit_level
    not_to_process = ['uid', 'credit_level']
elif 'star' in file_path:
    # 将star_level放置到最后一列
    star_level = df['star_level']
    df = df.drop(['star_level'], axis=1)
    df['star_level'] = star_level
    not_to_process = ['uid', 'star_level']

# 删除UID为空的行
df = df[df['uid'].notna()]

# 编码类别变量
label_encoder = LabelEncoder()
for col in df.columns[df.dtypes == 'object']:
    if col not in not_to_process:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# 对每一列应用随机森林填充，除了不需要处理的列
for col in df.columns:
    # 如果某个列缺失值大于90%，剔除该列
    if df[col].isnull().mean() > 0.9:
        df = df.drop(col, axis=1)
    # 如果某个列缺失值不大于90%，使用随机森林进行填充
    elif col not in not_to_process and df[col].isnull().any():
        df = impute_with_rf(df, col)



# 异常值处理
# 通常情况下，我们会选择3σ原则（或者说是Z-score方法）进行异常值检测，
# 这里简单起见，我们直接把所有大于3个标准差的数值视为异常值，用该列的中位数进行替换
for column in df.columns:
    if column not in not_to_process and df[column].dtype != 'object':  # 仅处理数值型数据
        mean = df[column].mean()
        std = df[column].std()
        outliers = df[(df[column] - mean).abs() > 3 * std]
        df.loc[outliers.index, column] = df[column].median()
# # 删除'uid'和'level'列，并保存为后续使用
uid = df['uid']
level = df['star_level']
df = df.drop(['uid', 'star_level'], axis=1)

# 计算所有列的 Z-score
z_scores = df.apply(zscore)

# 查找所有 Z-score 大于 3 或小于 -3 的值，并将其替换为 NaN
df_outliers_removed = df[(z_scores < 3).all(axis=1) & (z_scores > -3).all(axis=1)]

# 区间缩放
scaler = MinMaxScaler()
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# 将 'uid' 列添加回到 DataFrame 的第一列
df.insert(0, 'uid', uid)

# 将 'star_level' 或 'credit_level' 列添加回到 DataFrame 的最后一列
df['star_level' if 'star_level' in df.columns else 'credit_level'] = level

# 检查是否还有缺失值
print(df.isnull().sum())
# 保存处理后的数据
df.to_csv('../data/star_to_fill.csv', index=False)
