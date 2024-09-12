import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 读取数据
csv_data_path = './Biomarker_Clinical_Data_Images.csv'
data = pd.read_csv(csv_data_path)

# 准备特征和标签
X = data.iloc[:, 0].values.reshape(-1, 1)  # 图片名
y = data.iloc[:, 2:17].astype(str).agg('-'.join, axis=1).values  # 合并多标签

# 分层采样
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 获取训练集和验证集的索引
train_indices, val_indices = None, None
for train_index, val_index in skf.split(X, y):
    train_indices, val_indices = train_index, val_index
    break  # 只需要第一折

# 生成训练集和验证集
train_data = data.iloc[train_indices]
val_data = data.iloc[val_indices]

# 保存到csv文件
train_data.to_csv('train_dataset.csv', index=False)
val_data.to_csv('val_dataset.csv', index=False)

train_data.shape, val_data.shape
