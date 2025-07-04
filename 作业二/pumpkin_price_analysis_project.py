import pandas as pd
pumpkins = pd.read_csv('US-pumpkins.csv')
print(pumpkins.head())

print(pumpkins.head())#查看数据的前5行
print(pumpkins.tail())#查看数据的后5行
print(pumpkins.info()) #查看数据的组织结构

print(pumpkins.isnull().sum()) #检查每列数据中的缺失值的数量
print(pumpkins["Package"].is_unique)


