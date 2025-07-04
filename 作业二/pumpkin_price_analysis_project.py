import pandas as pd
pumpkins = pd.read_csv('US-pumpkins.csv')
print(pumpkins.head())

print(pumpkins.head())#查看数据的前5行
print(pumpkins.tail())#查看数据的后5行
print(pumpkins.info()) #查看数据的组织结构

print(pumpkins.isnull().sum()) #检查每列数据中的缺失值的数量
print(pumpkins["Package"].is_unique)


#统计"Package"列不同数据出现的频数以选择合适的数据
print(pumpkins["Package"].value_counts()) #统计"Package"列不同值出现的次数

#筛选只包含字符串“bushel”的数据
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
print(pumpkins)

#提取月份数据与价格数据
price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2 #price是平均价格
month = pd.DatetimeIndex(pumpkins['Date']).month #将日期列转换为日期并提取出月份数据

#提取数据至新列
new_pumpkins=pd.DataFrame({'Month':month,'Package':pumpkins['Package'],'LowPrice':pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
new_pumpkins.info()

new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9) # loc()函数通过索引行标签索引行数据，即包含1 1/9的行，price = price/(1 + 1/9)
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2) #同上






