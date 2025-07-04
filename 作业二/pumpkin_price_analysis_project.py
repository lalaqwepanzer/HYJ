import pandas as pd
import matplotlib.pyplot as plt
# 设置中文字体（以 Windows 系统为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题[4,7](@ref)

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

# 绘制散点图
plt.scatter(price, month, color='orange', alpha=0.7)
plt.title('南瓜平均价格与月份关系')  # 中文标题
plt.xlabel('价格（单位：美元）')     # X轴中文标签
plt.ylabel('月份')                 # Y轴中文标签
plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线增强可读性
plt.show()

#以分组聚合的方式创建分组直方图
ax = new_pumpkins.groupby(['Month'])['Price'].mean().plot(
    kind='bar',
    color='skyblue',  # 设置柱状图颜色
    edgecolor='black',  # 设置边框颜色
    alpha=0.7,  # 设置透明度
    figsize=(10, 6)  # 调整图形大小
)
plt.title('南瓜价格月度分布', fontsize=14, pad=20)  # 标题及间距
plt.xlabel('月份', fontsize=12)  # 横轴标签
plt.ylabel('平均价格（美元）', fontsize=12)  # 纵轴标签

# 旋转x轴标签为水平（0度）
plt.xticks(rotation=0, ha='center')  # ha参数控制水平对齐方式

# 在柱子上方添加数值标签（可选）
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",  # 显示两位小数
        (p.get_x() + p.get_width() / 2, p.get_height()),  # 标签位置
        ha='center',  # 水平居中
        va='bottom',  # 垂直对齐到柱子顶部
        xytext=(0, 5),  # 垂直偏移量
        textcoords='offset points',
        fontsize=10
    )

# 添加网格线（虚线）
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()  # 自动调整子图间距
plt.show()




