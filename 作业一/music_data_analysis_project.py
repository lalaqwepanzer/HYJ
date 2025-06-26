#1.数据加载
import matplotlib.pyplot as plt
import pandas as pd
 
df = pd.read_csv("nigerian-songs.csv")#以pandas库的read_csv函数读取csv文件
df.head()#查看前5行数据

#2.数据探索
df.info()
df.describe() 

#3.数据可视化
#3.1艺术家类型数量的条形图
import seaborn as sns
import matplotlib.pyplot as plt

# 对不同音乐家类型进行统计汇总
top = df['artist_top_genre'].value_counts()
# 设置图表大小
plt.figure(figsize=(10, 7))
# 绘制条形图
barplot = sns.barplot(x=top[:5].index, y=top[:5].values, hue=top[:5].index, palette="coolwarm", legend=False)
# 设置条形图标题内容及颜色
plt.title('Top genres', color='blue')
# 添加数值标签
for index, value in enumerate(top[:5].values):
    barplot.text(index, value + 0.05, str(value), ha='center')
# 显示图形
plt.show()

#3.2三大流派的条形图
# 筛选特定的音乐类型
df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
# 筛选流行度大于0的数据
df = df[(df['popularity'] > 0)]
# 统计每种音乐类型的数量
top = df['artist_top_genre'].value_counts()
# 创建一个颜色列表，用于为不同的条形图设置颜色
colors = ['red', 'green', 'blue']
# 绘制条形图
plt.figure(figsize=(10, 7))
bars = plt.bar(top.index, top.values, color=colors)
# 在每个条形图上显示具体数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')
# 设置标题
plt.title('Top genres', color='blue')
# 显示图形
plt.show()

#3.3相关性热力图
# 计算相关性矩阵
corrmat = df.corr(numeric_only=True)
# 绘制热力图
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": .8})
# 显示图形
plt.show()
# 找出最强相关性
corrmat_unstacked = corrmat.unstack()
corrmat_unstacked = corrmat_unstacked[corrmat_unstacked < 1]
sorted_corr = corrmat_unstacked.sort_values(ascending=False)
top5_corr = sorted_corr.head(5)
print("Top 5 strongest correlations:")
print(top5_corr)

#3.4同心圆
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:, 6:8] = df.iloc[:, 6:8].apply(LabelEncoder().fit_transform)
sns.set_theme(style="ticks")
g = sns.jointplot(
    data=df,
    x="popularity", y="danceability", hue="artist_top_genre",
    kind="kde",
)

#3.5箱型图
# 绘制箱型图并识别异常值
plt.figure(figsize=(20, 20), dpi=200)
# 绘制多个箱型图
plt.subplot(4, 3, 1)
sns.boxplot(x='popularity', data=df)
plt.title('Popularity')
# ...（省略其他箱型图的绘制代码）
plt.tight_layout()
plt.show()

#3.6剔除异常值后的箱型图
# 剔除异常值
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for column in numeric_cols:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
# 绘制剔除异常值后的箱型图
plt.figure(figsize=(20, 20), dpi=200)
# 绘制多个箱型图
plt.subplot(4, 3, 1)
sns.boxplot(x='popularity', data=df)
plt.title('Popularity')
# ...（省略其他箱型图的绘制代码）
plt.tight_layout()
plt.show()

#4. 模型训练
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = df.loc[:,('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
y = df['artist_top_genre']
X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
y = le.transform(y)

from sklearn.cluster import KMeans
nclusters = 3
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X)
y_cluster_kmeans = km.predict(X)

from sklearn import metrics
score = metrics.silhouette_score(X, y_cluster_kmeans)
print("score")

#5.模型调参
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# 绘制WCSS值随簇数变化的折线图
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
plt.title('Elbow')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

