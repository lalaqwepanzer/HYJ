import matplotlib.pyplot as plt
import pandas as pd
 
df = pd.read_csv("nigerian-songs.csv")#以pandas库的read_csv函数读取csv文件
print(df.head(5))#查看前5行数据

print(df.info())

print(df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

top = df['artist_top_genre'].value_counts()  # 对不同音乐家类型进行统计汇总
plt.figure(figsize=(10, 7))  # 设置图表大小
barplot = sns.barplot(x=top[:5].index, y=top[:5].values, hue=top[:5].index, palette="coolwarm", legend=False)
plt.title('Top genres', color='blue')  # 设置条形图标题内容及颜色

# 添加数值标签
for index, value in enumerate(top[:5].values):
    barplot.text(index, value + 0.05, str(value), ha='center')

plt.show()

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

# 计算相关性矩阵
corrmat = df.corr(numeric_only=True)

# 绘制热力图
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": .8})

# 显示图形
plt.show()

# 找出最强相关性
# 将相关性矩阵转换为一维数据
corrmat_unstacked = corrmat.unstack()

# 排除自身相关性（值为1的情况）
corrmat_unstacked = corrmat_unstacked[corrmat_unstacked < 1]

# 按相关性值排序
sorted_corr = corrmat_unstacked.sort_values(ascending=False)

# 找出最强相关性的前5对变量
top5_corr = sorted_corr.head(5)

# 打印最强相关性的前5对变量
print("Top 5 strongest correlations:")
print(top5_corr)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df.iloc[:, 6:8] = df.iloc[:, 6:8].apply(LabelEncoder().fit_transform)

sns.set_theme(style="ticks")
g = sns.jointplot(
    data=df,
    x="popularity", y="danceability", hue="artist_top_genre",
    kind="kde",
)

# 绘制箱型图并识别异常值
plt.figure(figsize=(20, 20), dpi=200)

plt.subplot(4, 3, 1)
sns.boxplot(x='popularity', data=df)
plt.title('Popularity')

plt.subplot(4, 3, 2)
sns.boxplot(x='acousticness', data=df)
plt.title('Acousticness')

plt.subplot(4, 3, 3)
sns.boxplot(x='energy', data=df)
plt.title('Energy')

plt.subplot(4, 3, 4)
sns.boxplot(x='instrumentalness', data=df)
plt.title('Instrumentalness')

plt.subplot(4, 3, 5)
sns.boxplot(x='liveness', data=df)
plt.title('Liveness')

plt.subplot(4, 3, 6)
sns.boxplot(x='loudness', data=df)
plt.title('Loudness')

plt.subplot(4, 3, 7)
sns.boxplot(x='speechiness', data=df)
plt.title('Speechiness')

plt.subplot(4, 3, 8)
sns.boxplot(x='tempo', data=df)
plt.title('Tempo')

plt.subplot(4, 3, 9)
sns.boxplot(x='time_signature', data=df)
plt.title('Time Signature')

plt.subplot(4, 3, 10)
sns.boxplot(x='danceability', data=df)
plt.title('Danceability')

plt.subplot(4, 3, 11)
sns.boxplot(x='length', data=df)
plt.title('Length')

plt.subplot(4, 3, 12)
sns.boxplot(x='release_date', data=df)
plt.title('Release Date')

plt.tight_layout()
plt.show()

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

plt.subplot(4, 3, 1)
sns.boxplot(x='popularity', data=df)
plt.title('Popularity')

plt.subplot(4, 3, 2)
sns.boxplot(x='acousticness', data=df)
plt.title('Acousticness')

plt.subplot(4, 3, 3)
sns.boxplot(x='energy', data=df)
plt.title('Energy')

plt.subplot(4, 3, 4)
sns.boxplot(x='instrumentalness', data=df)
plt.title('Instrumentalness')

plt.subplot(4, 3, 5)
sns.boxplot(x='liveness', data=df)
plt.title('Liveness')

plt.subplot(4, 3, 6)
sns.boxplot(x='loudness', data=df)
plt.title('Loudness')

plt.subplot(4, 3, 7)
sns.boxplot(x='speechiness', data=df)
plt.title('Speechiness')

plt.subplot(4, 3, 8)
sns.boxplot(x='tempo', data=df)
plt.title('Tempo')

plt.subplot(4, 3, 9)
sns.boxplot(x='time_signature', data=df)
plt.title('Time Signature')

plt.subplot(4, 3, 10)
sns.boxplot(x='danceability', data=df)
plt.title('Danceability')

plt.subplot(4, 3, 11)
sns.boxplot(x='length', data=df)
plt.title('Length')

plt.subplot(4, 3, 12)
sns.boxplot(x='release_date', data=df)
plt.title('Release Date')

plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  # 创建一个编码器

X = df.loc[:, ('artist_top_genre', 'popularity', 'danceability', 'acousticness', 'loudness',
               'energy')]  # loc为Selection by Label函数，即为按标签取数据；将需要的标签数据取出作为X训练特征样本

y = df['artist_top_genre']  # 将艺术家流派作为Y验证模型精度标签

X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])  # 对文本数据进行标签化为数值格式

y = le.transform(y)  # 对文本数据进行标签化为数值格式

from sklearn.cluster import KMeans

nclusters = 3  # 划分3中音乐类型
seed = 0  # 选择随机初始化种子

km = KMeans(n_clusters=nclusters,
            random_state=seed)  # 一个random_state对应一个质心随机初始化的随机数种子。如果不指定随机数种子，则 sklearn中的KMeans并不会只选择一个随机模式扔出结果
km.fit(X)  # 对Kmeans模型进行训练

# 使用训练好的模型进行预测

y_cluster_kmeans = km.predict(X)
print(y_cluster_kmeans)

from sklearn import metrics
score = metrics.silhouette_score(X, y_cluster_kmeans)# metrics.silhouette_score函数用于计算轮廓系数
print(score)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
plt.title('Elbow')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)#设置聚类的簇（类别）数量
kmeans.fit(X)#对模型进行训练
labels = kmeans.predict(X)#输出预测值
plt.scatter(df['popularity'],df['danceability'],c = labels)#以人气为X轴，可舞蹈性为Y轴，标签为类别绘制散点图
plt.xlabel('popularity')
plt.ylabel('danceability')
plt.show()

labels = kmeans.labels_  # 提取出模型中样本的预测值labels

correct_labels = sum(y == labels)  # 统计预测正确的值

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'.format(correct_labels / float(y.size)))








