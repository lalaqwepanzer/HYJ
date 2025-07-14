# 可视化分析
# Visualization
# ====================
import matplotlib as plt

def visualize_data(df):
    """
    数据可视化
    """
    # 价格与月份关系散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['DayOfYear'], df['Price'], color='orange', alpha=0.7)
    plt.title('南瓜价格与销售日期关系')
    plt.xlabel('年度日期序号')
    plt.ylabel('价格（美元）')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 按月分组的价格直方图
    ax = df.groupby(['Month'])['Price'].mean().plot(
        kind='bar',
        color='skyblue',
        edgecolor='black',
        alpha=0.7,
        figsize=(10, 6)
    )
    plt.title('南瓜价格月度分布')
    plt.xlabel('月份')
    plt.ylabel('平均价格（美元）')
    plt.xticks(rotation=0)

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=10
        )

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 按品种分色的散点图
    colors = ['red', 'blue', 'green', 'yellow']
    ax = None
    for i, var in enumerate(df['Variety'].unique()):
        temp_df = df[df['Variety'] == var]
        ax = temp_df.plot.scatter(
            x='DayOfYear',
            y='Price',
            ax=ax,
            c=colors[i],
            label=var,
            s=50,
            alpha=0.8
        )

    plt.title('销售日期-价格按不同南瓜种类分色散点图')
    plt.xlabel('DayOfYear')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), title='Variety')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
