import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(filepath):
    """
    加载数据并进行预处理
    """
    # 加载数据
    pumpkins = pd.read_csv(filepath)

    # 筛选只包含"bushel"的数据
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]

    # 计算平均价格
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    month = pd.DatetimeIndex(pumpkins['Date']).month

    # 创建新DataFrame
    new_pumpkins = pd.DataFrame({
        'Month': month,
        'Package': pumpkins['Package'],
        'Variety': pumpkins['Variety'],
        'City': pumpkins['City Name'],
        'LowPrice': pumpkins['Low Price'],
        'HighPrice': pumpkins['High Price'],
        'Price': price,
        'DayOfYear': pd.to_datetime(pumpkins['Date']).dt.dayofyear
    })

    # 处理特殊包装情况
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price / (1 + 1 / 9)
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price / (1 / 2)

    return new_pumpkins


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


def simple_linear_regression(df, variety='PIE TYPE'):
    """
    简单线性回归分析
    """
    # 筛选特定品种
    pie_pumpkins = df[df['Variety'] == variety].copy()
    pie_pumpkins.dropna(inplace=True)

    # 准备数据
    X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1, 1)
    Y = pie_pumpkins['Price']

    # 划分训练测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # 训练模型
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)

    # 预测与评估
    pred = lin_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    r2 = r2_score(Y_test, pred)

    print(f'=== {variety}简单线性回归结果 ===')
    print(f'RMSE: {rmse:.3f} ({rmse / np.mean(pred) * 100:.1f}%)')
    print(f'R² Score: {r2:.3f}')

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, Y_test, color='blue', label='实际值')
    plt.plot(X_test, pred, color='red', linewidth=2, label='预测线')
    plt.xlabel('年度日期序号')
    plt.ylabel('价格（美元）')
    plt.title(f'{variety}南瓜价格线性回归预测')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return lin_reg


def polynomial_regression(df, degree=2):
    """
    多项式回归分析
    """
    # 准备数据 - 使用品种独热编码
    X = pd.get_dummies(df['Variety'], prefix='var')
    Y = df['Price']

    # 划分训练测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )

    # 构建多项式回归管道
    pipeline = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(fit_intercept=True)
    )

    # 训练模型
    pipeline.fit(X_train, Y_train)

    # 预测
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    # 评估
    rmse = np.sqrt(mean_squared_error(Y_test, test_pred))
    mae = np.mean(np.abs(Y_test - test_pred))
    r2 = r2_score(Y_test, test_pred)
    score = pipeline.score(X_train, Y_train)

    print('\n=== 多项式回归结果 ===')
    print(f'RMSE: {rmse:.3f} ({rmse / np.mean(test_pred) * 100:.1f}%)')
    print(f'MAE: {mae:.3f}')
    print(f'测试集R²: {r2:.3f}')
    print(f'训练集R²: {score:.3f}')

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.scatter(Y_test, test_pred, c='blue', alpha=0.6, label='测试集')
    plt.scatter(Y_train, train_pred, c='green', alpha=0.3, label='训练集')
    max_price = max(max(Y_test), max(Y_train))
    plt.plot([0, max_price], [0, max_price], 'r--', label='完美预测')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('南瓜品种价格预测效果')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 特征重要性分析
    if degree == 2:
        coef = pipeline.named_steps['linearregression'].coef_
        features = pipeline.named_steps['polynomialfeatures'].get_feature_names_out(input_features=X.columns)
        coef_df = pd.DataFrame({'特征': features, '系数': coef}).sort_values('系数', key=abs, ascending=False)
        print("\n重要特征系数分析:\n", coef_df.head(5))

    return pipeline


def multiple_linear_regression(df):
    """
    多元线性回归分析
    """
    # 准备数据 - 包含品种、月份、城市和包装类型
    X = (pd.get_dummies(df['Variety'])
         .join(df['Month'])
         .join(pd.get_dummies(df['City']))
         .join(pd.get_dummies(df['Package'])))
    Y = df['Price']

    # 划分训练测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )

    # 训练模型
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)

    # 预测与评估
    pred = lin_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    score = lin_reg.score(X_train, Y_train)

    print('\n=== 多元线性回归结果 ===')
    print(f'Mean error: {rmse:3.3} ({rmse / np.mean(pred) * 100:3.3}%)')
    print('Model determination: ', score)

    return lin_reg


def polynomial_multiple_regression(df, degree=2):
    """
    多元多项式回归分析
    """
    # 准备数据
    X = (pd.get_dummies(df['Variety'])
         .join(df['Month'])
         .join(pd.get_dummies(df['City']))
         .join(pd.get_dummies(df['Package'])))
    Y = df['Price']

    # 划分训练测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )

    # 构建管道
    pipeline = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # 训练模型
    pipeline.fit(X_train, Y_train)

    # 预测与评估
    pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    score = pipeline.score(X_train, Y_train)

    print('\n=== 多元多项式回归结果 ===')
    print(f'RMSE: {rmse:.3f} ({rmse / np.mean(pred) * 100:.1f}%)')
    print('R² Score:', score)

    return pipeline


# 主程序
if __name__ == "__main__":
    # 1. 加载和预处理数据
    pumpkin_data = load_and_preprocess_data('US-pumpkins.csv')

    # 2. 数据可视化
    visualize_data(pumpkin_data)

    # 3. 简单线性回归分析
    slr_model = simple_linear_regression(pumpkin_data)

    # 4. 多项式回归分析
    poly_model = polynomial_regression(pumpkin_data)

    # 5. 多元线性回归分析
    mlr_model = multiple_linear_regression(pumpkin_data)

    # 6. 多元多项式回归分析
    poly_mlr_model = polynomial_multiple_regression(pumpkin_data)