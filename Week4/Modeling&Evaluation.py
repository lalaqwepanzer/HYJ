# ====================
# 建模与评估
# Modeling & Evaluation
# ====================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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
