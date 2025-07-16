# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import lightgbm as lgb
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def preprocess_pumpkin_data(filepath):
    """南瓜数据全样本预处理"""
    pumpkins = pd.read_csv(filepath)

    # 计算价格和日期特征
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    date_col = pd.to_datetime(pumpkins['Date'])

    # 创建全样本DataFrame
    df = pd.DataFrame({
        'Date': date_col,
        'Month': date_col.dt.month,
        'DayOfYear': date_col.dt.dayofyear,
        'Package': pumpkins['Package'],
        'Variety': pumpkins['Variety'],
        'City': pumpkins['City Name'],
        'Price': price
    })

    # 包装单位标准化
    df.loc[df['Package'].str.contains('1 1/9'), 'Price'] = price / (1 + 1 / 9)
    df.loc[df['Package'].str.contains('1/2'), 'Price'] = price / 0.5

    # 类别特征编码
    for col in ['Variety', 'City', 'Package']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def full_sample_training(df, target='Price'):
    """全样本训练函数"""
    # 特征工程
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['DayMonth_interact'] = df['DayOfYear'] * df['Month']

    # 准备数据（全样本）
    features = ['DayOfYear', 'Month', 'Year', 'Variety', 'City', 'Package', 'DayMonth_interact']
    X = df[features]
    y = df[target]

    # LightGBM参数配置（基于全样本优化）
    params = {
        'objective': 'regression',
        'metric': ['rmse', 'mape'],
        'boosting_type': 'gbdt',
        'num_leaves': 127,  # 增大叶子数量适应全样本[1,3](@ref)
        'learning_rate': 0.01,  # 更小的学习率
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'min_child_samples': 50,  # 更大的最小样本数
        'max_depth': -1,  # 不限制深度
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbosity': -1,
        'seed': 42,
        'extra_trees': True  # 增加随机性[3](@ref)
    }

    # 创建数据集（禁用早停）
    train_data = lgb.Dataset(X, label=y, free_raw_data=False)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(100)]
    )

    # 全样本评估（训练误差）
    y_pred = model.predict(X)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'R2': r2_score(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred) * 100
    }

    # 特征重要性可视化
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(model, importance_type='gain', max_num_features=15)
    plt.title('全样本特征重要性（按增益排序）')
    plt.tight_layout()
    plt.savefig('LGBM_feature_importance.png')
    plt.close()

    return model, metrics


def save_results(model, metrics):
    """保存模型和结果为JSON格式"""
    # 模型转JSON
    model_json = model.dump_model()

    # 添加评估指标
    results = {
        'model_params': model_json,
        'evaluation_metrics': metrics,
        'feature_importance': dict(zip(
            model.feature_name(),
            map(int, model.feature_importance(importance_type='gain')))
        )
    }

    # 保存JSON文件
    with open('LGBM_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        # 数据加载与预处理
        pumpkin_data = preprocess_pumpkin_data('US-pumpkins.csv')
        print(f"全样本数据加载成功，共{pumpkin_data.shape[0]}条记录")

        # 全样本训练
        model, metrics = full_sample_training(pumpkin_data)

        # 输出评估结果
        print("\n=== 全样本训练结果 ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # 保存结果
        save_results(model, metrics)
        print("\n模型和评估结果已保存为：")
        print("LGBM_model_results.json")


    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        # 模拟数据测试
        print("使用模拟数据示例...")
        pumpkin_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=1000),
            'DayOfYear': np.random.randint(1, 366, 1000),
            'Month': np.random.randint(1, 13, 1000),
            'Variety': np.random.choice([0, 1], 1000),
            'City': np.random.choice([0, 1], 1000),
            'Package': np.random.choice([0, 1], 1000),
            'Price': np.random.uniform(5, 50, 1000)
        })
        model, metrics = full_sample_training(pumpkin_data)
        save_results(model, metrics)