# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import xgboost as xgb
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit  # 时间序列分割[1,8](@ref)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def preprocess_pumpkin_data(filepath):
    """南瓜数据全样本预处理（与原始版本保持一致）"""
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

    # 类别特征编码（XGBoost需要手动处理类别特征）[7](@ref)
    for col in ['Variety', 'City', 'Package']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def xgboost_full_sample_training(df, target='Price'):
    """XGBoost全样本训练函数"""
    # 增强特征工程（添加时间序列特征）[1,8](@ref)
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['DayMonth_interact'] = df['DayOfYear'] * df['Month']
    df['Price_lag7'] = df['Price'].shift(7).fillna(method='bfill')  # 7天滞后特征
    df['Price_rolling_mean7'] = df['Price'].rolling(7).mean().fillna(method='bfill')

    # 准备数据（全样本）
    features = ['DayOfYear', 'Month', 'Year', 'Variety', 'City', 'Package',
                'DayMonth_interact', 'Price_lag7', 'Price_rolling_mean7']
    X = df[features]
    y = df[target]

    # 转换为DMatrix格式（XGBoost专用数据结构）[2](@ref)
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

    # XGBoost参数配置（基于金融数据特性优化）[1,6](@ref)
    params = {
        'objective': 'reg:squarederror',  # 回归任务[2](@ref)
        'eval_metric': ['rmse', 'mape'],  # 多评估指标
        'booster': 'gbtree',  # 使用树模型
        'eta': 0.05,  # 更低的学习率（类似LightGBM的0.01）
        'max_depth': 8,  # 限制树深度防止过拟合[6](@ref)
        'subsample': 0.8,  # 行采样比例
        'colsample_bytree': 0.8,  # 列采样比例
        'lambda': 0.5,  # L2正则化[6](@ref)
        'alpha': 0.2,  # L1正则化
        'tree_method': 'hist',  # 直方图算法加速（类似LightGBM）[7](@ref)
        'seed': 42,
        'verbosity': 0
    }

    # 训练配置（使用时间序列交叉验证）[8](@ref)
    tscv = TimeSeriesSplit(n_splits=5)
    evals_result = {}  # 存储评估结果

    # 全样本训练（带早停策略）[2](@ref)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=100
    )

    # 全样本评估
    y_pred = model.predict(dtrain)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'R2': r2_score(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred) * 100
    }

    # 特征重要性可视化（三种类型）[2](@ref)
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))
    importance_types = ['weight', 'gain', 'cover']
    for i, imp_type in enumerate(importance_types):
        xgb.plot_importance(model,
                            importance_type=imp_type,
                            ax=ax[i],
                            title=f'特征重要性 ({imp_type})',
                            max_num_features=15)
    plt.tight_layout()
    plt.savefig('XGBoost_feature_importance.png')
    plt.close()

    # 训练过程指标可视化[2](@ref)
    plt.figure(figsize=(12, 6))
    for metric in params['eval_metric']:
        plt.plot(evals_result['train'][metric], label=f'train {metric}')
    plt.legend()
    plt.title('XGBoost训练过程指标变化')
    plt.xlabel('迭代轮数')
    plt.ylabel('指标值')
    plt.savefig('XGBoost_training_metrics.png')
    plt.close()

    return model, metrics


def save_results(model, metrics):
    """保存XGBoost模型和结果为JSON格式"""
    # 获取模型参数和特征重要性[2](@ref)
    model_config = {
        'best_iteration': model.best_iteration,
        'best_score': model.best_score,
        'feature_names': model.feature_names
    }

    # 获取特征重要性（按gain排序）
    importance = model.get_score(importance_type='gain')

    # 合并结果
    results = {
        'model_config': model_config,
        'evaluation_metrics': metrics,
        'feature_importance': importance
    }

    # 保存JSON文件
    with open('XGBoost_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 额外保存原生模型文件（二进制）
    model.save_model('XGBoost_model.json')


if __name__ == "__main__":
    try:
        # 数据加载与预处理
        pumpkin_data = preprocess_pumpkin_data('US-pumpkins.csv')
        print(f"全样本数据加载成功，共{pumpkin_data.shape[0]}条记录")

        # XGBoost全样本训练
        model, metrics = xgboost_full_sample_training(pumpkin_data)

        # 输出评估结果
        print("\n=== XGBoost全样本训练结果 ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # 保存结果
        save_results(model, metrics)
        print("\n模型和评估结果已保存为：")
        print("XGBoost_model_results.json")

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
        model, metrics = xgboost_full_sample_training(pumpkin_data)
        save_results(model, metrics)