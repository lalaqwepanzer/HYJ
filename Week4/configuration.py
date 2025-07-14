# -*- coding: utf-8 -*-
"""
南瓜市场价格分析与预测模型
Pumpkin Market Price Analysis and Prediction Model
"""

# ====================
# 基础配置
# Basic Configuration
# ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 设置中文显示（Windows系统）
# Set Chinese font display (For Windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================
# 数据加载与探索
# Data Loading & Exploration
# ====================
def load_and_explore_data(filepath):
    """加载数据并进行初步探索"""
    pumpkins = pd.read_csv(filepath)

    # 数据概览
    print("=== 数据概览 ===")
    print(pumpkins.head())  # 前5行
    print(pumpkins.tail())  # 后5行
    print(pumpkins.info())  # 结构信息
    print("缺失值统计:\n", pumpkins.isnull().sum())

    return pumpkins