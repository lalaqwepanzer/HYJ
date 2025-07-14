# 数据预处理
# Data Preprocessing
# ====================
import pandas as pd

def preprocess_data(df):
    """数据清洗与特征工程"""
    # 筛选bushel计量单位的数据
    df = df[df['Package'].str.contains('bushel', case=True, regex=True)]

    # 计算平均价格和月份
    price = (df['Low Price'] + df['High Price']) / 2
    month = pd.DatetimeIndex(df['Date']).month

    # 创建新数据集
    new_df = pd.DataFrame({
        'Month': month,
        'Package': df['Package'],
        'Variety': df['Variety'],
        'LowPrice': df['Low Price'],
        'HighPrice': df['High Price'],
        'Price': price,
        'DayOfYear': pd.to_datetime(df['Date']).dt.dayofyear
    })

    # 单位标准化处理
    new_df.loc[new_df['Package'].str.contains('1 1/9'), 'Price'] = price / (1 + 1 / 9)
    new_df.loc[new_df['Package'].str.contains('1/2'), 'Price'] = price / (1 / 2)

    return new_df