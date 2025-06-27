import matplotlib.pyplot as plt
import pandas as pd
 
df = pd.read_csv("nigerian-songs.csv")#以pandas库的read_csv函数读取csv文件
print(df.head(5))#查看前5行数据

