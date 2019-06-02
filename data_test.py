import pandas as pd
csv_data = pd.read_csv('G:/R/heart.csv',)  # 读取训练数据
print(csv_data)
print(csv_data.shape)
data_y=csv_data['V']
print(data_y)
