import numpy as np
import pandas as pd
import datetime

cols = ['id','cust_name','sale_date','prod_key','quantity']
df =  pd.read_csv('sales.txt', delimiter='\t',names = cols,encoding='utf-8')

#If you want to do it quarterly
#df['sale_date'] = df['sale_date'].map(lambda x: (int(x[3:5]) // 4) + 1)

#If you want to use time as a weight penalty
time_split = 7
#time_stamp = '01/01/2022'
time_stamp = datetime.date(2022, 1, 1)
#time interval normalization
df['sale_date'] = df['sale_date'].map(lambda x: datetime.date(int(x[6:]), int(x[3:5]), int(x[:2])))
df['sale_date_normalization'] = df['sale_date'].map(lambda x: ((time_stamp - x).days) / time_split)

#Slice the data if necessary
grass_cut = list(range(0,2000,100))
cut_list = pd.cut(df['quantity'].values, grass_cut, right=True)
df = df[df['quantity'] > 0]
df = df[df['quantity'] <= 200]

df['quantity_normalization'] = df['quantity'] * (1 / (1 + alpha * df['sale_date_normalization']))
#.to_csv('grass_section.txt',encoding='utf-8')

#df['quantity'] = df['quantity'].map(lambda x: (int(x) // 5) + 1) 
print(len(df['quantity'].values))
print(df['quantity_normalization'].mean())
df.to_csv('grass_cut.txt',encoding='utf-8')

