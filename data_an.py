import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')
cols = ['id','cust_name','sale_date','prod_key','quantity']
df =  pd.read_csv('sales_dell.txt', delimiter='\t',names = cols,encoding='utf8')
#train_vec = pd.read_csv('ua.base', delimiter='\t', names = cols)
#print(df['sale_date'])
df['sale_date'] = df['sale_date'].map(lambda x: (int(x[3]) // 4) + 1)

num = df['quantity']

print(num.max())#3000
print(num.min())#-16

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
range_list = list(range(-400,2000,100))
cut_list = pd.cut(num.values, range_list, right=False)

plt.figure(figsize=(15,10))
sns.distplot(num, range_list,kde=False)
plt.grid()
plt.ylim(0, 100) 
plt.show()

#print(df['quantity'].groupby(cut_list).count()/len(num))
