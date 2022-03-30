import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')
cols = ['id','cust_name','sale_date','prod_key','quantity']
df =  pd.read_csv('sales.txt', delimiter='\t',names = cols,encoding='utf8')
#train_vec = pd.read_csv('ua.base', delimiter='\t', names = cols)
#print(df['sale_date'])
#df['sale_date'] = df['sale_date'].map(lambda x: (int(x[3]) // 4) + 1)

num = df['quantity']

print(num.max())#3000
print(num.min())#-16

#Simple frequency statistics for data slicing
#pd.set_option('display.max_columns', None)
#print(df['quantity'].groupby(cut_list).count()/len(num))

pd.set_option('display.max_rows', None)
range_list = list(range(-400,2000,100))
cut_list = pd.cut(num.values, range_list, right=False)

plt.figure(figsize=(15,10))
sns.distplot(num, range_list,kde=False)
plt.grid()
plt.ylim(0, 100) 
plt.show()

#Average online days
online_day = df[['sale_date','prod_key']]
online_day = online_day.sort_values(by='prod_key')
prod_list = list(set(df['prod_key'].values))
online_cou = np.zeros(len(prod_list),dtype=int)
online_cou1 = np.zeros(len(prod_list),dtype=int)
coun = 0
for i in prod_list:
    online_cou[coun] = len(set(online_day[online_day['prod_key'] == i]['sale_date'].values))
    online_cou1[coun] = len(online_day[online_day['prod_key'] == i])
    coun += 1
#print(online_day[online_day['prod_key'] == i]['sale_date'].values)

#plt.plot(list(range(len(online_cou))),online_cou)
#plt.plot(list(range(len(online_cou1))),online_cou1)
#plt.plot(list(range(len(online_cou1))),np.sort(online_cou))
#plt.plot(list(range(len(online_cou1))),np.sort(online_cou1))


parameter = np.polyfit(np.sort(online_cou), np.sort(online_cou1), 1)
y = parameter[0] * np.sort(online_cou)
plt.plot(np.sort(online_cou),y)
plt.scatter(np.sort(online_cou),np.sort(online_cou1))
plt.show()
