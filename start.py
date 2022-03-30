from hmac import trans_5C
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import csr
import time
import random

#Data type:Line OrderID, Tag UID, ItemID,so on, Estimator Rank or Binary
#SVG

#Rank problem
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit_lost(y, y_hat):
    return np.log(1 + np.exp(-y * y_hat))
def mes_lost(y,y_hat):
    return 0.5 * np.sum((y_hat - y) ** 2)

def partial_derivative(y,y_hat):
    #when linear
    return y_hat - y
    #when binary
    #return sigmoid (-y * y_hat) * -y

def FM_Main(X_I,w0,W_Mat,Vec):
    return w0 + X_I.dot(W_Mat) + 0.5 * np.sum((X_I.dot(Vec)) ** 2 - (X_I ** 2).dot(Vec ** 2))
    
def FM_SGD(X,Y,k=1,alpha=0.01,iter=50):
    #init
    m,n = np.shape(X)
    #print(m,n)
    w0 = 0
    W = np.zeros((n,1))
    params_list = []
    V = np.random.normal(loc=0, scale=1, size=(n,k))

    for it in range(iter):
        total_lost = 0
        #SGD, read one simple by one step
        for i in range(m):
            #print(X[i])
            V[V > 255] = 255
            V[V < -255] = -255
            y_hat = FM_Main(X[i], w0, W, V)
            #print(y_hat)
            #When linear
            total_lost += mes_lost(Y[i], y_hat)
            #When Binary
            #total_lost += logit_lost(Y[i], y_hat)
            pdlost = partial_derivative(Y[i], y_hat)
            #in FM model, overall pd = pd * 1
            w0 = w0 -alpha * pdlost
            for j in range(n):
                if X[i,j] != 0:
                    pdlost_Mat = pdlost * X[i,j]
                    W[j] = W[j] - alpha * pdlost_Mat
                    for d in range(k):
                        pdlost_Vec = pdlost * (X[i,j] * (X[i].dot(V[:,d])) - V[j,d] * X[i,j] ** 2)
                        V[j,d] = V[j,d] - alpha * pdlost_Vec
                        V[np.isnan(V)] = 0
            #print(W)
        time_end=time.time()
        print('totally cost',time_end-time_start)
        print('Epoch is {}, and the total lost is {}'.format(it+1, total_lost))
        params_list.append([w0,W,V])
    return params_list

def FM_Pred(X, w0, W, V):
    res_list = []
    for i in range(X.shape[0]):
        y_hat = FM_Main(X[i], w0, W, V)
        res_list.append(np.round(y_hat))
        #res_list.append(-1 if sigmoid(y_hat) < 0.5 else 1)
    #print(res_list)
    return np.array(res_list)

#Data process part
#simple [[1,2],[1,3],[2,1],[2,3]], Users 2, items 3 
#wile be a 4*(2+3)
def vec_transform(dic,y=None,g=3):
    #n = len(train.index)
    #{'users': array([  1,   1,   1, ..., 943, 943, 943], dtype=int64),
    #  'items': array([   1,    2,    3, ..., 1188, 1228, 1330], dtype=int64)}
    ny = 0
    ng = []
    i = 0

    for vue in dic.values():
        ng.append(list(set(vue)))
        #print(list(set(vue)).sort())
    n = len(vue)
    for i in range(g):
        ny += len(ng[i])
    #print(ng)
    #print(n,ny)
    x_mat = np.zeros((n,ny+1))
    #x_mat = np.zeros((n,ny))

    g_count = 0
    g_count1 = 0
    for v in dic.values():
        for i in range(len(v)):
            x_mat[i][ng[g_count1].index(v[i]) + g_count] += 1
            #print(int(v[i]) + g_count)
        g_count += len(set(v)) 
        g_count1 += 1
        #print(g_count)

    if y != None:
        for i in range(n):
            x_mat[i][-1] = y[i]

    return x_mat

def vec_transform_b(dic,y=None,g=2):
    mean_quantity = 2.93
    ny = 0
    ng = []
    n_u = len(set(dic['cust_name']))
    n_i = len(set(dic['prod_key']))
    i = 0

    for vue in dic.values():
        ng.append(list(set(vue)))
    n = len(vue)
    for i in range(g):
        ny += len(ng[i])

    print("U_count:",n_u)
    print("I_count:",n_i)
    x_mat = np.zeros((int(n*1.4),ny+2))

    g_count = 0
    g_count1 = 0
    for v in dic.values():
        for i in range(len(v)):
            x_mat[i][ng[g_count1].index(v[i]) + g_count] += 1
        g_count += len(set(v)) 
        g_count1 += 1

    
    for i in range(n):
        #x_mat[i][-1] = y[i]
        x_mat[i][-1] = 1
        x_mat[i][-2] = y[i]
    random_part = int(n*1.4) - i - 1
    random_count = 0
    while(random_count < random_part):
        temp_u = random.randint(1, n_u)
        temp_i = random.randint(1, n_i)
        x_mat[n+random_count][temp_u-1] = 1
        x_mat[n+random_count][n_u + temp_i - 1] = 1
        x_mat[n+random_count][-2] = mean_quantity
        random_count += 1
        
    return x_mat

def season_judge(str):
        #12/04/2021
        return (int(str[3]) // 4) + 1

if __name__ == '__main__':
    np.random.seed(123)
    time_start=time.time()

    cols = ['index','id','cust_name','sale_date','prod_key','quantity']
    df = pd.read_csv('sales.txt', delimiter=',',nrows = 20000,names = cols,encoding='utf-8')

    y_train_overview = df['quantity'].values
    #train_mat = vec_transform({'cust_name':df['cust_name'].values,'sale_date':df['sale_date'],'prod_key':df['prod_key'].values}, y_train_overview,g=3)
    #function vec_transform_b() is used for transfor list to data used for dichotomous judgment, and add obfuscated data
    train_mat = vec_transform_b({'cust_name':df['cust_name'].values,'prod_key':df['prod_key'].values},y_train_overview,k=2)

    X_train, X_test, Y_train, Y_test = train_test_split(train_mat[:, :-1],train_mat[:, -1], test_size = 0.3, random_state = 123)

    print(X_train.shape)

    all_FM_params = FM_SGD(X=X_train, Y=Y_train, k=3,alpha=0.003, iter=100)
    #set k = 8 is best
    w0, W, V = all_FM_params[-1] 
    predicts = FM_Pred(X=X_test, w0=w0, W=W, V=V)
    np.savetxt('w0.csv',w0,delimiter=',')
    np.savetxt('W.csv',W,delimiter=',')
    np.savetxt('V.csv',V,delimiter=',')
    np.savetxt('res.csv',np.c_[X_test,predicts,Y_test],delimiter=',')
    print('Correct ratio: {:.2%}'.format(accuracy_score(Y_test, predicts)))
    


              