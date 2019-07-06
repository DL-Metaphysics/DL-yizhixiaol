import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
raw_data_X=np.loadtxt('003-AI-KNN-datasets-Iris.txt',dtype=float,delimiter=',',usecols=(0,1,2,3))
raw_data_y=np.loadtxt('003-AI-KNN-datasets-Iris.txt',dtype=str,delimiter=',',usecols=(4))
arr = np.random.choice(int(len(raw_data_X)),size=30,replace=False)
X_train=np.delete(raw_data_X,arr,axis=0)
y_train=np.delete(raw_data_y,arr)
print(X_train)
x_test=[]
y_test=[]
for i in arr:
    x_test.append(raw_data_X[i])
    y_test.append(raw_data_y[i])
X_test=np.array(x_test)
print(X_test)
k=eval(input("请输入k的值："))
#杰卡斯
e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
        d=int(len(set(X_test[j])&set(x_train)))/int(len(set(X_test[j])|set(x_train)))
        distance.append(d)
        nearest = np.argsort(distance)
    b = 0
    c = 0
    a = 0
    ynearest=nearest[::-1]
    for l in ynearest[:k]:
        #print(l)y
        if y_train[l] == 'Iris-setosa':
            a = a + 1
        elif y_train[l] == 'Iris-versicolor':
            b = b + 1
        else:
            c = c + 1
    #print(a)
    #print(b)
    #print(c)
    #print(max(a,b,c))
    if a == max(a, b, c):
        #print('Iris-setosa')
        d = 'Iris-setosa'
    elif b == max(a, b, c):
        #print('Iris-versicolor')
        d = 'Iris-versicolor'
    else:
        #print('Iris-virginica')
        d = 'Iris-virginica'
    #print(d)
    if d == y_test[j]:
        e = e + 1
    #print(y_test[j])
print("杰卡德")
print(e / len(y_test))
#皮尔森
e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
        d=np.sum((X_test[j]-np.sum(X_test[j])/int(len(X_test[j])))*(x_train-np.sum(x_train)/int(len(x_train))))/sqrt((np.sum((X_test[j]-np.sum(X_test[j])/int(len(X_test[j])))**2))*(np.sum((x_train-np.sum(x_train)/int(len(x_train)))**2)))
        distance.append(d)
        nearest = np.argsort(distance)
    b = 0
    c = 0
    a = 0
    ynearest=nearest[::-1]
    for l in ynearest[:k]:
        #print(l)
        if y_train[l] == 'Iris-setosa':
            a = a + 1
        elif y_train[l] == 'Iris-versicolor':
            b = b + 1
        else:
            c = c + 1
    #print(a)
    #print(b)
    #print(c)
    #print(max(a,b,c))
    if a == max(a, b, c):
        #print('Iris-setosa')
        d = 'Iris-setosa'
    elif b == max(a, b, c):
        #print('Iris-versicolor')
        d = 'Iris-versicolor'
    else:
        #print('Iris-virginica')
        d = 'Iris-virginica'
    #print(d)
    if d == y_test[j]:
        e = e + 1
    #print(y_test[j])
print("皮尔森")
print(e / len(y_test))

#余弦相似度
e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
        d=np.sum(x_train*X_test[j])/(sqrt(np.sum(x_train**2))*sqrt(np.sum(X_test**2)))
        distance.append(d)
        nearest = np.argsort(distance)
    b = 0
    c = 0
    a = 0
    ynearest=nearest[::-1]
    for l in ynearest[:k]:
        #print(l)
        if y_train[l] == 'Iris-setosa':
            a = a + 1
        elif y_train[l] == 'Iris-versicolor':
            b = b + 1
        else:
            c = c + 1
    #print(a)
    #print(b)
    #print(c)
    #print(max(a,b,c))
    if a == max(a, b, c):
        #print('Iris-setosa')
        d = 'Iris-setosa'
    elif b == max(a, b, c):
        #print('Iris-versicolor')
        d = 'Iris-versicolor'
    else:
        #print('Iris-virginica')
        d = 'Iris-virginica'
    #print(d)
    if d == y_test[j]:
        e = e + 1
    #print(y_test[j])
print("余弦相似度")
print(e / len(y_test))
#曼哈顿距离
e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
        d=np.sum(abs(x_train-X_test[j]))
        distance.append(d)
        nearest=np.argsort(distance)
    b = 0
    c = 0
    a = 0
    for l in nearest[:k]:
        # print(l)
        if y_train[l] == 'Iris-setosa':
            a = a + 1
        elif y_train[l] == 'Iris-versicolor':
            b = b + 1
        else:
            c = c + 1
    # print(a)
    # print(b)
    # print(c)
    # print(max(a,b,c))
    if a == max(a, b, c):
        # print('Iris-setosa')
        d = 'Iris-setosa'
    elif b == max(a, b, c):
        # print('Iris-versicolor')
        d = 'Iris-versicolor'
    else:
        # print('Iris-virginica')
        d = 'Iris-virginica'
    # print(d)
    if d == y_test[j]:
        e = e + 1
print("蛮哈顿距离")
print(e / len(y_test))

#欧式距离
e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
        d=sqrt(np.sum((x_train - X_test[j])**2))
        distance.append(d)
        nearest=np.argsort(distance)

    b=0
    c=0
    a=0
    for l in nearest[:k]:
        #print(l)
        if y_train[l]=='Iris-setosa':
            a=a+1
        elif y_train[l]=='Iris-versicolor':
            b=b+1
        else:
            c=c+1
    #print(a)
    #print(b)
    #print(c)
   # print(max(a,b,c))
    if a==max(a,b,c):
        #print('Iris-setosa')
        d='Iris-setosa'
    elif b==max(a,b,c):
        #print('Iris-versicolor')
        d='Iris-versicolor'
    else:
        #print('Iris-virginica')
        d='Iris-virginica'
    #print(d)
    if d==y_test[j]:
        e=e+1
print("欧式距离")
print(e/len(y_test))





