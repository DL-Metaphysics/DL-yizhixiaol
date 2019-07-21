# MF


```
import numpy as np
from math import pow
import matplotlib.pyplot as plt
R=np.array([[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4]])
print("原始的评分矩阵R为：")
print(R)
alpha=0.0002#学习率
beta=0.02
N=len(R)
M=len(R[0])
K=2
P=np.random.rand(N,K)#生成N行K列的矩阵
Q=np.random.rand(K,M)#生成K行M列的矩阵
result=[]
for i in range(5000):#运行5000次
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j]>0:
                eij=R[i][j]-np.dot(P[i,:],Q[:,j])
                for k in range(K):
                    P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
                    Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])
    eR=np.dot(P,Q)#填充后的矩阵

    e=0#误差
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j]>0:
                e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
    for k in range(K):
        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
    result.append(e)
    if e<0.001:
        break
print("经过填充后的矩阵eR:")
print(eR)
n = len(result)
x = range(n)
print(x)
plt.plot(x, result, color='r', linewidth=3)
plt.title("Convergence curve")
plt.xlabel("generation")
plt.ylabel("loss")
plt.show()
```
**结果**
原始的评分矩阵R为：
[[5 3 0 1]
 [4 0 0 1]
 [1 1 0 5]
 [1 0 0 4]
 [0 1 5 4]]
经过填充后的矩阵eR:
[[4.99624873 2.93544304 4.48473294 0.99914131]
 [3.96256443 2.3373952  3.7417222  0.99661392]
 [1.06212841 0.83976764 5.25422718 4.96327875]
 [0.96875908 0.74074439 4.28993813 3.97199539]
 [1.76812146 1.20587323 4.91735138 4.03231098]]
 **结果图**
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190721232020144.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzOTA0MzA5,size_16,color_FFFFFF,t_70)
 
