# 房屋面积与价格
#python代码
```
import matplotlib.pyplot as plt
import random
import matplotlib
#数据调递增的一次函数
x = [150,200,250,300,350,400,600]
y = [6450,7450,8450,9450,11450,15450,18450]
#步长
alpha = 0.00001
#计算样本个数
m = len(x)
#初始化参数的值，拟合函数为 y=theta0+theta1*x
ptheta0 = 0
ptheta1 = 0
stheta1=0
stheta0=0
#误差
error0=0
error1=0
#退出迭代的两次误差差值的阈值
epsilon=0.000001
def p(x):
    return ptheta1*x+ptheta0
def s(x):
    return stheta1*x+stheta0
#开始迭代批量梯度
presult0 = []
presult1 = []
while True:
    diff = [0, 0]
    # 梯度下降
    for i in range(m):
        diff[0] += p(x[i]) - y[i]  # 对theta0求导
        diff[1] += (p(x[i]) - y[i]) * x[i]  # 对theta1求导
    ptheta0 = ptheta0 - alpha / m * diff[0]
    ptheta1 = ptheta1 - alpha / m * diff[1]
    presult0.append(ptheta0)
    presult1.append(ptheta1)
    error1 = 0
    # 计算两次迭代的误差的差值，小于阈值则退出迭代，输出拟合结果
    for i in range(len(x)):
        error1 += (y[i] - (ptheta0 + ptheta1 * x[i])) ** 2 / 2
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1

#开始迭代随机梯度
sresult0 = []
sresult1 = []
for j in range(5000):
    diff = [0, 0]
    # 梯度下降
    i = random.randint(0, m - 1)
    diff[0] += s(x[i]) - y[i]  # 对theta0求导
    diff[1] += (s(x[i]) - y[i]) * x[i]  # 对theta1求导
    stheta0 = stheta0 - alpha / m * diff[0]
    stheta1 = stheta1 - alpha / m * diff[1]
    sresult0.append(stheta0)
    sresult1.append(stheta1)
    error1 = 0
    # 计算两次迭代的误差的差值，小于阈值则退出迭代，输出拟合结果
    for k in range(len(x)):
        error1 += (y[i] - (stheta0 + stheta1 * x[i])) ** 2 / 2
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
#结果
print(ptheta1,ptheta0)
print(stheta1,stheta0)
#画图
a=len(presult0)
C=len(presult1)
b=range(a)
c=range(C)
plt.plot(b,presult0)
plt.xlabel("Runs")
plt.ylabel("theta0")
plt.show()
plt.plot(c,presult1)
plt.xlabel("Runs")
plt.ylabel("theta1")
plt.show()
a1=len(sresult0)
C1=len(sresult1)
b1=range(a1)
c1=range(C1)
plt.plot(b1,sresult0)
plt.xlabel("Runs")
plt.ylabel("theta0")
plt.show()
plt.plot(c1,sresult1)
plt.xlabel("运行次数")
plt.ylabel("theta1")
plt.show()
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(x,[p(x) for x in x],label='批量梯度')
plt.plot(x,[s(x) for x in x],label='随机梯度')
plt.plot(x,y,'bo',label='数据')
plt.legend()
plt.show()
```
###批量梯度                       
theta1=28.778604659732547
theta0=1771.0428917695647
###theta0和theta1随运行次数的变化
![theta0](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot.png)
![theta1](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot1.png)
###随机梯度
theta1=32.79757157029413
theta0=2.1026714437678806
###theta0和theta1随运行次数的变化
![theta0](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot2.png)
![theta1](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot3.png)

###结果图
![结果](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot4.png)



#工资与阅历
```
import matplotlib.pyplot as plt
import random
import matplotlib
#数据调递增的一次函数
x=[1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5]
y=[39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872]
#步长
alpha = 0.00001
#计算样本个数
m = len(x)
#初始化参数的值，拟合函数为 y=theta0+theta1*x
ptheta0 = 0
ptheta1 = 0
stheta1=0
stheta0=0
#误差
error0=0
error1=0
#退出迭代的两次误差差值的阈值
epsilon=0.000001
def p(x):
    return ptheta1*x+ptheta0
def s(x):
    return stheta1*x+stheta0
#开始迭代批量梯度
presult0 = []
presult1 = []
while True:
    diff = [0, 0]
    # 梯度下降
    for i in range(m):
        diff[0] += p(x[i]) - y[i]  # 对theta0求导
        diff[1] += (p(x[i]) - y[i]) * x[i]  # 对theta1求导
    ptheta0 = ptheta0 - alpha / m * diff[0]
    ptheta1 = ptheta1 - alpha / m * diff[1]
    presult0.append(ptheta0)
    presult1.append(ptheta1)
    error1 = 0
    # 计算两次迭代的误差的差值，小于阈值则退出迭代，输出拟合结果
    for i in range(len(x)):
        error1 += (y[i] - (ptheta0 + ptheta1 * x[i])) ** 2 / 2
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1

#开始迭代随机梯度
sresult0 = []
sresult1 = []
for i in range(1000000):
    diff = [0, 0]
    # 梯度下降
    i = random.randint(0, m-1)
    diff[0] += s(x[i]) - y[i]  # 对theta0求导
    diff[1] += (s(x[i]) - y[i]) * x[i]  # 对theta1求导
    stheta0 = stheta0 - alpha / m * diff[0]
    stheta1 = stheta1 - alpha / m * diff[1]
    sresult0.append(stheta0)
    sresult1.append(stheta1)
    error1 = 0
    # 计算两次迭代的误差的差值，小于阈值则退出迭代，输出拟合结果
    for i in range(len(x)):
        error1 += (y[i] - (stheta0 + stheta1 * x[i])) ** 2 / 2
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
print(ptheta1,ptheta0)
print(stheta1,stheta0)
a=len(presult0)
C=len(presult1)
b=range(a)
c=range(C)
plt.plot(b,presult0)
plt.xlabel("Runs")
plt.ylabel("theta0")
plt.show()
plt.plot(c,presult1)
plt.xlabel("Runs")
plt.ylabel("theta1")
plt.show()
a1=len(sresult0)
C1=len(sresult1)
b1=range(a1)
c1=range(C1)
plt.plot(b1,sresult0)
plt.xlabel("Runs")
plt.ylabel("theta0")
plt.show()
plt.plot(c1,sresult1)
plt.xlabel("Runs")
plt.ylabel("theta1")
plt.show()
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(x,[p(x) for x in x],label='批量梯度')
plt.plot(x,[s(x) for x in x],label='随机梯度')
plt.plot(x,y,'bo',label='数据')
plt.legend()
plt.show()

```
###批量梯度                       
theta1=9450.01658025328
theta0=25791.834563028777
###theta0和theta1随运行次数的变化
![theta0](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot5.png)
![theta1](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot6.png)
###随机梯度
theta1=12752.3407772289
theta0=3549.5689256100904
###theta0和theta1随运行次数的变化
![theta0](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot7.png)
![theta1](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot8.png)

###结果图
![结果](https://raw.githubusercontent.com/DL-Metaphysics/DL-yizhixiaol/master/myplot9.png)





    

    

