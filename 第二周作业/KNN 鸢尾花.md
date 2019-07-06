# KNN 鸢尾花
[数据集及源代码][1]
```
import numpy as np
from math import sqrt
#对003-AI-KNN-datasets-Iris.txt数据进行处理
raw_data_X=np.loadtxt('003-AI-KNN-datasets-Iris.txt',dtype=float,delimiter=',',usecols=(0,1,2,3))
raw_data_y=np.loadtxt('003-AI-KNN-datasets-Iris.txt',dtype=str,delimiter=',',usecols=(4))
#把整个数据集以1：4的比例随机分为测试集和训练集
arr = np.random.choice(int(len(raw_data_X)),size=30,replace=False)
X_train=np.delete(raw_data_X,arr,axis=0)
y_train=np.delete(raw_data_y,arr)
print("训练集数据")
print(X_train)
x_test=[]
y_test=[]
for i in arr:
    x_test.append(raw_data_X[i])
    y_test.append(raw_data_y[i])
X_test=np.array(x_test)
print("测试集数据")
print(X_test)
k=9#k值可以自己设定
```       
##欧式距离，曼哈顿距离
```
def oushi(x_train,X_test,j):
    return d=sqrt(np.sum((x_train - X_test[j])**2))
def manhadun(x_train,X_test,j):
    return d=np.sum(abs(x_train-X_test[j]))
e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
        #oushi(x_train,X_test,j)欧式距离
        #manhadun(x_train,X_test,j)曼哈顿距离
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
    if a==max(a,b,c):
        print('第{}组，预测值：Iris-setosa，真实值：{}'.format(j+1,y_test[j]))
        d='Iris-setosa'
    elif b==max(a,b,c):
        print('第{}组，预测值：Iris-versicolor，真实值：{}'.format(j+1,y_test[j]))
        d='Iris-versicolor'
    else:
        print('第{}组，预测值：Iris-virginica，真实值：{}'.format(j + 1, y_test[j]))
        d='Iris-virginica'
    #print(d)
    if d==y_test[j]:
        e=e+1
print("准确率")
print(e/len(y_test))
```
###结果
###欧式距离
第1组，预测值：Iris-versicolor，真实值：Iris-versicolor
第2组，预测值：Iris-versicolor，真实值：Iris-versicolor
第3组，预测值：Iris-virginica，真实值：Iris-virginica
第4组，预测值：Iris-versicolor，真实值：Iris-versicolor
第5组，预测值：Iris-virginica，真实值：Iris-virginica
第6组，预测值：Iris-versicolor，真实值：Iris-versicolor
第7组，预测值：Iris-versicolor，真实值：Iris-versicolor
第8组，预测值：Iris-virginica，真实值：Iris-virginica
第9组，预测值：Iris-setosa，真实值：Iris-setosa
第10组，预测值：Iris-versicolor，真实值：Iris-versicolor
第11组，预测值：Iris-setosa，真实值：Iris-setosa
第12组，预测值：Iris-virginica，真实值：Iris-virginica
第13组，预测值：Iris-virginica，真实值：Iris-virginica
第14组，预测值：Iris-virginica，真实值：Iris-virginica
第15组，预测值：Iris-setosa，真实值：Iris-setosa
第16组，预测值：Iris-virginica，真实值：Iris-virginica
第17组，预测值：Iris-virginica，真实值：Iris-virginica
第18组，预测值：Iris-virginica，真实值：Iris-virginica
第19组，预测值：Iris-setosa，真实值：Iris-setosa
第20组，预测值：Iris-versicolor，真实值：Iris-versicolor
第21组，预测值：Iris-versicolor，真实值：Iris-versicolor
第22组，预测值：Iris-versicolor，真实值：Iris-versicolor
第23组，预测值：Iris-virginica，真实值：Iris-virginica
第24组，预测值：Iris-setosa，真实值：Iris-setosa
第25组，预测值：Iris-versicolor，真实值：Iris-versicolor
第26组，预测值：Iris-setosa，真实值：Iris-setosa
第27组，预测值：Iris-virginica，真实值：Iris-virginica
第28组，预测值：Iris-virginica，真实值：Iris-virginica
第29组，预测值：Iris-setosa，真实值：Iris-setosa
第30组，预测值：Iris-virginica，真实值：Iris-versicolor
准确率
0.9666666666666667
###曼哈顿距离
第1组，预测值：Iris-virginica，真实值：Iris-virginica
第2组，预测值：Iris-versicolor，真实值：Iris-versicolor
第3组，预测值：Iris-virginica，真实值：Iris-virginica
第4组，预测值：Iris-virginica，真实值：Iris-virginica
第5组，预测值：Iris-versicolor，真实值：Iris-versicolor
第6组，预测值：Iris-virginica，真实值：Iris-virginica
第7组，预测值：Iris-versicolor，真实值：Iris-versicolor
第8组，预测值：Iris-virginica，真实值：Iris-virginica
第9组，预测值：Iris-virginica，真实值：Iris-virginica
第10组，预测值：Iris-setosa，真实值：Iris-setosa
第11组，预测值：Iris-versicolor，真实值：Iris-versicolor
第12组，预测值：Iris-versicolor，真实值：Iris-versicolor
第13组，预测值：Iris-virginica，真实值：Iris-virginica
第14组，预测值：Iris-setosa，真实值：Iris-setosa
第15组，预测值：Iris-setosa，真实值：Iris-setosa
第16组，预测值：Iris-setosa，真实值：Iris-setosa
第17组，预测值：Iris-versicolor，真实值：Iris-versicolor
第18组，预测值：Iris-versicolor，真实值：Iris-virginica
第19组，预测值：Iris-versicolor，真实值：Iris-versicolor
第20组，预测值：Iris-setosa，真实值：Iris-setosa
第21组，预测值：Iris-virginica，真实值：Iris-virginica
第22组，预测值：Iris-virginica，真实值：Iris-virginica
第23组，预测值：Iris-setosa，真实值：Iris-setosa
第24组，预测值：Iris-setosa，真实值：Iris-setosa
第25组，预测值：Iris-virginica，真实值：Iris-versicolor
第26组，预测值：Iris-virginica，真实值：Iris-virginica
第27组，预测值：Iris-versicolor，真实值：Iris-versicolor
第28组，预测值：Iris-virginica，真实值：Iris-virginica
第29组，预测值：Iris-versicolor，真实值：Iris-versicolor
第30组，预测值：Iris-versicolor，真实值：Iris-versicolor
准确率
0.933333333333333
##皮尔森，余弦相似度，杰卡德
```

e=0
for j in range(len(X_test)):
    distance=[]
    for x_train in X_train:
     
       
        distance.append(d)
        nearest=np.argsort(distance)
    b=0
    c=0
    a=0
    ynearest=nearest[::-1]
    for l in ynearest[:k]:
        if y_train[l]=='Iris-setosa':
            a=a+1
        elif y_train[l]=='Iris-versicolor':
            b=b+1
        else:
            c=c+1
    if a==max(a,b,c):
        print('第{}组，预测值：Iris-setosa，真实值：{}'.format(j+1,y_test[j]))
        d='Iris-setosa'
    elif b==max(a,b,c):
        print('第{}组，预测值：Iris-versicolor，真实值：{}'.format(j+1,y_test[j]))
        d='Iris-versicolor'
    else:
        print('第{}组，预测值：Iris-virginica，真实值：{}'.format(j + 1, y_test[j]))
        d='Iris-virginica'
    #print(d)
    if d==y_test[j]:
        e=e+1
print("准确率")
print(e/len(y_test))
```
###结果
###余弦相似度
第1组，预测值：Iris-setosa，真实值：Iris-setosa
第2组，预测值：Iris-setosa，真实值：Iris-setosa
第3组，预测值：Iris-versicolor，真实值：Iris-versicolor
第4组，预测值：Iris-virginica，真实值：Iris-virginica
第5组，预测值：Iris-virginica，真实值：Iris-virginica
第6组，预测值：Iris-virginica，真实值：Iris-virginica
第7组，预测值：Iris-setosa，真实值：Iris-setosa
第8组，预测值：Iris-virginica，真实值：Iris-virginica
第9组，预测值：Iris-versicolor，真实值：Iris-versicolor
第10组，预测值：Iris-versicolor，真实值：Iris-virginica
第11组，预测值：Iris-versicolor，真实值：Iris-versicolor
第12组，预测值：Iris-versicolor，真实值：Iris-versicolor
第13组，预测值：Iris-virginica，真实值：Iris-virginica
第14组，预测值：Iris-virginica，真实值：Iris-virginica
第15组，预测值：Iris-versicolor，真实值：Iris-versicolor
第16组，预测值：Iris-versicolor，真实值：Iris-versicolor
第17组，预测值：Iris-virginica，真实值：Iris-virginica
第18组，预测值：Iris-virginica，真实值：Iris-virginica
第19组，预测值：Iris-setosa，真实值：Iris-setosa
第20组，预测值：Iris-versicolor，真实值：Iris-versicolor
第21组，预测值：Iris-virginica，真实值：Iris-virginica
第22组，预测值：Iris-setosa，真实值：Iris-setosa
第23组，预测值：Iris-virginica，真实值：Iris-virginica
第24组，预测值：Iris-virginica，真实值：Iris-virginica
第25组，预测值：Iris-setosa，真实值：Iris-setosa
第26组，预测值：Iris-versicolor，真实值：Iris-versicolor
第27组，预测值：Iris-versicolor，真实值：Iris-versicolor
第28组，预测值：Iris-versicolor，真实值：Iris-versicolor
第29组，预测值：Iris-virginica，真实值：Iris-virginica
第30组，预测值：Iris-versicolor，真实值：Iris-versicolor
准确率
0.9666666666666667
###皮尔森
第1组，预测值：Iris-setosa，真实值：Iris-setosa
第2组，预测值：Iris-versicolor，真实值：Iris-virginica
第3组，预测值：Iris-setosa，真实值：Iris-setosa
第4组，预测值：Iris-setosa，真实值：Iris-setosa
第5组，预测值：Iris-setosa，真实值：Iris-setosa
第6组，预测值：Iris-versicolor，真实值：Iris-versicolor
第7组，预测值：Iris-versicolor，真实值：Iris-versicolor
第8组，预测值：Iris-versicolor，真实值：Iris-versicolor
第9组，预测值：Iris-virginica，真实值：Iris-virginica
第10组，预测值：Iris-setosa，真实值：Iris-setosa
第11组，预测值：Iris-setosa，真实值：Iris-setosa
第12组，预测值：Iris-versicolor，真实值：Iris-versicolor
第13组，预测值：Iris-virginica，真实值：Iris-virginica
第14组，预测值：Iris-versicolor，真实值：Iris-versicolor
第15组，预测值：Iris-versicolor，真实值：Iris-versicolor
第16组，预测值：Iris-setosa，真实值：Iris-setosa
第17组，预测值：Iris-virginica，真实值：Iris-virginica
第18组，预测值：Iris-virginica，真实值：Iris-virginica
第19组，预测值：Iris-virginica，真实值：Iris-virginica
第20组，预测值：Iris-versicolor，真实值：Iris-versicolor
第21组，预测值：Iris-setosa，真实值：Iris-setosa
第22组，预测值：Iris-setosa，真实值：Iris-setosa
第23组，预测值：Iris-versicolor，真实值：Iris-versicolor
第24组，预测值：Iris-versicolor，真实值：Iris-versicolor
第25组，预测值：Iris-versicolor，真实值：Iris-versicolor
第26组，预测值：Iris-setosa，真实值：Iris-setosa
第27组，预测值：Iris-versicolor，真实值：Iris-versicolor
第28组，预测值：Iris-setosa，真实值：Iris-setosa
第29组，预测值：Iris-setosa，真实值：Iris-setosa
第30组，预测值：Iris-setosa，真实值：Iris-setosa
准确率
0.9666666666666667
###杰卡德
第1组，预测值：Iris-setosa，真实值：Iris-setosa
第2组，预测值：Iris-virginica，真实值：Iris-versicolor
第3组，预测值：Iris-versicolor，真实值：Iris-versicolor
第4组，预测值：Iris-virginica，真实值：Iris-virginica
第5组，预测值：Iris-versicolor，真实值：Iris-virginica
第6组，预测值：Iris-virginica，真实值：Iris-virginica
第7组，预测值：Iris-virginica，真实值：Iris-virginica
第8组，预测值：Iris-virginica，真实值：Iris-virginica
第9组，预测值：Iris-setosa，真实值：Iris-versicolor
第10组，预测值：Iris-setosa，真实值：Iris-setosa
第11组，预测值：Iris-setosa，真实值：Iris-setosa
第12组，预测值：Iris-versicolor，真实值：Iris-virginica
第13组，预测值：Iris-setosa，真实值：Iris-setosa
第14组，预测值：Iris-virginica，真实值：Iris-virginica
第15组，预测值：Iris-setosa，真实值：Iris-setosa
第16组，预测值：Iris-setosa，真实值：Iris-setosa
第17组，预测值：Iris-setosa，真实值：Iris-virginica
第18组，预测值：Iris-virginica，真实值：Iris-virginica
第19组，预测值：Iris-setosa，真实值：Iris-setosa
第20组，预测值：Iris-versicolor，真实值：Iris-virginica
第21组，预测值：Iris-setosa，真实值：Iris-setosa
第22组，预测值：Iris-virginica，真实值：Iris-virginica
第23组，预测值：Iris-virginica，真实值：Iris-virginica
第24组，预测值：Iris-setosa，真实值：Iris-setosa
第25组，预测值：Iris-setosa，真实值：Iris-setosa
第26组，预测值：Iris-virginica，真实值：Iris-virginica
第27组，预测值：Iris-versicolor，真实值：Iris-versicolor
第28组，预测值：Iris-setosa，真实值：Iris-setosa
第29组，预测值：Iris-setosa，真实值：Iris-setosa
第30组，预测值：Iris-versicolor，真实值：Iris-versicolor
准确率
0.8
#手写数字识别
```
from os import listdir
from numpy import *
import numpy as np
import operator
def KNN(test_data,train_data,train_label,k):
    # 已知分类的数据集（训练集）的行数
    dataSetSize=train_data.shape[0]
    # 求所有距离：先tile函数将输入点拓展成与训练集相同维数的矩阵，计算测试样本与每一个训练样本的距离
    all_distances=np.sqrt(np.sum(np.square(tile(test_data,(dataSetSize,1))-train_data),axis=1))
    # print("所有距离：",all_distances)
    # 按all_distances中元素进行升序排序后得到其对应索引的列表
    sort_distance_index=all_distances.argsort()
    #print("文件索引排序：",sort_distance_index)
    #选择距离最小的k个点
    classCount={}
    for i in range(k):
        # 返回最小距离的训练集的索引(预测值)
        voteIlabel=train_label[sort_distance_index[i]]
        # print('第',i+1,'次预测值',voteIlabel)
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # 求众数：按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount[0][0])
    return sortedClassCount[0][0]
#文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect=[]
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    #print(returnVect)
    return returnVect
#从文件名中解析分类数字
def classnumCut(fileName):
    fileStr=fileName.split('.')[0]
    classNumStr=int(fileStr.split('_')[0])
    return classNumStr
#构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    train_label=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    train_data= zeros((m,1024))
    # 获取训练集的标签
    for i in range(m):
        # fileNameStr:所有训练集文件名
        fileNameStr=trainingFileList[i]
        # 得到训练集索引
        train_label.append(classnumCut(fileNameStr))
        train_data[i, :] = img2vector('trainingDigits/%s' %fileNameStr)
    return train_label,train_data
# 测试函数

Nearest_Neighbor_number=9#k值可自己定义
train_label,train_data=trainingDataSet()
testFileList=listdir('testDigits')
error_sum=0
test_number=len(testFileList)
for i in range(test_number):
    # 测试集文件名
    fileNameStr=testFileList[i]
    # 切片后得到测试集索引
    classNumStr=classnumCut(fileNameStr)
    test_data=img2vector('testDigits/%s' % fileNameStr)
    ##调用knn算法进行测试
    classifierResult = KNN(test_data, train_data, train_label, Nearest_Neighbor_number)
    print("第", i + 1, "组：", "预测值:", classifierResult, "真实值:", classNumStr)
    if (classifierResult != classNumStr):
        error_sum+=1
print ("\n测试集总数为:",test_number)
print ("测试出错总数:",error_sum)
print ("\n错误率:",error_sum/float(test_number)*100,'%')
```
###结果
第 1 组： 预测值: 0 真实值: 0
第 2 组： 预测值: 0 真实值: 0
第 3 组： 预测值: 0 真实值: 0
第 4 组： 预测值: 0 真实值: 0
第 5 组： 预测值: 0 真实值: 0
第 6 组： 预测值: 0 真实值: 0
第 7 组： 预测值: 0 真实值: 0
第 8 组： 预测值: 0 真实值: 0
第 9 组： 预测值: 0 真实值: 0
第 10 组： 预测值: 0 真实值: 0
第 11 组： 预测值: 1 真实值: 1
第 12 组： 预测值: 1 真实值: 1
第 13 组： 预测值: 1 真实值: 1
第 14 组： 预测值: 1 真实值: 1
第 15 组： 预测值: 1 真实值: 1
第 16 组： 预测值: 1 真实值: 1
第 17 组： 预测值: 1 真实值: 1
第 18 组： 预测值: 1 真实值: 1
第 19 组： 预测值: 1 真实值: 1
第 20 组： 预测值: 1 真实值: 1
第 21 组： 预测值: 2 真实值: 2
第 22 组： 预测值: 2 真实值: 2
第 23 组： 预测值: 2 真实值: 2
第 24 组： 预测值: 2 真实值: 2
第 25 组： 预测值: 2 真实值: 2
第 26 组： 预测值: 2 真实值: 2
第 27 组： 预测值: 2 真实值: 2
第 28 组： 预测值: 2 真实值: 2
第 29 组： 预测值: 2 真实值: 2
第 30 组： 预测值: 2 真实值: 2
第 31 组： 预测值: 3 真实值: 3
第 32 组： 预测值: 3 真实值: 3
第 33 组： 预测值: 3 真实值: 3
第 34 组： 预测值: 3 真实值: 3
第 35 组： 预测值: 3 真实值: 3
第 36 组： 预测值: 3 真实值: 3
第 37 组： 预测值: 3 真实值: 3
第 38 组： 预测值: 3 真实值: 3
第 39 组： 预测值: 3 真实值: 3
第 40 组： 预测值: 3 真实值: 3
第 41 组： 预测值: 9 真实值: 4
第 42 组： 预测值: 9 真实值: 4
第 43 组： 预测值: 9 真实值: 4
第 44 组： 预测值: 9 真实值: 4
第 45 组： 预测值: 1 真实值: 4
第 46 组： 预测值: 4 真实值: 4
第 47 组： 预测值: 9 真实值: 4
第 48 组： 预测值: 4 真实值: 4
第 49 组： 预测值: 4 真实值: 4
第 50 组： 预测值: 4 真实值: 4
第 51 组： 预测值: 9 真实值: 5
第 52 组： 预测值: 5 真实值: 5
第 53 组： 预测值: 5 真实值: 5
第 54 组： 预测值: 5 真实值: 5
第 55 组： 预测值: 5 真实值: 5
第 56 组： 预测值: 5 真实值: 5
第 57 组： 预测值: 5 真实值: 5
第 58 组： 预测值: 5 真实值: 5
第 59 组： 预测值: 5 真实值: 5
第 60 组： 预测值: 5 真实值: 5
第 61 组： 预测值: 6 真实值: 6
第 62 组： 预测值: 6 真实值: 6
第 63 组： 预测值: 6 真实值: 6
第 64 组： 预测值: 6 真实值: 6
第 65 组： 预测值: 6 真实值: 6
第 66 组： 预测值: 6 真实值: 6
第 67 组： 预测值: 6 真实值: 6
第 68 组： 预测值: 6 真实值: 6
第 69 组： 预测值: 6 真实值: 6
第 70 组： 预测值: 6 真实值: 6
第 71 组： 预测值: 7 真实值: 7
第 72 组： 预测值: 7 真实值: 7
第 73 组： 预测值: 7 真实值: 7
第 74 组： 预测值: 7 真实值: 7
第 75 组： 预测值: 7 真实值: 7
第 76 组： 预测值: 7 真实值: 7
第 77 组： 预测值: 7 真实值: 7
第 78 组： 预测值: 7 真实值: 7
第 79 组： 预测值: 7 真实值: 7
第 80 组： 预测值: 7 真实值: 7
第 81 组： 预测值: 8 真实值: 8
第 82 组： 预测值: 9 真实值: 8
第 83 组： 预测值: 8 真实值: 8
第 84 组： 预测值: 5 真实值: 8
第 85 组： 预测值: 9 真实值: 8
第 86 组： 预测值: 8 真实值: 8
第 87 组： 预测值: 8 真实值: 8
第 88 组： 预测值: 8 真实值: 8
第 89 组： 预测值: 8 真实值: 8
第 90 组： 预测值: 8 真实值: 8
第 91 组： 预测值: 9 真实值: 9
第 92 组： 预测值: 9 真实值: 9
第 93 组： 预测值: 9 真实值: 9
第 94 组： 预测值: 9 真实值: 9
第 95 组： 预测值: 9 真实值: 9
第 96 组： 预测值: 9 真实值: 9
第 97 组： 预测值: 9 真实值: 9
第 98 组： 预测值: 9 真实值: 9
第 99 组： 预测值: 9 真实值: 9
第 100 组： 预测值: 9 真实值: 9

测试集总数为: 100
测试出错总数: 10

错误率: 10.0 %


  [1]: https://github.com/DL-Metaphysics/DL-yizhixiaol/tree/master/%E7%AC%AC%E4%BA%8C%E5%91%A8%E4%BD%9C%E4%B8%9A




