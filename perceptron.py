#导入模块
import numpy as np
import matplotlib.pyplot as plt

#标签样本大小
sample_size = 50
#中心位置
sample_location = 2

#训练次数
training_num=100
#学习率
learning_rate=0.001

#权值
w=np.array([-1,1])
#偏置
b=0

#生成数据样本
np.random.seed(0)
x=np.r_[np.random.randn(sample_size,2)-[sample_location,sample_location],np.random.randn(sample_size,2)+[sample_location,sample_location]]

#生成标签
y=[-1]*sample_size+[1]*sample_size

#画样本散点图
for i in range(len(x)):
    if y[i]==1:
        plt.plot(x[i][0],x[i][1],'rx')
    else:
        plt.plot(x[i][0],x[i][1],'bo')
plt.title('sample distribution')
lim_x=plt.xlim(-5,5)
lim_y=plt.ylim(-4.5,4.5)

x_lab=plt.xlabel('x',size=15)
y_lab=plt.ylabel('y',size=15)


for i in range(len(x)):
    if y[i]==1:
        plt.plot(x[i][0],x[i][1],'rx')
    else:
        plt.plot(x[i][0],x[i][1],'bo')
#开始训练
for step in range(training_num+1):
    grad_x=np.array([0,0])
    grad_b=0
    
    #每训练10次更新一次分界线
    if step%10==0:
        ref_x=[-10,10]
        ref_y=[0,0]
        for i in range(len(ref_x)):
            ref_y[i]=-(w[0]*ref_x[i]+b)/w[1]
        pp=plt.plot(ref_x,ref_y)

    #遍历训练样本集寻找错分样本、
    for j in range(len(x)):
        if np.sign(np.dot(w,x[j])+b)<0 and y[j]!=np.sign(np.dot(w,x[j])+b):
            grad_x=grad_x-x[j]
            grad_b=grad_b+y[j]
    
    #利用梯度下降法更新参数
    w=w+learning_rate*grad_x
    b=b+learning_rate*grad_b
    
plt.title('training')

lim_x=plt.xlim(-5,5)
lim_y=plt.ylim(-4.5,4.5)

x_lab=plt.xlabel('x',size=15)
y_lab=plt.ylabel('y',size=15)