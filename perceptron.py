#����ģ��
import numpy as np
import matplotlib.pyplot as plt

#��ǩ������С
sample_size = 50
#����λ��
sample_location = 2

#ѵ������
training_num=100
#ѧϰ��
learning_rate=0.001

#Ȩֵ
w=np.array([-1,1])
#ƫ��
b=0

#������������
np.random.seed(0)
x=np.r_[np.random.randn(sample_size,2)-[sample_location,sample_location],np.random.randn(sample_size,2)+[sample_location,sample_location]]

#���ɱ�ǩ
y=[-1]*sample_size+[1]*sample_size

#������ɢ��ͼ
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
#��ʼѵ��
for step in range(training_num+1):
    grad_x=np.array([0,0])
    grad_b=0
    
    #ÿѵ��10�θ���һ�ηֽ���
    if step%10==0:
        ref_x=[-10,10]
        ref_y=[0,0]
        for i in range(len(ref_x)):
            ref_y[i]=-(w[0]*ref_x[i]+b)/w[1]
        pp=plt.plot(ref_x,ref_y)

    #����ѵ��������Ѱ�Ҵ��������
    for j in range(len(x)):
        if np.sign(np.dot(w,x[j])+b)<0 and y[j]!=np.sign(np.dot(w,x[j])+b):
            grad_x=grad_x-x[j]
            grad_b=grad_b+y[j]
    
    #�����ݶ��½������²���
    w=w+learning_rate*grad_x
    b=b+learning_rate*grad_b
    
plt.title('training')

lim_x=plt.xlim(-5,5)
lim_y=plt.ylim(-4.5,4.5)

x_lab=plt.xlabel('x',size=15)
y_lab=plt.ylabel('y',size=15)