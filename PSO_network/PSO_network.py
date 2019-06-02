import numpy as np
import random 
import matplotlib.pyplot as plt
import neurolab as nl
import pandas as pd
#http://code.google.com/p/neurolab/downloads/list
train_x =[]
d=[]
samplescount=1000
myrndsmp=np.random.rand(samplescount)
for yb_i in range(0,samplescount):
    train_x.append([myrndsmp[yb_i]*4*np.pi-2*np.pi])
for yb_i in range(0,samplescount):
    d.append(np.sin(train_x[yb_i])*0.5+np.cos(train_x[yb_i])*0.5)
myinput=np.array(train_x)   
mytarget=np.array(d)
csv_data = pd.read_csv('G:/R/heart.csv',)  # 读取训练数据
data_y=csv_data['V']
csv_data = np.array(csv_data)
data_y=np.array(data_y)
print(csv_data)
#PSO参数设置
class PSO():
    def __init__(self,max_iter):
        #self.w = 0.8  
        self.c1 = 2   
        self.c2 = 2   
        self.pN =10               #粒子数量
        self.dim = 1              #搜索维度
        self.max_iter = max_iter    #迭代次数
        self.X = np.ones((self.pN,self.dim))       #所有粒子的位置和速度
        self.V = np.zeros((self.pN,self.dim))
        self.pbest = np.zeros((self.pN,self.dim))   #个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1,self.dim))
        self.p_fit = np.zeros(self.pN)              #每个个体的历史最佳适应值
        self.fit = 1e10             #全局最佳适应值
        self.wmax=0.9
        self.wmin=0.4
#目标函数
    def fun(self,err):
        fitness=err
        return fitness
#初始化种群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(1,20)
                self.V[i][j] = random.uniform(0,2)
            self.pbest[i] = self.X[i]
          
            for x in self.pbest[i]:
                return x
            bpnet = nl.net.newff([[-2*np.pi, 2*np.pi]], [int(self.X)+1, 1])
            err = bpnet.train(myinput, mytarget, epochs=800, show=100, goal=0.02)
            #out=net.sim(input)
            tmp = self.fun(err)
            self.p_fit[i] = tmp
            if(tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]
    
#更新粒子位置
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            w=self.wmax-(self.wmax-self.wmin)*(float(t)/self.max_iter)
            for i in range(self.pN):
                for x in self.pbest[i]:
                    return x
                print (self.X)
                bpnet = nl.net.newff([[-2*np.pi, 2*np.pi]], [int(self.X)+1, 1])
                err = bpnet.train(myinput, mytarget, epochs=800, show=100, goal=0.02)
                temp = self.fun(err)
                if(temp<self.p_fit[i]):      #更新个体最优
                   self.p_fit[i] = temp
                   self.pbest[i] = self.X[i]
                   if(self.p_fit[i] < self.fit):  #更新全局最优
                       self.gbest = self.X[i]
                       self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = w*self.V[i] + self.c1*np.random.uniform(0,1)*(self.pbest[i] - self.X[i])\
                       + self.c2*np.random.uniform(0,1)*(self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            #print(self.fit)                   #输出最优值
        
        return self.X
 
#-程序执行
my_pso = PSO(max_iter=100)
my_pso.init_Population()
x= my_pso.iterator()
print (int(x)+1)
bpnet = nl.net.newff([[-2*np.pi, 2*np.pi]], [int(x)+1, 1])
err = bpnet.train(myinput, mytarget, epochs=800, show=10, goal=0.02)
#误差曲线
plt.title("pso-bp")
plt.plot(range(len(err)),err)
plt.xlabel('Epoch number')
plt.ylabel('err (default SSE)')
#可视化图
plt.show()