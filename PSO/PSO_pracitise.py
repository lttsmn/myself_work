#使用蜂群算法计算这个函数f = @(x,y) -20.*exp(-0.2.*sqrt((x.^2+y.^2)./2))-exp((cos(2.*pi.*x)+cos(2.*pi.*y))./2)+20+exp(1)在区间[-4,4]上的最小值
#它真正的最小值点是(0,0)
#https://blog.csdn.net/weixin_42528077/article/details/83795056
import numpy as np
import matplotlib.pyplot as plt

#定义待优化函数：只能处理行向量形式的单个输入，若有矩阵形式的多个输入应当进行迭代
def CostFunction(input):
    x = input[0]
    y = input[1]
    result = -20*np.exp(-0.2*np.sqrt((x*x+y*y)/2))- \
             np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)+20+np.exp(1)
    return result

#初始化各参数

nVar = 2
VarMin = -4 #求解边界
VarMax = 4
VelMin = -8 #速度边界
VelMax = 8
nPop = 40
iter_max = 100 #最大迭代次数
C1 = 2 #单粒子最优加速常数
C2 = 2 #全局最优加速常数
W = 0.5 #惯性因子
Gbest = np.inf #全局最优
History = np.inf*np.ones(iter_max) #历史最优值记录

#定义“粒子”类
class Particle(object):

    #初始化粒子的所有属性
    def __init__(self):
        self.Position = VarMin + (VarMax-VarMin)*np.random.rand(nVar)
        self.Velocity = VelMin + (VelMax-VelMin)*np.random.rand(nVar)
        self.Cost = np.inf
        self.Pbest = self.Position
        self.Cbest = np.inf

    #根据当前位置更新代价值的方法
    def UpdateCost(self):
        global Gbest
        self.Cost = CostFunction(self.Position)
        if self.Cost < self.Cbest:
            self.Cbest = self.Cost
            self.Pbest = self.Position
        if self.Cost < Gbest:
            Gbest = self.Cost

    #根据当前速度和单粒子历史最优位置、全局最优位置更新粒子速度
    def UpdateVelocity(self):
        global Gbest
        global VelMax
        global VelMin
        global nVar
        self.Velocity = W*self.Velocity + C1*np.random.rand(1)\
                        *(self.Pbest-self.Position) + C2*np.random.rand(1)\
                        *(Gbest-self.Position)
        for s in range(nVar):
            if self.Velocity[s] > VelMax:
                self.Velocity[s] = VelMax
            if self.Velocity[s] < VelMin:
                self.Velocity[s] = VelMin

    #更新粒子位置
    def UpdatePosition(self):
        global VarMin
        global VarMax
        global nVar
        self.Position = self.Position + self.Velocity
        for s in range(nVar):
            if self.Position[s] > VarMax:
                self.Position[s] = VarMax
            if self.Position[s] < VarMin:
                self.Position[s] = VarMin

#初始化粒子群
Group = []
for j in range(nPop):
    Group.append(Particle())

#开始迭代
for iter in range(iter_max):

    for j in range(nPop):
        Group[j].UpdateCost()
        if Group[j].Cost < History[iter]:
            History[iter] = Group[j].Cost

    for j in range(nPop):
        Group[j].UpdateVelocity()
        Group[j].UpdatePosition()


#输出结果
print(Gbest)
for i in range(nPop):
    if i % 5 == 0:
        print("这是最后的第i个粒子：",Group[i].Position, Group[i].Cost)

y = History.tolist()
x = [i for i in range(iter_max)]
plt.plot(x,y)
plt.show()
