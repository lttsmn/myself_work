"""
超参数的选择，初始权值的选择等等
"""
# -*- coding: utf-8 -*-
from bp import *
import numpy as np
# 适应度函数
def ras(case, label, weight_input, weight_output):
    sea = nn.train1(case, label, weight_input, weight_output)
    return sea
# 转为list
def tolist(weight_input,weight_output):
    g=[]
    for i in weight_input:
        g.extend(i.tolist())
    for i in weight_output:
        g.extend(i.tolist())
    return g
# list转为 数组
def toarray(arr):
    weightinput=arr[0:8].reshape(4,2)
    weightoutput=arr[8:10].reshape(2,1)
    return weightinput,weightoutput

def PSO(case, label):
    # 参数初始化
    w = 0.5
    c1 = 2.0
    c2 = 2.0
    maxgen = 10  # 进化次数
    sizepop = 1000   # 种群规模

    # 粒子速度和位置的范围
    Vmax =  1
    Vmin = -1
    popmax =  1.0
    popmin = -1.0

    # 产生初始粒子位置和速度
    pop = 1.0* np.random.uniform(-0.2,0.2,(sizepop,10))
    v = np.random.uniform(-1,1,(sizepop,10))
    fitness=[]
    for i in pop:
        weight1,weight2=toarray(i)
        fitness.append(ras(case, label,weight1,weight2) )          # 计算适应度
        print('49行')

    i = np.argmin(np.array(fitness))      # 找最好的个体
    gbest = pop                    # 记录个体最优位置
    # 取第i列的数据
    zbest = pop[i]              # 记录群体最优位置
    print('55行')
    print(zbest)
    print('-----------')
    fitnessgbest = fitness        # 个体最佳适应度值
    fitnesszbest = fitness[i]      # 全局最佳适应度值

    # 迭代寻优
    t = 0
    record = np.zeros(maxgen)
    while t < maxgen:

        # 速度更新
        v = w * v + c1 * np.random.random() * (gbest - pop) + c2 * np.random.random() * (zbest - pop)
        v[v > Vmax] = Vmax     # 限制速度
        v[v < Vmin] = Vmin

        # 位置更新
        pop = pop + 0.5 * v;
        pop[pop > popmax] = popmax  # 限制位置
        pop[pop < popmin] = popmin
        # # 自适应变异
        # p = np.random.random()             # 随机生成一个0~1内的数
        # if p > 0.8:                          # 如果这个数落在变异概率区间内，则进行变异处理
        #     k = np.random.randint(0,2)     # 在[0,2)之间随机选一个整数
        #     pop[:,k] = np.random.random()  # 在选定的位置进行变异
        # 计算适应度值
        for i in range(len(pop)):
            w1,w2=toarray(pop[i])
            fit = ras(case,label,w1,w2)

            # 个体最优位置更新
            if fit>fitnessgbest[i]:
                fitnessgbest[i]=fit
            # index = fit < fitnessgbest
            # print('index是')
            # print((fitness[0]))
            # fitnessgbest[index] = fitness[index]
            # gbest[:,index] = pop[:,index]

        # 群体最优更新
        j = np.argmin(fitnessgbest)
        if fitnessgbest [j] < fitnesszbest:
            zbest = pop[:,j]
            fitnesszbest = fitnessgbest[j]

        record[t] = fitnesszbest # 记录群体最优位置的变化
        t = t + 1
    print('成功了')
    return toarray(zbest)

nn=BPNeuralNetwork()
nn.setup(4,2,1)
case=np.random.rand(30,4)
label=np.random.randint(0,2,(30,1))
weight_input=np.random.uniform(-0.2,0.2,8).reshape(4,2)
weight_output=np.random.uniform(-0.2,0.2,2).reshape(2,1)

sea=ras(case,label,weight_input,weight_output)

arr=tolist(weight_input,weight_output)
a1,a2=PSO(case,label)
print('result are')
print(a1,a2)
nn.train(case,label,a1,a2)
x=[]
for cas in case[0:10]:
    x.append(nn.predict(cas))
print('----------------------------')
print(label.tolist())
print(x)

