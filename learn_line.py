import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,cross_val_score
#from funct import Plt_classifier,Plt_confusion_matrix
from sklearn.metrics import f1_score,confusion_matrix,classification_report


iris_data = load_iris()
features = iris_data.data[:,0:2] #矩阵切片,只要两个变量  注意数组切片为[,:,]
labels = iris_data.target
features_train,features_test,labels_train,labels_test = train_test_split(features,
                                                                         labels , test_size = 0.3,random_state = 0)


#前面的函数有一些输入参数需要设置，但是最重要的两个参数是solver和C。参数solver
#用于设置求解系统方程的算法类型，参数C表示正则化强度，数值越大，惩罚值越高，表示正则化强度越大。
classifier = LogisticRegression(solver = 'liblinear', C = 10)
logistic = classifier.fit(features_train,labels_train)

#绘制分类边界,根据两个特征的分类边界
#Plt_classifier(classifier,features_train,labels_train)

#验证集预测
label_pred = logistic.predict(features_test)
f1_score = f1_score(labels_test, label_pred, average = 'weighted') #多分类必须加上average参数设置，否则默认为binary，无法正确输出
print('f1',f1_score)
#Plt_classifier(classifier,features_test,labels_test)

#在上面，我们把数据分成了训练数据集和测试数据集。
#然而，为了能够让模型更加稳定，还需要用数据集的不同子集进行反复的验证。如果只是对特定
#的子集进行微调，最终可能会过度拟合（overfitting）模型
num_validation = 5
f1 = cross_val_score(logistic, features, labels, scoring = 'f1_weighted', cv = num_validation )
f1 = f1.mean()
print('f1',f1)
recall = cross_val_score(logistic, features, labels, scoring = 'recall_weighted', cv = num_validation)
recall = recall.mean()
print('recall',recall)
precision = cross_val_score(logistic, features, labels, scoring = 'precision_weighted', cv = num_validation)
precision = precision.mean()
print('precision',precision)


#继续使用随机森林建立模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve,learning_curve
features = iris_data.data
labels = iris_data.target
 
RF = RandomForestClassifier(max_depth = 8, random_state = 0)
params_grid = np.linspace(25,200,8).astype(int)
 
#其他参数不变，观察评估器数量对训练得分的影响
train_scores,validation_scores = validation_curve(RF,features,labels,'n_estimators',params_grid,cv=5)
print('######Validation Curve')
print('train_score\n',train_scores)
print('validation_score\n',validation_scores)
#可视化生成训练、验证曲线
plt.figure()
plt.plot(params_grid, np.average(train_scores,axis = 1),color = 'red')
plt.plot(params_grid,np.average(validation_scores,axis = 1),color = 'black')
plt.title('Training curve')
plt.xlabel('number of estimator')
plt.ylabel('accuracy')
plt.show()
#同样的方法可以验证其他变量对训练的影响，多次操作，进行参数调整
#生成学习曲线
print(len(features))
size_grid = np.array([0.2,0.5,0.7,1])
train_size,train_scores,validation_scores = learning_curve(RF,features,labels,train_sizes = size_grid, cv = 5)
print('######Learning Curve')
print('train_score\n',train_scores)
print('validation_score\n',validation_scores)
#学习曲线可视化
plt.figure()
plt.plot(size_grid,np.average(train_scores, axis = 1), color = 'red')
plt.plot(size_grid, np.average(validation_scores, axis = 1), color = 'black')
plt.title('Learning Curve')
plt.xlabel('sample size')
plt.ylabel('accuracy')
plt.show()