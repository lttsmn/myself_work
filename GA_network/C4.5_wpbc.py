import cv2
import os
import numpy as np
import random
import datetime
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils.np_utils import to_categorical
def get_sen_spe(test,predict):
    TP=0
    TN=0
    FP=0
    FN=0
    l=len(test)
    for i in range(0,l):
        if predict[i]==1:
            if test[i]==1:
                TP=TP+1
            else:
                FP=FP+1
        else:
            if test[i]==0:
                TN=TN+1
            else:
                FN=FN+1
    #print(TP,TN,FP,FN)
    if TP==0:
        sen=0
    else:
        sen=TP/(TP+FN)
    if TN==0:
        spe=0
    else:
        spe=TN/(FP+TN)
    return sen,spe
def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    sen,spe=get_sen_spe(y,prob)
    print("Sensitivity:",sen)
    print("specificity:",spe)
    print("AUC:",roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
def get_data():
    nb_classes = 2
    csv_data = pd.read_csv('./wpbc.csv',)  # 读取训练数据
    data_y=csv_data['V']
    csv_data=csv_data.drop('V',axis=1)
    data_y=np.array(data_y)
    csv_data=np.array(csv_data)
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.2, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 33)
    x_test = x_test.reshape(-1, 33)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    # convert class vectors to binary class matrices
   # y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    return x_train, x_test,y_train, y_test
X_train, X_test, y_train, y_test = get_data()
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
#print(y_test)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf0 = clf.fit(X_train, y_train)
print('The accuracy of C4.5 Classifier is',clf0.score(X_test,y_test))
predictions0 = clf0.predict(X_test)
acu_curve(y_test,predictions0)
