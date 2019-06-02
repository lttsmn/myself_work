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

nb_classes = 2
csv_data = pd.read_csv('./CBC.csv',)  # 读取训练数据
data_y=csv_data['V']
#csv_data=csv_data.drop('V',axis=1)
data_y=np.array(data_y)
csv_data=np.array(csv_data)
acc=[]
from sklearn import svm
for i in range(20):
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.2, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 10)
    x_test = x_test.reshape(-1, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    clf0 = svm.SVC(gamma='auto').fit(x_train, y_train)
    predictions0 = clf0.predict(x_test)
    score=clf0.score(x_test,predictions0)
    print("score is:",score)
    acc.append(score)
print("mean accuracy is",np.mean(acc))
#acu_curve(y_test,predictions0)
