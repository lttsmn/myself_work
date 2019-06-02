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
    print(roc_auc)
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
    csv_data = pd.read_csv('./CBC.csv',)  # 读取训练数据
    data_y=csv_data['V']
    csv_data=csv_data.drop('V',axis=1)
    data_y=np.array(data_y)
    csv_data=np.array(csv_data)
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.2, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 9)
    x_test = x_test.reshape(-1, 9)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    # convert class vectors to binary class matrices
   # y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    return x_train, x_test,y_train, y_test
def get_result():
    acc_data=[]
    spe_data=[]
    sen_data=[]
    auc_data=[]
    knn_acc=[]
    knn_spe=[]
    knn_sen=[]
    knn_auc=[]
    svm_acc=[]
    svm_spe=[]
    svm_sen=[]
    svm_auc=[]
    nb_acc=[]
    nb_spe=[]
    nb_sen=[]
    nb_auc=[]
    from sklearn import tree
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import BernoulliNB
    bnl=BernoulliNB()
    clf = tree.DecisionTreeClassifier()
    clf_knn = KNeighborsClassifier(n_neighbors=8)
    clf_svm = svm.SVC(gamma='auto')
    for i in range(10):
        X_train, X_test, y_train, y_test = get_data()
        clf0 = clf.fit(X_train, y_train)
        acc=clf0.score(X_test,y_test)
        predictions0 = clf0.predict(X_test)
        sen,spe=get_sen_spe(y_test,predictions0)
        fpr,tpr,threshold = roc_curve(y_test, predictions0 ) ###计算真正率和假正率
        roc_auc = auc(fpr,tpr) ###计算auc的值
        acc_data.append(acc)
        sen_data.append(sen)
        spe_data.append(spe)
        auc_data.append(roc_auc)

        #KNN 模型
        clf1=clf_knn.fit(X_train, y_train)
        acc=clf1.score(X_test,y_test)
        predictions1 = clf1.predict(X_test)
        sen,spe=get_sen_spe(y_test,predictions1)
        fpr1,tpr1,threshold = roc_curve(y_test, predictions1 )
        roc_auc1 = auc(fpr1,tpr1)
        knn_acc.append(acc)
        knn_sen.append(sen)
        knn_spe.append(spe)
        knn_auc.append(roc_auc1)

        #SVM模型
        clf2=svm.SVC(gamma='auto').fit(X_train, y_train)
        acc=clf2.score(X_test,y_test)
        predictions2 = clf2.predict(X_test)
        #print(predictions2)
        sen,spe=get_sen_spe(y_test,predictions2)
        fpr2,tpr2,threshold = roc_curve(y_test, predictions2 ) ###计算真正率和假正率
        roc_auc2 = auc(fpr2,tpr2) ###计算auc的值
        svm_acc.append(acc)
        svm_sen.append(sen)
        svm_spe.append(spe)
        svm_auc.append(roc_auc2)

        #NB 模型
        clf3=bnl.fit(X_train, y_train)
        acc=clf3.score(X_test,y_test)
        predictions3 = clf3.predict(X_test)
        sen,spe=get_sen_spe(y_test,predictions3)
        fpr3,tpr3,threshold = roc_curve(y_test, predictions3 ) ###计算真正率和假正率
        roc_auc3 = auc(fpr3,tpr3) ###计算auc的值
        nb_acc.append(acc)
        nb_sen.append(sen)
        nb_spe.append(spe)
        nb_auc.append(roc_auc3)



    print("CBC C4.5  acc is %d---%d",np.mean(acc_data),np.std(acc_data))
    #print("CBC std acc is:",acc_std)
    print("CBC  C4.5 sen is %d---%d",np.mean(sen_data),np.std(sen_data))
    #print("CBC std sen is:",sen_std)
    print("CBC  C4.5 spe is %d---%d",np.mean(spe_data),np.std(spe_data))
    #print("CBC std spe is:",spe_std)
    print("CBC  C4.5 auc is %d---%d",np.mean(auc_data),np.std(auc_data))


    print("CBC  KNN acc is %f---%f",np.mean(knn_acc),np.std(knn_acc))
    #print("CBC std acc is:",acc_std)
    print("CBC  KNN sen is %f---%f",np.mean(knn_sen),np.std(knn_sen))
    #print("CBC std sen is:",sen_std)
    print("CBC  KNN spe is %f---%f",np.mean(knn_spe),np.std(knn_spe))
    #print("CBC std spe is:",spe_std)
    print("CBC  KNN auc is %f---%f",np.mean(knn_auc),np.std(knn_auc))


    print("CBC  SVM acc is %f---%f",np.mean(svm_acc),np.std(svm_acc))
    #print("CBC std acc is:",acc_std)
    print("CBC  SVM sen is %f---%f",np.mean(svm_sen),np.std(svm_sen))
    #print("CBC std sen is:",sen_std)
    print("CBC  SVM spe is %f---%f",np.mean(svm_spe),np.std(svm_spe))
    #print("CBC std spe is:",spe_std)
    print("CBC  SVM auc is %f---%f",np.mean(svm_auc),np.std(svm_auc))



    print("CBC  NB acc is %f---%f",np.mean(nb_acc),np.std(nb_acc))
    #print("CBC std acc is:",acc_std)
    print("CBC  NB sen is %f---%f",np.mean(nb_sen),np.std(nb_sen))
    #print("CBC std sen is:",sen_std)
    print("CBC  NB spe is %f---%f",np.mean(nb_spe),np.std(nb_spe))
    #print("CBC std spe is:",spe_std)
    print("CBC  NB auc is %f---%f",np.mean(nb_auc),np.std(nb_auc))
    return fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3
def main():
    fpr,tpr,fpr1, tpr1,fpr2, tpr2,fpr3, tpr3,roc_auc,roc_auc1,roc_auc2,roc_auc3=get_result()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='green',
                 lw=lw, label='C4.5 (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr1, tpr1, color='red',
                 lw=lw, label='KNN (area = %0.3f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr2, tpr2, color='skyblue',
                 lw=lw, label='SVM (area = %0.3f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr3, tpr3, color='yellow',
                 lw=lw, label='NB (area = %0.3f)' % roc_auc3) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
if __name__ == '__main__':
    main()

