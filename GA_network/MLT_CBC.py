from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import numpy as np
from keras.callbacks import EarlyStopping
import random
import datetime
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
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
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return x_train, x_test,y_train, y_test,test
acc_data=[]
spe_data=[]
sen_data=[]
auc_data=[]
early_stopper = EarlyStopping(patience=5)
input_shape = (9,)
for i in range(20):
    x_train, x_test, y_train, y_test ,test= get_data()
    model = Sequential()
    model.add(Dense(256,activation='sigmoid',input_shape=input_shape))

    model.add(Dense(128,activation='sigmoid'))

    model.add(Dense(64,activation='sigmoid'))

    model.add(Dense(2,activation='sigmoid'))

    #model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adamax',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=5,
              epochs=3000, #epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])
    score = model.evaluate(x_test, y_test, verbose=0)
    predict=model.predict_classes(x_test)
    sen,spe=get_sen_spe(test,predict)
    #acu_curve(test,predict)
    fpr,tpr,threshold = roc_curve(test,predict) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    acc_data.append(score)
    sen_data.append(sen)
    spe_data.append(spe)
    auc_data.append(roc_auc)
acc_mean=np.mean(acc_data)
acc_std=np.std(acc_data)
sen_mean=np.mean(sen_data)
sen_std=np.std(sen_data)
spe_mean=np.mean(spe_data)
spe_std=np.std(spe_data)
auc_mean=np.mean(auc_data)
auc_std=np.std(auc_data)
print("mean acc is",acc_mean)
print("std acc is:",acc_std)
print("mean sen is",sen_mean)
print("std sen is:",sen_std)
print("mean spe is",spe_mean)
print("std spe is:",spe_std)
print("mean auc is",auc_mean)
print("std auc is:",auc_std)
acu_curve(test,predict)
