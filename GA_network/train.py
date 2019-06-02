"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)
def get_wpbc():
    nb_classes = 2
    batch_size = 5
    input_shape = (33,)
    csv_data = pd.read_csv('./wpbc.csv',)  # 读取训练数据
    #print(csv_data)
    #print(csv_data.shape)
    data_y=csv_data['V']
    #print(data_y)
    # Get the data.
    csv_data=csv_data.drop('V',axis=1)
    csv_data = np.array(csv_data)
    data_y=np.array(data_y)
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.2, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 33)
    x_test = x_test.reshape(-1, 33)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,test)

def get_CBC():
    nb_classes = 2
    batch_size = 5
    input_shape = (9,)
    csv_data = pd.read_csv('./CBC.csv',)  # 读取训练数据
    #print(csv_data)
    #print(csv_data.shape)
    data_y=csv_data['V']
    csv_data=csv_data.drop('V',axis=1)
    #print(data_y)
    # Get the data.
    csv_data = np.array(csv_data)
    data_y=np.array(data_y)
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.2, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 9)
    x_test = x_test.reshape(-1, 9)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    #print(test)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,test)
def get_wbc():
    nb_classes = 2
    batch_size = 5
    input_shape = (9,)
    csv_data = pd.read_csv('./wbc.csv',)  # 读取训练数据
    #print(csv_data)
    data_y=csv_data['V']
    #print(data_y)
    #csv_data=eval(csv_data)
    #print(data_y)
    # Get the data.
    #print(type(csv_data))
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
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,test)
def get_breasttissue():
    nb_classes = 2
    batch_size = 5
    input_shape = (9,)
    csv_data = pd.read_excel('./BreastTissue.xlsx',sheetname="Data")  # 读取训练数据

    #print(csv_data)
    #print(csv_data.shape)
    y=csv_data['Class']
    csv_data=csv_data.drop('Case',axis=1)
    csv_data=csv_data[-csv_data.Class.isin(['fad','mas','gla','con'])]
    #print(csv_data)
    csv_data=csv_data.drop('Class',axis=1)

    data_y=[]
    for i in y:
        if i=="car": data_y.append(0)
        elif i=="adi":data_y.append(1)


    #print(data_y)
   #print(csv_data)
    # Get the data.
    csv_data = np.array(csv_data)
    data_y=np.array(data_y)
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.3, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 9)
    x_test = x_test.reshape(-1, 9)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    #print(test)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,test)
def get_data():
    nb_classes = 2
    batch_size = 5
    input_shape = (31,)
    csv_data = pd.read_csv('./wdbc.csv',)  # 读取训练数据
    #print(csv_data)
    #print(csv_data.shape)
    data_y=csv_data['V']
    #print(data_y)
    # Get the data.
    csv_data=csv_data.drop('V',axis=1)
    csv_data = np.array(csv_data)
    data_y=np.array(data_y)
    x_train, x_test,y_train, y_test = train_test_split(csv_data, data_y, test_size=0.2, random_state=random.randint(0,100))
    x_train = x_train.reshape(-1, 31)
    x_test = x_test.reshape(-1, 31)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    test=y_test
    #print(y_test)
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model
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
def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'CBC':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test,test = get_CBC()
    elif dataset == 'BreastTossue':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test,test = get_breasttissue()
    elif dataset=="wdbc":
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test,test = get_data()
    elif dataset=="wpbc":
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test,test = get_wpbc()
    elif dataset=="wbc":
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test,test = get_wbc()
    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
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
    return score[1],sen,spe,test,predict,roc_auc  # 1 is accuracy. 0 is loss.
