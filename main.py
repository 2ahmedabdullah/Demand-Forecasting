#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:08:52 2021

@author: abdul
"""


import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing
from datetime import datetime, timedelta


data = pd.read_csv('train.csv')

data['date'] = pd.to_datetime(data['date'])

data['weekday'] = data['date'].dt.dayofweek


#SPLITTING THE DATASET

no_of_items = 50
no_of_stores = 10


train_window = 180
test_window = 90
strides = 1


def formatting(data1, strides, train_window, test_window):
    x_train = []
#    y_train = []
    x_test = []
#    y_test = []
    for i in range(1, no_of_stores+1):
        print('store no.',i)
        for j in range(1, no_of_items+1):
            #print('item no.', j)
            df = data1.loc[(data1['store'] == i) & (data1['item'] == j)]
            if len(df)>0:
                df = df.reset_index(drop=True)

                k=0
                X=[]
                #Y=[]
                while k< (len(df)-train_window-test_window):
                    x = df[['sales','weekday']].iloc[k:train_window+k]
                    x = x.transpose()
                    x = x.values
#                    y = df[['sales']].iloc[k+train_window:k+train_window+test_window]
                    k=k+strides
                    X.append(x)
#                    y = y.transpose()
#                    y = y.values
#                    Y.append(y)
                    
                x_tr = X[0:len(X)-1]
#                y_tr = Y[0:len(Y)-1]
                
                x_te = X[len(X)-1]
#                y_te = Y[len(Y)-1]
                #print(len(x_tr))
                
                x_train.append(x_tr)
#                y_train.append(y_tr)
                x_test.append(x_te)
#                y_test.append(y_te)        
    return x_train, x_test
            
            
x_train, x_test = formatting(data, strides, train_window, test_window)


x_train = np.concatenate(x_train, axis=0 )
x_test = np.concatenate(x_test, axis=0 )

y_train = np.concatenate(y_train, axis=0 )
y_test = np.concatenate(y_test, axis=0 )

import pickle

with open ('xtrain_rnn1.pkl','wb') as f:
    pickle.dump(x_train, f)

with open ('xtest_rnn1.pkl','wb') as f:
    pickle.dump(x_test, f)

with open ('ytrain_rnn2.pkl','wb') as f:
    pickle.dump(y_train, f)

with open ('ytest_rnn2.pkl','wb') as f:
    pickle.dump(y_test, f)

with open ('names.pkl','wb') as f:
    pickle.dump(names, f)

#load pickle files
import pickle

with open ('xtrain.pkl','rb') as f:
    x_train = pickle.load(f)

with open ('xtest.pkl','rb') as f:
    x_test = pickle.load(f)

with open ('ytrain.pkl','rb') as f:
    y_train = pickle.load(f)

with open ('ytest.pkl','rb') as f:
    y_test = pickle.load(f)

with open ('names.pkl','rb') as f:
    names = pickle.load(f)


#MODEL BUILDING

input_dim=train_window

model = Sequential()
model.add(Dense(150, activation='elu', input_shape=(input_dim,)))
model.add(Dense(120,   activation='elu'))
model.add(Dense(test_window,   activation='elu'))
model.compile(loss='mse', optimizer = Adam())

model1= model.fit(x_train, y_train, batch_size=128, epochs=100, 
                                verbose=1, validation_data=(x_test, y_test))

model.save('ANN.h5')
saved_model = load_model('ANN.h5')


plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

    
y_pred = model.predict(x_test)
y_pred1 = y_pred.ravel()
y_truth =y_test.ravel()


plt.scatter(y_truth, y_pred1)
plt.title('ANN Scatter Plot')
p1, p2 = [0, 180], [0, 180]
plt.plot(p1, p2, color ='red')    
plt.show()

corr, _ = pearsonr(y_truth, y_pred1)
print('Pearsons correlation: %.3f' % corr)
rmse= np.sqrt(np.square(y_truth - y_pred1))
print('Avg RMSE:', np.average(rmse))



store_id =input("Enter Store Id :")
item_id = input("Enter Item Id :")


k=50*(int(store_id)-1)+int(item_id)-1
y1 = y_test[k]
y2 = y_pred[k]
x_axis = np.arange(1, len(y1)+1)

plt.figure(figsize = (15, 7))
plt.plot(x_axis, y1)
plt.plot(x_axis, y2)
plt.title('Store No.'+store_id+' Item No.'+item_id, fontsize=20)
plt.legend(['actual', 'pred'], loc='upper right')
plt.figsize=(20, 6)
plt.xlabel('time step', fontsize=15)
plt.ylabel('sales', fontsize=15)
plt.show()

from keras.layers import LSTM

model = Sequential()
model.add(LSTM(180, activation='elu', input_shape=(1, 180)))
model.add(Dense(100, activation='elu'))
model.add(Dense(90))
model.compile(loss='mse', optimizer='adam')
# fit network
model2= model.fit(x_train, y_train, batch_size=128, epochs=100, 
                                verbose=1, validation_data=(x_test, y_test))



