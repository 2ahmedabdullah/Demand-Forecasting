#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:04:48 2021

@author: abdul
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def train_test_split(data1, strides, train_window, test_window):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    lists = []
    for i in range(1, no_of_stores+1):
        print('store no.',i)
        for j in range(1, no_of_items+1):
            print('item no.', j)
            df = data1.loc[(data1['store'] == i) & (data1['item'] == j)]
            if len(df)>0:
                m = [i, j]
                df = df.reset_index(drop=True)
                k=0
                X=[]
                Y=[]
                while k< (len(df)-train_window-test_window):
                    x = df[['sales']].iloc[k:train_window+k]
                    x = x.transpose()
                    x = x.values
                    y = df[['sales']].iloc[k+train_window:k+train_window+test_window]
                    k=k+strides
                    X.append(x)
                    y = y.transpose()
                    y = y.values
                    Y.append(y)
                    
                x_tr = X[0:len(X)-1]
                y_tr = Y[0:len(Y)-1]
                
                x_te = X[len(X)-1]
                y_te = Y[len(Y)-1]
                #print(len(x_tr))
                
                x_train.append(x_tr)
                y_train.append(y_tr)
                x_test.append(x_te)
                y_test.append(y_te)
                lists.append(m)
    return x_train, x_test, y_train, y_test, lists
            
            
x_train, x_test, y_train, y_test, names = train_test_split(data, strides, train_window, test_window)


x_train = np.concatenate(x_train, axis=0 )
x_test = np.concatenate(x_test, axis=0 )

y_train = np.concatenate(y_train, axis=0 )
y_test = np.concatenate(y_test, axis=0 )

import pickle

with open ('xtrain.pkl','wb') as f:
    pickle.dump(x_train, f)

with open ('xtest.pkl','wb') as f:
    pickle.dump(x_test, f)

with open ('ytrain.pkl','wb') as f:
    pickle.dump(y_train, f)

with open ('ytest.pkl','wb') as f:
    pickle.dump(y_test, f)

with open ('names.pkl','wb') as f:
    pickle.dump(names, f)


