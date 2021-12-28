#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:36:44 2021

@author: abdul
"""

#load pickle files
import pickle
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import warnings
from scipy.stats import pearsonr, chi2_contingency
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
import matplotlib.pyplot as plt


with open ('xtrain.pkl','rb') as f:
    xtrain = pickle.load(f)

with open ('xtest.pkl','rb') as f:
    xtest = pickle.load(f)

with open ('ytrain.pkl','rb') as f:
    ytrain = pickle.load(f)

with open ('ytest.pkl','rb') as f:
    ytest = pickle.load(f)


#MODEL BUILDING
input_dim = 180
test_window = 90

model = Sequential()
model.add(Dense(150, activation='elu', input_shape=(input_dim,)))
model.add(Dense(120,   activation='elu'))
model.add(Dense(test_window,   activation='elu'))
model.compile(loss='mse', optimizer = Adam())

model1= model.fit(xtrain, ytrain, batch_size=256, epochs=100, 
                                verbose=1, validation_data=(xtest, ytest))

model.save('ANN.h5')
saved_model = load_model('ANN.h5')




plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

y_pred = model.predict(xtest)
y_pred1 = y_pred.ravel()
y_truth =ytest.ravel()


plt.scatter(y_truth, y_pred1)
plt.title('MLP Scatter Plot')
p1, p2 = [0, 180], [0, 180]
plt.plot(p1, p2, color ='red')    
plt.show()

corr, _ = pearsonr(y_truth, y_pred1)
print('Pearsons correlation: %.3f' % corr)
rmse= np.sqrt(np.square(y_truth - y_pred1))
print('Avg RMSE:', round(np.average(rmse),3))


#LOG ROOT MEAN SQUARE ERROR
l1 = np.log(y_truth+1)
l2 = np.log(y_pred1+1)

diff = np.square(l1 - l2)
v = np.sqrt(np.sum(diff)/len(diff))
print('Log Root Mean Square Error: ', round(v,3))


