#!/usr/bin/env python
# coding: utf-8

#
# Run this code as follows
#
# ex_multiclass.py <MODELNAME> <PDFNAME>
#
# where <MODELNAME> is the name of output model name (say my_model.h5)
# and <PDFNAME> is the name of the output PDF file with plots (say output.pdf)
#

#Import the necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import np_utils

import os
import sys
import warnings
warnings.filterwarnings('ignore')

#modelname = sys.argv[1]
#outputname = sys.argv[2]

modelname = "../ML_Output/3LTest_norm-5to5/multiclass_vll_3l.h5"
outputname = "../ML_Output/3LTest_norm-5to5/multiclass_vll_3l.pdf"

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

#This is the full list of input variables in the text file
col_names=['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l'];

#### For 2L1T: ['LT_2L1T', 'HT_2L1T', 'ngenjet', 'j0pT', 'missinget', 'ST_2L1T', 'LTHT', 'LTMET', 'dilepMass_2l1T', 'mT_2l1t_0', 'mT_2l1t_1', 'mtT_2l1t', 'dPhi_l0l1', 'dPhi_l0T', 'dPhi_l1T', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l0T', 'dR_l1T', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_TMET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'dPhi_Tj0', 'pT0_2l1t', 'pT1_2l1t', 'pT_tau_2l1t']

#cols = list(range(0,48))

#We start by using a small subset
#col_names=['Pt0','Pt1','Pt2','NBJet','MET','MaxDphi_LMet','MaxDphi_LL','MinDphi_LL','LLPairPt','Mt0','Mt1','Mt2']
#cols = [0,1,2,3,4,5,7,8,9,10,13,15,16,20,24,25] #26
#cols= list(range(len(col_names)))

cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


VLL_m500 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500_1 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500 = pd.concat([VLL_m500, VLL_m500_1])
VLL_m500['label']=0

VLL_m750 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750_1 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750 = pd.concat([VLL_m750, VLL_m750_1])
VLL_m750['label']=1

VLL_m1000 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_1 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000 = pd.concat([VLL_m1000, VLL_m1000_1])
VLL_m1000['label']=2

#Reducing to 1000
#VLL_m500 = VLL_m500[:1000]
#VLL_m750 = VLL_m750[:1000]
#VLL_m1000 = VLL_m1000[:1000]

data = pd.concat([VLL_m500,VLL_m750,VLL_m1000], ignore_index = True)
X, y = data.values[:,:-1], data.values[:,-1]


maxValues = X.max(axis=0)
minValues = X.min(axis=0)
print("Max values")
print(list(maxValues))
print("Min values")
print(list(minValues))
MaxMinusMin = X.max(axis=0) - X.min(axis=0)
normedX = 2*((X-X.min(axis=0))/(MaxMinusMin)) -1.0
X = normedX


### New Normalisation:

meanVals = X.mean(axis=0)
stdVals = X.std(axis=0)
normedX = (X-meanVals)/stdVals # Normalising the values
boolarr = abs((X-meanVals))/stdVals < 5   # Creating Boolean array of whether the values are within the given range 
normedX = normedX[boolarr.sum(axis = 1) == boolarr.shape[1]] # Selecting only events which have all the values within 5 sigma
X = normedX
y = y[boolarr.sum(axis = 1) == boolarr.shape[1]] # Updating the outputs too

print("Mean Values:")
print(list(meanVals))
print("STD Values:")
print(list(stdVals))

#Here the y values, or labels are turned from 0,1,2 into
# one hot encoded values (1,0,0),(0,1,0),(0,0,1)
ohe_y = tf.keras.utils.to_categorical(y)

### Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,ohe_y,test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(f'The number of input variables is {n_features}')


model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
#model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=20,batch_size=50,validation_data=(X_test,y_test),verbose=1)
                 
#Now we print the model summary to screen and save the model file
print('The NN architecture is')
model.summary()
model.save(modelname)


# Thats it. Now the rest of this file is just making various plots


# Let us start by making plots of the accuracy and loss as a function of epochs
# this tells us how the training went.
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')

#print("Training Accuracy:", history.history['accuracy'])
#print("Validation Accuracy:", history.history['val_accuracy'])

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='upper left')
#plt.savefig('acc_v_epoch.png')
plt.savefig(pp, format='pdf')
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')

#print("loss:", history.history['loss'])
#print("val_loss:", history.history['val_loss'])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
#plt.savefig('loss_v_epoch.png')
plt.savefig(pp, format='pdf')
plt.close()

pp.close()
print(f'All done. Output is saved as {outputname}')
