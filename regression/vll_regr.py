#!/usr/bin/env python
# coding: utf-8

# First we import all the basic things we need.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from matplotlib.backends.backend_pdf import PdfPages

import os
import warnings

import keras.backend
#warnings.filterwarnings('ignore')

keras.backend.clear_session()
# Lets set up to put all output plots in one output PDF

outputname = '../ML_Output/regr/newvar_ep20_shuffle_train1200/newvar_training.pdf'
modelname = '../ML_Output/regr/newvar_ep20_shuffle_train1200/newvar_model.h5'
pp = PdfPages(outputname)

# input file has tof,height,distance,vel,angleDegrees
# from this, we decide to read in  tof,height,distance,vel

# First we prepare the dataframe in which we get the data
col_names = ['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1', 'Mos2', 
           'trilepM', 'mt0', 'mt1', 'mt2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 
           'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 
           'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l', 'l3mt', 'highmT', 'pT0j0pT', 
           'j0pTbyHT', 'pT0byLT']
cols = [0, 1, 9, 10, 11, 38, 39, 40]


VLL_m500 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500 = VLL_m500.head(1200)
#VLL_m500_1 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
#VLL_m500 = pd.concat([VLL_m500, VLL_m500_1])
VLL_m500['True_VLLD_Mass']=500


VLL_m1000 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_1000 = VLL_m1000.head(1200)
#VLL_m1000_1 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
#VLL_m1000 = pd.concat([VLL_m1000, VLL_m1000_1])
VLL_m1000['True_VLLD_Mass']=1000


VLL_m1500 = pd.read_csv('../VLLD_output/M1500/VLLD_mu_M1500_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500 = VLL_m1500.head(1200)
#VLL_m1500_1 = pd.read_csv('../VLLD_output/M1500/VLLD_mu_M1500_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
#VLL_m1500 = pd.concat([VLL_m1500, VLL_m1500_1])
VLL_m1500['True_VLLD_Mass']=1500

data = pd.concat([VLL_m500,VLL_m1000,VLL_m1500], ignore_index = True)
#Shuffling the order of data
data = data.sample(frac=1).reset_index(drop=True)
# ========   BEGIN TRAINING ====================================

# Save the label column as y, and the input variables as X
#(X and y are numpy arrays)

X, y = data.values[:,:-1], data.values[:,-1]

print(f'Shapes of data, X, y are {data.shape}, {X.shape} , {y.shape}')

n_features = X.shape[1]
print(f'The number of input variables is {n_features}')


# Now we declare a neural network with 2 hidden layers
#
# first hidden layer has 16 neurons, and takes n_features number of inputs
# second hidden layer has 8 neurons
# output layer has 1 neuron
#
# We have initialized weights using option 'he_normal' and
#  we have using the ReLU activation function for all neurons.

model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='relu'))

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', # metric to be monitored, typically 'loss' for training loss or 'val_loss' for validation loss. 
        patience=6, # number of epochs with no improvement after which training will be stopped
)
#compile the model, by choosing a learning rate and a loss function

model.compile(optimizer='adam', loss='MeanSquaredError')


# Now we train the model
history = model.fit(X,y,epochs=20,batch_size=50,verbose=1) #, callbacks = [early_stopping]
print(history.history.keys())

#Now we print the model summary to screen and save the trained model to a file

print('The NN architecture is')
model.summary()
model.save(modelname)


# ======================================================================================


# ============== NOW VISUALIZE THE TRAINING PROCESS  ===================================
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig(pp,format='pdf')


# ======================================================================================


# ============== EVALUATING THE MODEL  =================================================

# Now we evaluate the model using our test file.
# Let us begin by reading in the test file, and separating out the first three variables and the output


VLL_m500_test = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500_test['True_VLLD_Mass']=500

VLL_m1000_test = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_test['True_VLLD_Mass']=1000

VLL_m1500_test = pd.read_csv('../VLLD_output/M1500/VLLD_mu_M1500_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500_test['True_VLLD_Mass']=1500

VLL_m750_test = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750_test['True_VLLD_Mass']=750


data_test = pd.concat([VLL_m500_test,VLL_m1000_test,VLL_m1500_test])
#, VLL_m750_test
print(VLL_m500_test.shape, VLL_m1000_test.shape, VLL_m1500_test.shape, VLL_m750_test.shape)

# Then separate the variables and the result columns
X_test, y_true = data_test.values[:,:-1], data_test.values[:,-1]

# Make the prediction
pred_y = model.predict(X_test)

#At this point,  y_true has the true answers  and pred_y  has the predicted answers


# Arrange them back in a nice dataframe

results = pd.DataFrame()
results['True_VLLD_Mass'] = y_true.ravel()
results['Pred_VLLD_Mass'] = pred_y
print(pred_y.shape, type(pred_y))

# Let us calculate a quick figure of merit for our sake
results['diffsquare'] = results.apply(lambda row: np.square(row.True_VLLD_Mass-row.Pred_VLLD_Mass) , axis=1 )

#Now let us print the dataframe
print(results.head(10))

TotalDiff = results['diffsquare'].sum() / results.shape[0]
print(f"The total difference between expectation and prediction, the MSE = {TotalDiff:.4f}")

fig, ax = plt.subplots(figsize = (10, 10))

ax.hist(results.diffsquare, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('diffsquare', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Performance plot: Total MSE = {TotalDiff:.4f}', fontsize = 16)
plt.savefig(pp, format = 'pdf')


#### Plotting hists of predictes mass points for the three different masses

fig, ax = plt.subplots(figsize = (10, 10))

pred_500 = results[results['True_VLLD_Mass']==500]['Pred_VLLD_Mass']

ax.hist(pred_500, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('Predicted_VLLD_Mass', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Pred Mass for VLLD_M = 500', fontsize = 16)

meanmine= round(pred_500.mean(),2)
print(meanmine)
ax.annotate(f"Mean = %.2f\nSD = {round(pred_500.std(),2)}"%meanmine, xy = (0.67,0.7), xycoords = 'figure fraction',fontsize=16)
plt.savefig(pp, format = 'pdf')




fig, ax = plt.subplots(figsize = (10, 10))


pred_1000 = results[results['True_VLLD_Mass']==1000]['Pred_VLLD_Mass']

ax.hist(pred_1000, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('Predicted_VLLD_Mass', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Pred Mass for VLLD_M = 1000', fontsize = 16)
meanmine= round(pred_1000.mean(),2)
print(meanmine)
ax.annotate(f"Mean = %.2f\nSD = {round(pred_1000.std(),2)}"%meanmine, xy = (0.67,0.7), xycoords = 'figure fraction',fontsize=16)
plt.savefig(pp, format = 'pdf')



fig, ax = plt.subplots(figsize = (10, 10))


pred_1500 = results[results['True_VLLD_Mass']==1500]['Pred_VLLD_Mass']

ax.hist(pred_1500, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('Predicted_VLLD_Mass', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Pred Mass for VLLD_M = 1500', fontsize = 16)
meanmine= round(pred_1500.mean(),2)
print(meanmine, len(str(meanmine)))

ax.annotate(f"Mean = %.2f\nSD = {round(pred_1500.std(),2)}"%meanmine, xy = (0.67,0.7), xycoords = 'figure fraction',fontsize=16)
plt.savefig(pp, format = 'pdf')


fig, ax = plt.subplots(figsize = (10, 10))

'''
pred_750 = results[results['True_VLLD_Mass']==750]['Pred_VLLD_Mass']

ax.hist(pred_750, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('Predicted_VLLD_Mass', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Pred Mass for VLLD_M = 750', fontsize = 16)
meanmine= round(pred_750.mean(),2)
print(meanmine, len(str(meanmine)))

ax.annotate(f"Mean = %.2f\nSD = {round(pred_750.std(),2)}"%meanmine, xy = (0.67,0.7), xycoords = 'figure fraction',fontsize=16)
plt.savefig(pp, format = 'pdf')
'''




# Now we close the output file
pp.close()
