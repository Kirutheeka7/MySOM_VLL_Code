#!/usr/bin/env python
# coding: utf-8


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

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys
import warnings
warnings.filterwarnings('ignore')

#modelname = sys.argv[1]
#outputname = sys.argv[2]



###### Begin Function definitions ######
def normalise_df(df, arg1, arg2):
    #Normalising to z-score
    #arg1 = meanvals, arg2 = stdvals
    normeddf = (df - arg1)/arg2
    
    #Normalising from 0 to 1
    #arg1 = maxvals; arg2 = minvals
    #normeddf = (df - arg2)/(arg1 - arg2)

    #Normalising from -1 to 1
    #arg1 = maxvals; arg2 = minvals
    #normeddf = 2 * ((df - arg2)/(arg1-arg2)) - 1.0

    return normeddf


def plot_process_scores(nnscore, dfname):

    plt.figure(figsize=(8,6))
    df_length = nnscore.shape[0]
    for ind in range(0,3):
        
        plt.hist(nnscore[:,ind],bins=mybins,histtype='step',label=labelnames[ind],linewidth=3,
                 color=mycols[ind],density=False,log=False, weights=1/df_length*np.ones(df_length))
    
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'NN Output for {dfname}',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.savefig(pp, format='pdf')
    plt.close()



def plot_neuron_scores(nnscore_list, dfnames, proclabel):
    ### proclabel: {0: 'M500', 1:'M1000', 2:'M1500'}
    
    plt.figure(figsize=(8,6))
    
    for ind in range(0,3):
        df_length = nnscore_list[ind].shape[0]
        ### Plotting the neuron score for different processes
        plt.hist(nnscore_list[ind][:,proclabel],bins=mybins,histtype='step',label=dfnames[ind],
                 linewidth=3,color=mycols[ind],density=False,log=False, weights=1/df_length*np.ones(df_length))
        
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'{labelnames[proclabel]} NN Output',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.savefig(pp, format='pdf')    
    plt.close()


####### End Function definitons #######


foldername = "mc_3Ltest_meanstd_50ep/"
modelname = "../ML_Output/"+foldername+"mc_model.h5"
outputname = "../ML_Output/"+foldername+"train_test_plots.pdf"
textfilename = "../ML_Output/"+foldername+"summary_metrics.txt"


pp = PdfPages(outputname)

#This is the full list of input variables in the text file
col_names=['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1', 'Mos2', 
           'trilepM', 'mt0', 'mt1', 'mt2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 
           'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 
           'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l', 'l3mt', 'highmT', 'pT0j0pT', 
           'j0pTbyHT', 'pT0byLT']

#These are the variables we will be training on

train_vars = ['LT_3L', 'HT_3L', 'Mos1', 'Mos2', 'trilepM', 'l3mt', 'highmT', 'pT0j0pT'] #Training variables
cols = [col_names.index(var) for var in train_vars]   #Getting the indices of the training varaibles



### Loading all the datasets
VLL_m500_0 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500_1 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m500_2 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

VLL_m1000_0 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_1 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000_2 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

VLL_m1500_0 = pd.read_csv('../VLLD_output/M1500/VLLD_mu_M1500_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500_1 = pd.read_csv('../VLLD_output/M1500/VLLD_mu_M1500_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1500_2 = pd.read_csv('../VLLD_output/M1500/VLLD_mu_M1500_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

VLL_m750_0 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750_1 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_1.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750_2 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)



### Setting the training datasets
VLL_m500_train = pd.concat([VLL_m500_0, VLL_m500_1])
VLL_m500_train['label']=0


VLL_m1000_train = pd.concat([VLL_m1000_0, VLL_m1000_1])
VLL_m1000_train['label']=1

VLL_m1500_train = pd.concat([VLL_m1500_0, VLL_m1500_1])
VLL_m1500_train['label']=2


data = pd.concat([VLL_m500_train,VLL_m1000_train,VLL_m1500_train], ignore_index = True)


#To randomise the order of training samples
#data = data.sample(frac=1).reset_index(drop=True)


X, y = data.values[:,:-1], data.values[:,-1]


maxValues = X.max(axis=0)
minValues = X.min(axis=0)
meanValues = X.mean(axis=0)
stdValues = X.std(axis=0)
print("Max values")
print(list(maxValues))
print("Min values")
print(list(minValues))
print("Mean values")
print(list(meanValues))
print("STD values")
print(list(stdValues))

#Mean-std scaling
arg1 = meanValues; arg2 = stdValues;

#Max-min scaling
#arg1 = maxValues; arg2 = minValues;

#Normalising the dataframe
X = normalise_df(X, arg1, arg2)


#Here the y values, or labels are turned from 0,1,2 into
# one hot encoded values (1,0,0),(0,1,0),(0,0,1)
ohe_y = tf.keras.utils.to_categorical(y)

### Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,ohe_y,test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(f'The number of input variables is {n_features}')


model = Sequential()
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=50,batch_size=50,validation_data=(X_test,y_test),verbose=1, shuffle = True)
                 
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


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='upper left')
plt.savefig(pp, format='pdf')
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')



plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig(pp, format='pdf')
plt.close()

print("Training is done!")


#pp.close()
#print(f'All done. Output is saved as {outputname}')


##### Printing the weights
#model_weights = model.layers[0].get_weights()[0]
#print("Model Weights: ", model_weights)



##### Code for Testing #####

testing_df = [VLL_m500_2, VLL_m1000_2, VLL_m1500_2]
df_names = ['M500', 'M1000', 'M1500']
mycols=['xkcd:sky blue','xkcd:red','xkcd:green']
labelnames = ['M500neuron','M1000neuron','M1500neuron']
mybins = np.arange(0,1.05,0.05)

#Normalising the testing DFs
norm_dfs = [normalise_df(df, arg1, arg2) for df in testing_df]



#Will have NN_scores for the M500, M1000 and M1500 datasets
NNscores_list = [model.predict(df) for df in norm_dfs]


#### Plottig the scores
plot_process_scores(NNscores_list[0], 'procM500')
plot_process_scores(NNscores_list[1], 'procM1000')
plot_process_scores(NNscores_list[2], 'procM1500')

plot_neuron_scores(NNscores_list, df_names, 0)
plot_neuron_scores(NNscores_list, df_names, 1)
plot_neuron_scores(NNscores_list, df_names, 2)


pp.close()


#### Plotting is done


#### Calculating the different metrics #### 

M500_proc_scores = NNscores_list[0]
M1000_proc_scores = NNscores_list[1]
M1500_proc_scores = NNscores_list[2]

#### Predict score for the 750 mass point as well:
norm_M750 = normalise_df(VLL_m750_2, arg1, arg2)
M750_proc_scores  = model.predict(norm_M750)

#### Finding mean score
M500_proc_scores_avg =  M500_proc_scores.mean(axis = 0)
M750_proc_scores_avg =  M750_proc_scores.mean(axis = 0)
M1000_proc_scores_avg =  M1000_proc_scores.mean(axis = 0)
M1500_proc_scores_avg =  M1500_proc_scores.mean(axis = 0)




print("Avg of scores for M500 proc: ", M500_proc_scores_avg)
print("Avg of scores for M1000 proc: ", M1000_proc_scores_avg)
print("Avg of scores for M1500 proc: ", M1500_proc_scores_avg)
print()
print("Avg of scores for M750 proc: ", M750_proc_scores_avg)
print("*"*30)
print()

M500_pred = M500_proc_scores_avg[0] * 500 + M500_proc_scores_avg[1]*1000 + M500_proc_scores_avg[2]*1500
M1000_pred = M1000_proc_scores_avg[0] * 500 + M1000_proc_scores_avg[1]*1000 + M1000_proc_scores_avg[2]*1500
M1500_pred = M1500_proc_scores_avg[0] * 500 + M1500_proc_scores_avg[1]*1000 + M1500_proc_scores_avg[2]*1500

M750_pred = M750_proc_scores_avg[0] * 500 + M750_proc_scores_avg[1]*1000 + M750_proc_scores_avg[2]*1500

print("Weighted prediction of mass for proc 500: ", M500_pred)
print("Weighted prediction of mass for proc 1000: ", M1000_pred)
print("Weighted prediction of mass for proc 1500: ", M1500_pred)
print()
print("Weighted prediction of mass for proc 750: ", M750_pred)

my_metric = (M500_proc_scores_avg[0] - M500_proc_scores_avg[1] - M500_proc_scores_avg[2]
             + M1000_proc_scores_avg[1] - M1000_proc_scores_avg[0] - M1000_proc_scores_avg[2]
             + M1500_proc_scores_avg[2] - M1500_proc_scores_avg[0] - M1500_proc_scores_avg[1])/3
print("My metric:", my_metric)


#### Calculating the confusion matrix ####
for i in range(0,3):
    norm_dfs[i]['Original_Class'] = i
    norm_dfs[i]['Predicted_Class'] = NNscores_list[i].argmax(axis=1)

full_testdf = pd.concat(norm_dfs)
mycon = confusion_matrix(full_testdf['Original_Class'], full_testdf['Predicted_Class'])


print("*"*30+"\n")

print("Confusion matrix:")
print(mycon)


acc = mycon.trace()/mycon.sum()
bal_acc = sum([mycon[i,i]/mycon.sum(axis = 1)[i] for i in range(mycon.shape[0])])/3

precisions = [mycon[i,i]/mycon.sum(axis = 0)[i] for i in range(mycon.shape[0])]
recalls = [mycon[i,i]/mycon.sum(axis = 1)[i] for i in range(mycon.shape[0])]

avg_prec = sum(precisions)/3
avg_rec = sum(recalls)/3

macro_f1_score = 2/(1/avg_prec + 1/avg_rec)


print("Accuracy:", acc, "Balanced Accuracy:", bal_acc)
print("Precisions: ", precisions, "Avg = ", avg_prec)
print("Recalls: ", recalls, "Avg = ", avg_rec)
print("Macro F1 score: ", macro_f1_score)


with open(textfilename, "w") as txtfile:
    txtfile.write(f"Avg of scores for M500 proc: {M500_proc_scores_avg}\n")
    txtfile.write(f"Avg of scores for M1000 proc: {M1000_proc_scores_avg}\n")
    txtfile.write(f"Avg of scores for M1500 proc: {M1500_proc_scores_avg}\n\n")

    txtfile.write(f"Avg of scores for M750 proc: {M750_proc_scores_avg}\n")

    txtfile.write("*"*30+"\n\n")

    
    txtfile.write(f"Weighted prediction of mass for proc 500: {M500_pred}\n")
    txtfile.write(f"Weighted prediction of mass for proc 1000: {M1000_pred}\n")
    txtfile.write(f"Weighted prediction of mass for proc 1500: {M1500_pred}\n\n")
    txtfile.write(f"Weighted prediction of mass for proc 750: {M750_pred}\n")
    txtfile.write(f"My metric: {my_metric}\n")
    txtfile.write("*"*30+"\n\n")


    txtfile.write("Confusion matrix: \n")
    txtfile.write(str(mycon)+"\n\n")
    txtfile.write(f"Accuracy:{ acc} Balanced Accuracy: {bal_acc}\n")
    txtfile.write(f"Precisions:{precisions} Avg: {avg_prec}\n")
    txtfile.write(f"Recalls:{recalls} Avg: {avg_rec}\n")
    txtfile.write(f"Macro F1 score:{macro_f1_score}\n")


print(f'All done. Output is saved as {outputname} and {textfilename}')
