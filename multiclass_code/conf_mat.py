#Import the necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc, confusion_matrix

import os
import sys
import warnings
warnings.filterwarnings('ignore')

modelname = "../ML_Output/3LTest_norm-5to5/multiclass_vll_3l.h5"#input("Please enter full path of model name: ")
outputname = "../ML_Output/3LTest_norm-5to5/confusion_mat_vll_3l.txt"#input("Please enter full path of output pdf: ")#


col_names=['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l']
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

VLL_m500 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

alldfs = [VLL_m500,VLL_m750,VLL_m1000]
alldfnames = ['M500','M750','M1000']

maxVals = np.array([[3853.49, 4858.7, 13.0, 2239.86, 2600.15, 8457.11, 8035.31, 4756.4, 3815.36, 4455.1]])
minVals = np.array([[77.0744, 37.4362, 1.0, 37.4362, 1.00582, 188.919, 125.084, 127.837, 0.0, 74.8925]])


meanVals = np.array([[875.9111615503876, 1209.6366846511628, 4.716327519379845, 499.2981335319767, 270.24741829118216, 2355.7952771802325, 2085.5478483042634, 1146.1585742248062, 461.2242037306202, 937.8816425968993]])
stdVals = np.array([[396.10768659831433, 536.2110115807294, 1.2795366474884335, 230.38747025602152, 207.5140772531654, 925.4488594207648, 894.8704288463225, 457.67349631934815, 385.30691898646177, 469.6052755141552]])

                    
mymodel = tf.keras.models.load_model(modelname)
mymodel.load_weights(modelname)


mybins = np.arange(0,1.05,0.05)
'''
norm_M500 = 2*( (VLL_m500-minVals)/(maxVals-minVals) ) -1.0
norm_M750 = 2*( (VLL_m750-minVals)/(maxVals-minVals) ) -1.0
norm_M1000 = 2*( (VLL_m1000-minVals)/(maxVals-minVals) ) -1.0

norm_M500['class'] = 0; norm_M750['class'] = 1; norm_M1000['class'] = 2
'''

norm_M500 = (VLL_m500-meanVals)/stdVals # Normalising the values
norm_M750 = (VLL_m750-meanVals)/stdVals # Normalising the values
norm_M1000 = (VLL_m1000-meanVals)/stdVals # Normalising the values
norm_M500['class'] = 0; norm_M750['class'] = 1; norm_M1000['class'] = 2


fulldf = pd.concat([norm_M500, norm_M750, norm_M1000])
nnscore_all = mymodel.predict(fulldf[list(fulldf.keys()[:-1])])

predicted_class = nnscore_all.argmax(axis = 1)

fulldf['predicted class'] = predicted_class

mycon = confusion_matrix(fulldf['class'], fulldf['predicted class'])




### Calculating and Printing out the metrics of the model:
print("Model name:", modelname)


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

with open(outputname, "w") as txtfile:
    txtfile.write("Confusion matrix: \n")
    txtfile.write(str(mycon)+"\n")
    txtfile.write(f"Accuracy:{ acc} Balanced Accuracy: {bal_acc}\n")
    txtfile.write(f"Precisions:{precisions} Avg: {avg_prec}\n")
    txtfile.write(f"Recalls:{recalls} Avg: {avg_rec}\n")
    txtfile.write(f"Macro F1 score:{macro_f1_score}\n")
