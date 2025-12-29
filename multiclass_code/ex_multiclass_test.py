#!/usr/bin/env python
# coding: utf-8

#
# Run this code as follows
#
# ex_multiclass_test.py <MODELNAME> <PDFNAME>
#
# where <MODELNAME> is the name of input model name (say my_model.h5)
# and <PDFNAME> is the name of the output PDF file with plots (say output.pdf)
#

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

#modelname = sys.argv[1]
#outputname = sys.argv[2]

modelname = "../ML_Output/3LTest_norm-5to5/multiclass_vll_3l.h5"
outputname = "../ML_Output/3LTest_norm-5to5/multiclass_vll_3l_testing.pdf"

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

#First we put in the same columns here that we have in the mynn.py

col_names=['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l'];

cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]





VLL_m500 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

#Reducing to 1000
#VLL_m500 = VLL_m500[:1000]
#VLL_m750 = VLL_m750[:1000]
#VLL_m1000 = VLL_m1000[:1000]


alldfs = [VLL_m500,VLL_m750,VLL_m1000]
alldfnames = ['M500','M750','M1000']

#=====================================
#Now we must declare arrays to keep the max and min values that we got from ex_multiclass.py
#maxVals = np.array([[2.03079e+03, 1.13610e+03, 4.53007e+02, 4.00000e+00, 1.27904e+03, 3.14159e+00,
#                      3.14158e+00, 2.09007e+00, 2.28011e+03, 2.30414e+03, 1.23756e+03, 7.52082e+02]])
#minVals = np.array([[10.3434,    10.0244,    10.0001,     0.,         0.0885814,  0.0112305,
#                     0.,         0.,         0.0437537,  0.,         0.,         0.       ]])

'''
maxVals = np.array([[1360.896973, 1184.820679,  458.599152,    5.,       1613.865112,    3.141593,
                     3.141584,    2.090327, 1366.192383, 2078.448242, 2033.94812,  1022.813843]])
minVals = np.array([[2.9000189e+01,  1.0023091e+01,  1.0000089e+01,  0.0000000e+00,
                     5.7219000e-02,  4.4246000e-02,  1.6981000e-02,  0.0000000e+00,
                     -1.0000000e+00,  7.3600000e-04,  1.9700000e-04,  1.8100000e-04]])
'''

maxVals = np.array([[3853.49, 4858.7, 13.0, 2239.86, 2600.15, 8457.11, 8035.31, 4756.4, 3815.36, 4455.1]])
minVals = np.array([[77.0744, 37.4362, 1.0, 37.4362, 1.00582, 188.919, 125.084, 127.837, 0.0, 74.8925]])

meanVals = np.array([[875.9111615503876, 1209.6366846511628, 4.716327519379845, 499.2981335319767, 270.24741829118216, 2355.7952771802325, 2085.5478483042634, 1146.1585742248062, 461.2242037306202, 937.8816425968993]])
stdVals = np.array([[396.10768659831433, 536.2110115807294, 1.2795366474884335, 230.38747025602152, 207.5140772531654, 925.4488594207648, 894.8704288463225, 457.67349631934815, 385.30691898646177, 469.6052755141552]])

mymodel = tf.keras.models.load_model(modelname)
mymodel.load_weights(modelname)



mybins = np.arange(0,1.05,0.05)

#This function takes a dataframe and plots all three scores for it
def processdf(df,dfname):
    #normeddf = 2*( (df-minVals)/(maxVals-minVals) ) -1.0
    #df = normeddf

    ### New Normalisation:
    normedX = (df-meanVals)/stdVals # Normalising the values
    #boolarr = abs((df-meanVals))/stdVals < 5   # Creating Boolean array of whether the values are within the given range 
    #normedX = df[boolarr.sum(axis = 1) == boolarr.shape[1]] # Selecting only events which have all the values within 5 sigma
    df = normedX


    nnscore = mymodel.predict(df)
    plt.figure(figsize=(8,6))
    plt.hist(nnscore[:,0],bins=mybins,histtype='step',label='M500neuron',linewidth=3,color='xkcd:sky blue',density=False,log=False)
    plt.hist(nnscore[:,1],bins=mybins,histtype='step',label='M750neuron',linewidth=3,color='xkcd:red',density=False,log=False)
    plt.hist(nnscore[:,2],bins=mybins,histtype='step',label='M1000neuron',linewidth=3,color='xkcd:green',density=False,log=False)
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'NN Output for {dfname}',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    #plt.ylim([1, 1.1e5])
    #Save to file instead of individual image
    #plt.savefig('score'+dfname+'.png')
    plt.savefig(pp, format='pdf')
    plt.close()

#This function takes a list of dataframes and plots a specific score for it
mycols=['xkcd:sky blue','xkcd:red','xkcd:green']
labelnames = ['M500neuron','M750neuron','M1000neuron']
def processdflist(dflist,dfnames,proclabel):

    plt.figure(figsize=(8,6))
    for index, df in enumerate(dflist):
        #normeddf = 2*( (df-minVals)/(maxVals-minVals) ) -1.0
        #df = normeddf

        ### New Normalisation:
        normedX = (df-meanVals)/stdVals # Normalising the values
        #boolarr = abs((df-meanVals))/stdVals < 5   # Creating Boolean array of whether the values are within the given range 
        #normedX = df[boolarr.sum(axis = 1) == boolarr.shape[1]] # Selecting only events which have all the values within 5 sigma
        df = normedX
     
        nnscore = mymodel.predict(df)
        plt.hist(nnscore[:,proclabel],bins=mybins,histtype='step',label=dfnames[index],
                 linewidth=3,color=mycols[index],density=False,log=False)
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'{labelnames[proclabel]} NN Output',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    #plt.ylim([1, 1.1e5])
    #plt.savefig('score'+labelnames[proclabel]+'.png')
    plt.savefig(pp, format='pdf')    
    plt.close()
    
    
processdf(VLL_m500,'procM500')
processdf(VLL_m750,'procM750')
processdf(VLL_m1000,'procM1000')
processdflist(alldfs,alldfnames,0)
processdflist(alldfs,alldfnames,1)
processdflist(alldfs,alldfnames,2)


pp.close()
print(f'All done. Output is saved as {outputname}')
