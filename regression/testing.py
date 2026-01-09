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

col_names = ['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'Mos1', 'Mos2', 
           'trilepM', 'mt0', 'mt1', 'mt2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 
           'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 
           'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l', 'l3mt', 'highmT', 'pT0j0pT', 
           'j0pTbyHT', 'pT0byLT']
cols = [0, 1, 9, 10, 11, 38, 39, 40]


outputname = '../ML_Output/regr/newvar_ep20_shuffle_train1200/testing_fulldf.pdf'
modelname = '../ML_Output/regr/newvar_ep20_shuffle_train1200/newvar_model.h5'
pp = PdfPages(outputname)


VLL_m750 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_2.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

mymodel = tf.keras.models.load_model(modelname)
mymodel.load_weights(modelname)

pred_y = mymodel.predict(VLL_m750)

results = pd.DataFrame()

results['Pred_VLLD_Mass'] = pred_y.flatten()
results['True_VLLD_Mass'] = 750

#results = results.head(100)
print(type(pred_y), pred_y.shape)

results['diffsquare'] = results.apply(lambda row: np.square(row.True_VLLD_Mass-row.Pred_VLLD_Mass) , axis=1 )

TotalDiff = results['diffsquare'].sum() / results.shape[0]
print(f"The total difference between expectation and prediction, the MSE = {TotalDiff:.4f}")

print(results.head())

fig, ax = plt.subplots(figsize = (10, 10))

ax.hist(results.diffsquare, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('diffsquare', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Performance plot: Total MSE = {TotalDiff:.4f}', fontsize = 16)
plt.savefig(pp, format = 'pdf')


#### Plotting hists for the testing mass point

fig, ax = plt.subplots(figsize = (10, 10))

pred_750 = results['Pred_VLLD_Mass'] #[results['True_VLLD_Mass']==750]

ax.hist(pred_750, bins = 50, lw = 3, log = False, alpha = 0.5)
ax.set_xlabel('Predicted_VLLD_Mass', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.set_title(f'Pred Mass for VLLD_M = 750', fontsize = 16)

meanmine= round(pred_750.mean(),2)
print(meanmine)
ax.annotate(f"Mean = %.2f\nSD = {round(pred_750.std(),2)}\nRMS Error = %.2f"%(meanmine,TotalDiff), xy = (0.6,0.7), xycoords = 'figure fraction',fontsize=16)
plt.savefig(pp, format = 'pdf')

pp.close()
