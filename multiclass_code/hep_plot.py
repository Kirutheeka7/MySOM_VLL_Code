#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings('ignore')

outputname = 'hep_classify_output_var_3L.pdf'
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

#For 2L1T
col_names=['LT_2L1T', 'HT_2L1T', 'ngenjet', 'j0pT', 'missinget', 'ST_2L1T', 'LTHT', 'LTMET', 'dilepMass_2l1T', 'mT_2l1t_0', 'mT_2l1t_1', 'mtT_2l1t', 'dPhi_l0l1', 'dPhi_l0T', 'dPhi_l1T', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l0T', 'dR_l1T', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_TMET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'dPhi_Tj0', 'pT0_2l1t', 'pT1_2l1t', 'pT_tau_2l1t']
cols = list(range(len(col_names)))

#For 3L
col_names = ['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l']
cols = list(range(len(col_names)))

VLL_m500 = pd.read_csv('../VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m750 = pd.read_csv('../VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
VLL_m1000 = pd.read_csv('../VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_0.txt',sep=' ',index_col=None, usecols=cols,names=col_names)



def plotVars(mybins,plotnames, ylog_scale = False):
    for pname in plotnames:
        plt.figure(figsize=(8,8))
        plt.hist(VLL_m500[pname],bins=mybins,histtype='step',label="VLL_m500",linewidth=3, color='blue',density=False,log=ylog_scale)
        plt.hist(VLL_m750[pname],bins=mybins,histtype='step',label="VLL_m750",linewidth=3, color='red',density=False,log=ylog_scale)
        plt.hist(VLL_m1000[pname],bins=mybins,histtype='step',label="VLL_m1000",linewidth=3, color='green',density=False,log=ylog_scale)
        plt.legend(loc='upper center')
        plt.xlabel(pname,fontsize=20)
        plt.ylabel('Entries',fontsize=20)
        if ylog_scale: plt.ylim([0.01,500_000])
        
        plt.title(pname,fontsize=20)
        plt.savefig(pp,format='pdf')
        plt.close()


plotnames=['LT_3L', 'HT_3L', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l']
mybins = np.arange(0,1000,20)
plotVars(mybins,plotnames)

plotnames=['dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met']
mybins = np.arange(0,3.2,0.1)
plotVars(mybins,plotnames)

plotnames=['ngenjet']
mybins = np.arange(0,10,1)
plotVars(mybins,plotnames)

'''
plotnames=['LT_2L1T', 'HT_2L1T', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2',]
mybins = np.arange(0,1000,20)
plotVars(mybins,plotnames)

plotnames=['dPhi_l0l1', 'dPhi_l0T', 'dPhi_l1T', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l0T', 'dR_l1T', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_TMET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'dPhi_Tj0']
mybins = np.arange(0,3.2,0.1)
plotVars(mybins,plotnames)

plotnames=['ngenjet']
mybins = np.arange(0,10,1)
plotVars(mybins,plotnames)
'''


'''

plt.figure(figsize=(8,8))
#plt.hist(VLL_m500[pname],bins=mybins,histtype='step',label="WZ",linewidth=3, color='blue',density=False,log=False)
#plt.hist(VLL_m750[pname],bins=mybins,histtype='step',label="ZZ",linewidth=3, color='red',density=False,log=False)
plt.hist2d(VLL_m500['MinDphi_LL'], VLL_m500['MaxDphi_LL'], bins = (20,20))
#plt.legend(loc='upper center')
plt.xlabel('MinDphiLL',fontsize=20)
plt.ylabel('MaxDphiLL',fontsize=20)
plt.title("WZ 2d hist",fontsize=20)
plt.savefig(pp,format='pdf')
plt.close()


plt.figure(figsize=(8,8))
#plt.hist(VLL_m500[pname],bins=mybins,histtype='step',label="WZ",linewidth=3, color='blue',density=False,log=False)
#plt.hist(VLL_m750[pname],bins=mybins,histtype='step',label="ZZ",linewidth=3, color='red',density=False,log=False)
plt.hist2d(VLL_m750['MinDphi_LL'], VLL_m750['MaxDphi_LL'], bins = (20,20))
#plt.legend(loc='upper center')
plt.xlabel('MinDphiLL',fontsize=20)
plt.ylabel('MaxDphiLL',fontsize=20)
plt.title("ZZ 2d hist",fontsize=20)
plt.savefig(pp,format='pdf')
plt.close()
'''

pp.close()
