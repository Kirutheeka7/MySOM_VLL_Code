col_names=['Pt0','Pt1','Pt2','Pt3','NBJet','MET','LQSum','LepMt0','LepMt1','LepMt2','LepMt3','HT','LLBestMOSSF','MaxDphi_LMet','MinDphi_LMet','MaxDphi_LL','MinDphi_LL','LMOSOF','TBestMOSOF','TBestMOSSF','LLPairPt','LLSSSFMass','LLSSOFMass','TMSSOF','Mt0','Mt1','Mt2','LDrmin','LMass','L3BelowZ','L3OnZ','L3AboveZ','L3SS','L4DoubleOnZ','L4SingleOnZ','L4OffZ','L2T1BelowZ','L2T1OnZ','L2T1AboveZ','L2T1SS','L1T2OSLowMT','L1T2OSHighMT','L1T2SS','Rare3L1TOnZ','Rare3L1TOffZ','Rare2L2TOnZ','Rare2L2TOffZ','Rare1L3TOffZ'];

myvars = ['Pt0',
'MET',
'LLBestMOSSF',
'MaxDphi_LMet',
'LMOSOF',
'LLSSSFMass',
'LLSSOFMass',
'LDrmin',
'LMass']


##### Fur VLLs
col_names=['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'dR_j0met', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l'];

myvars = ['LT_3L', 'HT_3L', 'ngenjet', 'j0pT', 'missinget', 'ST_3L', 'LTHT', 'LTMET', 'bestMOSSF_3L', 'trilepM', 'mt_0', 'mt_1', 'mt_2', 'dPhi_l0l1', 'dPhi_l1l2', 'dPhi_l0l2', 'maxdPhi', 'mindPhi', 'dR_l0l1', 'dR_l1l2', 'dR_l0l2', 'dRmin', 'dRmax', 'dPhi_l0MET', 'dPhi_l1MET', 'dPhi_l2MET', 'maxdPhi_LMET', 'mindPhi_LMET', 'dPhi_l0j0', 'dPhi_j0met', 'dR_l0j0', 'pT0_3l', 'pT1_3l', 'pT2_3l', 'pT_tau_3l']


#myvars = ['Pt2', 'Pt3']

ind = []
for i in range(len(myvars)):
    ind.append(col_names.index(myvars[i]))

print(ind)
