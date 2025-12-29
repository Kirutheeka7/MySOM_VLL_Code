#define delZAna_cxx

#include "delZAna.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>



void delZAna::Begin()
{
  //Initialize global variables here.
  GEV = 1000.;
  MEV2GEV = .001;
  nEvtTotal = 0;
  njet=0;
  //Create the histogram file
  _HstFile = new TFile(_HstFileName,"recreate");
  //Call the function to book the histograms we declared in Hists.
  BookHistograms();
  
  outputFile3L.open(_OpFileName3L);
  outputFile2L1T.open(_OpFileName2L1T);
}

void delZAna::End()
{
  //Write histograms and close histogram file
  _HstFile->Write();
  _HstFile->Close();
  //Output to screen.
  cout<<"Total events = "<<nEvtTotal<<endl;
  //Open the text output file
  ofstream fout(_SumFileName);
  //Put text output in the summary file.
  fout<<"Total events = "<<nEvtTotal<<endl;
  outputFile3L.close();
  outputFile2L1T.close();
  
}

void delZAna::Loop()
{
  //
  // This function should contain the "body" of the analysis. It can contain
  // simple or elaborate selection criteria, run algorithms on the data
  // of the event and typically fill histograms.
  //
  
  if(fChain==0) return;
  Long64_t numberOfEntries = treeReader->GetEntries();
  //This prints total entries before processing begins
  cout<<"Total entries = "<<numberOfEntries<<endl;
  nEvtTotal=0;

  //This is the main loop over the number of events
  for(Int_t entry = 0; entry < numberOfEntries; ++entry){

    treeReader->ReadEntry(entry);
    nEvtTotal++;
    //Output processing information to screen based on verbosity level.
    if(_verbosity>5000)cout<<"Processed "<<nEvtTotal<<" event..."<<endl;          
    if(_verbosity>1000 && nEvtTotal%5000==0)cout<<"Processed "<<nEvtTotal<<" event..."<<endl;      
    else if(_verbosity>0 && nEvtTotal%50000==0)cout<<"Processed "<<nEvtTotal<<" event..."<<endl;


    // ..............................CODE STARTS HERE...........................................
 
    //Your CODE starts here
    //MissingET *met = (MissingET*)brMissingET->At(0); //not using this.
    TLorentzVector mymet(0,0,0,0); // build our own met

    //clear array from previous event
    goodEle.clear();
    goodMu.clear(); 
    goodTau.clear();
    jetgen.clear();
    goodLep.clear();
    
    
    int nmuons=0; int nparts=0; int nelec=0; int goodlep=0;

    // Tau builder is a function that builds visible hadronic taus from the Gen collection
    TauBuilder();
    h.ngoodtau->Fill((int)goodTau.size());
    if((int)goodTau.size()>0) h.goodtaupt->Fill(goodTau.at(0).v.Pt());

    // These are genJets. Here we use the Delphes Collection.
    for(int ij=0; ij< brGenJet->GetEntries(); ij++){
      Jet *jt = (Jet*)brGenJet->At(ij);
      Lepton temp; temp.v.SetPtEtaPhiM(jt->PT,jt->Eta,jt->Phi,jt->Mass); temp.ind = ij;
      
      bool PassCuts = temp.v.Pt()>30 && fabs(temp.v.Eta())<2.4; //what is jet_jetId??
      
      if(PassCuts){
	jetgen.push_back(temp);
      }
    }
    Sort(2);//sorting the genjets
    
    TLorentzVector q1,q2;
    q1.SetPtEtaPhiM(0,0,0,0); q2.SetPtEtaPhiM(0,0,0,0);

    //Now we loop over truth particles and keep what we need.
   
    for(int no=0; no<brParticle->GetEntries(); no++){
      GenParticle *par = (GenParticle*)brParticle->At(no);

      // Building my missing et from neutrinos
      if(par->Status==1){ //its a stable particle
	if(abs(par->PID)==12 || abs(par->PID)==14 || abs(par->PID)==16){
	  TLorentzVector neut; neut.SetPtEtaPhiM(par->PT,par->Eta,par->Phi,0);
	  mymet += neut;
	}
      }

      //genELECTRON ARRAY
      if(par->Status==1 && abs(par->PID)==11){ //its a stable electron
	nelec++;
	Lepton temp; temp.v.SetPtEtaPhiM(par->PT, par->Eta, par->Phi, par->Mass);
	temp.id = par->PID, temp.ind=no;
	float iso = calc_iso(0.4, temp.v, no);
	if(temp.v.Pt()>10 && fabs(temp.v.Eta())<2.4 && iso/temp.v.Pt()<0.15){
	  goodEle.push_back(temp);
	}
      }

      
      //genMUON ARRAY
      // Saving muons
      if(par->Status==1 && abs(par->PID)==13){//its a stable muon
	nmuons++;
	Lepton temp; temp.v.SetPtEtaPhiM(par->PT,par->Eta,par->Phi,par->Mass);
	temp.id = par->PID; temp.ind=no;
	float isol = calc_iso(0.4,temp.v,no);
	if(temp.v.Pt()>10 && fabs(temp.v.Eta())<2.4 && isol/temp.v.Pt()<0.15){//check if this candidate passes selections.
	  goodMu.push_back(temp); //store it in the goodMu array.
	}
      }


      //goodLep array (includes e and mu)
      if(par->Status==1 && (abs(par->PID)==11 || abs(par->PID)==13)){
	goodlep++;
	Lepton temp; temp.v.SetPtEtaPhiM(par->PT, par->Eta, par->Phi, par->Mass);
	temp.id = par->PID;
	temp.ind = no;
	float iso = calc_iso(0.4, temp.v,no);
	if(temp.v.Pt()>10 && fabs(temp.v.Eta())<2.4 && iso/temp.v.Pt()<0.15){
	  goodLep.push_back(temp);
	}
      }
    } //Loop over truth particles ends
     
    Sort(1); //sort the goodMu,goodEle and goodTau array descending by pT.
      
    float missinget = mymet.Pt();
    float metphi = mymet.Phi();
    h.etmiss->Fill(missinget);//fill a histogram with the missing Et of the event.
      
    if((int)goodMu.size()>1){
      h.massjj[1]->Fill( (goodMu.at(0).v+goodMu.at(1).v).M() );
    }

    if((int)goodEle.size()>1){
      h.massEE->Fill((goodEle.at(0).v + goodEle.at(1).v).M());
    }

    //Sum of jet pts:
    float ht=0;
    if((int)jetgen.size()>=0){
      for(unsigned int i=0; i<jetgen.size();i++){
	ht = ht + jetgen.at(i).v.Pt();
      }
      h.HT->Fill(ht);
      h.ngenjets->Fill(jetgen.size());
    }

    h.ngoodele->Fill((int)goodEle.size());
    if((int)goodEle.size()>0){
      h.goodelept->Fill(goodEle.at(0).v.Pt()); //Leading electron pt
    }

    
    if((int)goodLep.size()>0 && goodLep.at(0).v.Pt()>30){
      
      // -------------------------------- 3L selection (>=3L + >=0 T (tau)) ------------------------------------------------

      int cat_3l = -1; int onZ_3l=-1;

      if((int)goodLep.size()>=3 && (int)goodTau.size()>=0 && missinget>1){ //if 3L

	//category counts
	if(abs(goodLep.at(0).id)==13 && abs(goodLep.at(1).id)==13 && abs(goodLep.at(2).id)==13){
	  cat_3l=0; //mmm
	}
	if(abs(goodLep.at(0).id)==13 && abs(goodLep.at(1).id)==13 && abs(goodLep.at(2).id)==11){
	  cat_3l=1; //mme
	}
	if(abs(goodLep.at(0).id)==13 && abs(goodLep.at(1).id)==11 && abs(goodLep.at(2).id)==11){
	  cat_3l=2; //mee
	}
	if(abs(goodLep.at(0).id)==11 && abs(goodLep.at(1).id)==11 && abs(goodLep.at(2).id)==11){
	  cat_3l=3; //eee
	}
	if(abs(goodLep.at(0).id)==11 && abs(goodLep.at(1).id)==11 && abs(goodLep.at(2).id)==13){
	  cat_3l=4; //eem
	}
	if(abs(goodLep.at(0).id)==11 && abs(goodLep.at(1).id)==13 && abs(goodLep.at(2).id)==13){
	  cat_3l=5; //emm
	}
	if(abs(goodLep.at(0).id)==13 && abs(goodLep.at(1).id)==11 && abs(goodLep.at(2).id)==13){
	  cat_3l=6; //mem
	}
	if(abs(goodLep.at(0).id)==11 && abs(goodLep.at(1).id)==13 && abs(goodLep.at(2).id)==11){
	  cat_3l=7; //eme
	}
	h.cat_3l->Fill(cat_3l);

	//LT_3L
	//In multilep paper, LT is defined to be the scalar pT sum of all charged leptons than constitute the channel
	float LT_3L=0;
	if((int)goodTau.size()>0){
	  LT_3L = goodLep.at(0).v.Pt() + goodLep.at(1).v.Pt() + goodLep.at(2).v.Pt() + goodTau.at(0).v.Pt();
	}
	else{
	  LT_3L = goodLep.at(0).v.Pt() + goodLep.at(1).v.Pt() + goodLep.at(2).v.Pt();
	}
	h.lt_3l->Fill(LT_3L);

	//HT_3L
	float HT_3L=0;
	if((int)jetgen.size()>=0){
	  for(unsigned int i=0; i<jetgen.size(); i++){
	    HT_3L = HT_3L + jetgen.at(i).v.Pt();
	  }
	  h.ht_3l->Fill(HT_3L);
	  h.njets_3l->Fill((int)jetgen.size()); //number of jets
	}
	//leading jet pT
	float j0pT = 0;
	if((int)jetgen.size()>0){
	  j0pT = jetgen.at(0).v.Pt();
	}
	h.j0pT_3L->Fill(j0pT);

	//MET
	h.MET_3L->Fill(missinget);

	//ST
	float ST_3L = LT_3L+HT_3L+missinget;
	h.st_3L->Fill(ST_3L);

	//LT+HT
	float LTHT = LT_3L + HT_3L;
	h.ltht_3L->Fill(LTHT);

	//LT+MET
	float LTMET = LT_3L + missinget;
	h.ltmet_3L->Fill(LTMET);
	
	//BestM_OSSF
        float bestMOSSF_3L=0;
	for(unsigned int i=0; i<3; i++){
	  for(unsigned int j=0; j<3; j++){
	    if(i!=j){
	      if((fabs(goodLep.at(i).id) == fabs(goodLep.at(j).id)) && (goodLep.at(i).id * goodLep.at(j).id < 0)){
		bestMOSSF_3L = (goodLep.at(i).v + goodLep.at(j).v).M();
		break;
	      }
	    }
	  }
	}
	h.bestMass_3l->Fill(bestMOSSF_3L);


	//TriLepMass
	float trilepM = (goodLep.at(0).v + goodLep.at(1).v + goodLep.at(2).v).M();
	h.triLepM_3L->Fill(trilepM);

	//Transverse Mass
	float mt[3];
	for(unsigned int i=0; i<3; i++){
	  mt[i] = {transv_mass(goodLep.at(i).v.Pt(), missinget, delta_phi(goodLep.at(i).v.Phi(), metphi))};
	  h.mT_3L[i]->Fill(mt[i]);
	}
	if((int)goodTau.size()>0){
	  float mtT_3l=transv_mass(goodTau.at(0).v.Pt(), missinget, delta_phi(goodLep.at(0).v.Phi(), metphi));
	  h.mtT_3L->Fill(mtT_3l);
	}
	
	//dPhi's
	float dPhi_l0l1 = goodLep.at(0).v.DeltaPhi(goodLep.at(1).v);
	float dPhi_l1l2 = goodLep.at(1).v.DeltaPhi(goodLep.at(2).v);
	float dPhi_l0l2 = goodLep.at(0).v.DeltaPhi(goodLep.at(2).v);
	h.dPhi_l0l1_3L->Fill(dPhi_l0l1);
	h.dPhi_l1l2_3L->Fill(dPhi_l1l2);
	h.dPhi_l0l2_3L->Fill(dPhi_l0l2);

	//maxdPhi out of above 3
	float maxdPhi = maxfloat(dPhi_l0l1, dPhi_l1l2, dPhi_l0l2);
	h.maxdPhi_3L->Fill(maxdPhi);
	//mindPhi out of the above 3
	float mindPhi = minfloat(dPhi_l0l1, dPhi_l1l2, dPhi_l0l2);
	h.mindPhi_3L->Fill(mindPhi);
	
	//dR's
	float dR_l0l1 = goodLep.at(0).v.DeltaR(goodLep.at(1).v);
	float dR_l1l2 = goodLep.at(1).v.DeltaR(goodLep.at(2).v);
	float dR_l0l2 = goodLep.at(0).v.DeltaR(goodLep.at(2).v);
	h.dR_l0l1_3L->Fill(dR_l0l1);
	h.dR_l1l2_3L->Fill(dR_l1l2);
	h.dR_l0l2_3L->Fill(dR_l0l2);
	
	//dRMin out of the above 3    
	float dRmin = minfloat(dR_l0l1, dR_l1l2, dR_l0l2);
	h.dRmin_3L->Fill(dRmin);
	//dRmax out of the above 3
	float dRmax = maxfloat(dR_l0l1, dR_l1l2, dR_l0l2);
	h.dRmax_3L->Fill(dRmax);

	//dPhi's between L and MET
	float dPhi_l0MET = delta_phi(goodLep.at(0).v.Phi(), metphi);
	float dPhi_l1MET = delta_phi(goodLep.at(1).v.Phi(), metphi);
	float dPhi_l2MET = delta_phi(goodLep.at(2).v.Phi(), metphi);
	h.dPhi_l0MET_3L->Fill(dPhi_l0MET);
	h.dPhi_l1MET_3L->Fill(dPhi_l1MET);
	h.dPhi_l2MET_3L->Fill(dPhi_l2MET);
	//maxdPhi_LMET
	float maxdPhi_LMET = maxfloat(dPhi_l0MET, dPhi_l1MET, dPhi_l2MET);
	h.maxdPhi_LMET_3L->Fill(maxdPhi_LMET);
	float mindPhi_LMET = minfloat(dPhi_l0MET, dPhi_l1MET, dPhi_l2MET);
	h.mindPhi_LMET_3L->Fill(mindPhi_LMET);

	//dPhi and dR between leading jet and leading lepton
	//dPhi and dR between leading jet and met
	float dPhi_l0j0 = 0; float dPhi_j0met = 0;
	float dR_l0j0 = 0; float dR_j0met = 0;
	if((int)jetgen.size()>0){
	  dPhi_l0j0 = delta_phi(goodLep.at(0).v.Phi(), jetgen.at(0).v.Phi());
	  dPhi_j0met = delta_phi(jetgen.at(0).v.Phi(), metphi);
	  dR_l0j0 = goodLep.at(0).v.DeltaR(jetgen.at(0).v);
	  //dR_j0met = jetgen.at(0).v.DeltaR(mymet.Pt());
	}
	h.dPhi_l0j0_3L->Fill(dPhi_l0j0);
	h.dPhi_j0met_3L->Fill(dPhi_j0met);
	h.dR_l0j0_3L->Fill(dR_l0j0);
	h.dR_j0met_3L->Fill(dR_j0met);
	
	//pT's of leptons
	float pT0_3l = goodLep.at(0).v.Pt();
	float pT1_3l = goodLep.at(1).v.Pt();
	float pT2_3l = goodLep.at(2).v.Pt();
	h.pT0_3L->Fill(pT0_3l);
	h.pT1_3L->Fill(pT1_3l);
	h.pT2_3L->Fill(pT2_3l);

	float pT_tau_3l = 0;
	if((int)goodTau.size()>0){
	  pT_tau_3l = goodTau.at(0).v.Pt();
	  h.pT_tau_3L->Fill(pT_tau_3l);
	}
 
	//Write in textfile
	 outputFile3L << LT_3L << " " << HT_3L << " " << ((int)jetgen.size()) <<" " << j0pT << " " << missinget << " " << ST_3L << " " << LTHT << " " << LTMET << " " << bestMOSSF_3L << " " << trilepM << " " << mt[0] << " " << mt[1] << " " << mt[2] << " " << dPhi_l0l1 << " " << dPhi_l1l2 << " " << dPhi_l0l2 << " " << maxdPhi << " " << mindPhi << " " << dR_l0l1 << " " << dR_l1l2 << " " << dR_l0l2 << " " << dRmin << " " << dRmax << " " << dPhi_l0MET << " " << dPhi_l1MET << " " << dPhi_l2MET << " " << maxdPhi_LMET << " " << mindPhi_LMET << " " << dPhi_l0j0 << " " << dPhi_j0met << " " << dR_l0j0 << " " << dR_j0met << " " << pT0_3l << " " << pT1_3l << " " << pT2_3l << " " << pT_tau_3l << endl;
	
	/*
	MOSSF conditons:
	1) same flavor leptons either both should be e or mu
	2) they should be opp sign, i.e the multiplication of their Pid's should be -ve
	(if the above is true, compute invariant mass)
	3) the dilep mass should be between 76 and 106
	*/
	
	
	
      } //if 3L loop ends
      
      
      // ------------------------------- 2L1T selection (=2T + >=1 T (tau)) ------------------------------------------------

      int cat_2l1T = -1; int onZ_2l1T = -1;

      if((int)goodLep.size()==2 && (int)goodTau.size()>=1 && missinget>1){ //if 2L1T

	//category counts
	if(abs(goodLep.at(0).id)==13 && abs(goodLep.at(1).id)==13){
	  cat_2l1T=0; //mm
	}
	if(abs(goodLep.at(0).id)==13 && abs(goodLep.at(1).id)==11){
	  cat_2l1T=1; //me
	}
	if(abs(goodLep.at(0).id)==11 && abs(goodLep.at(1).id)==13){
	  cat_2l1T=2; //em
	}
	if(abs(goodLep.at(0).id)==11 && abs(goodLep.at(1).id)==11){
	  cat_2l1T=3; //ee
	}
	h.cat_2l1T->Fill(cat_2l1T);

	//LT_2L1T
	float LT_2L1T = goodLep.at(0).v.Pt() + goodLep.at(1).v.Pt() + goodTau.at(0).v.Pt();
	h.lt_2l1t->Fill(LT_2L1T);

	//HT_2L1T
	float HT_2L1T=0;
	if((int)jetgen.size()>=0){
	  for(unsigned int i=0; i<jetgen.size(); i++){
	    HT_2L1T = HT_2L1T + jetgen.at(i).v.Pt();
	  }
	  h.ht_2l1t->Fill(HT_2L1T);
	  h.njets_2l1t->Fill((int)jetgen.size()); //number of jets
	}
	//leading jet pT
	float j0pT = 0;
	if((int)jetgen.size()>0){
	  j0pT = jetgen.at(0).v.Pt();
	}
	h.j0pT_2L1T->Fill(j0pT);
	  
	
	//MET
	h.MET_2L1T->Fill(missinget);

	//ST_2L1T
	float ST_2L1T = LT_2L1T+HT_2L1T+missinget;
	h.st_2L1T->Fill(ST_2L1T);

	//LT+HT
	float LTHT = LT_2L1T + HT_2L1T;
	h.ltht_2L1T->Fill(LTHT);

	//LT+MET
	float LTMET = LT_2L1T + missinget;
	h.ltmet_2L1T->Fill(LTMET);
	
	//Dilep mass
	float dilepMass_2l1T=0;
	if((fabs(goodLep.at(0).id) == fabs(goodLep.at(1).id)) && (goodLep.at(0).id*goodLep.at(1).id<0)){
	  dilepMass_2l1T = (goodLep.at(0).v + goodLep.at(1).v).M();
	  h.dilepMass_2l1t->Fill(dilepMass_2l1T);
	}

	//BestM_OSSF

	//triLepMass
	float triLepM = (goodLep.at(0).v + goodLep.at(1).v + goodTau.at(0).v).M();
	h.triLepM_2L1T->Fill(triLepM);
	
	//Transverse Mass
	float mT_2l1t[2];
	for(unsigned int i=0; i<2; i++){
	  mT_2l1t[i] = {transv_mass(goodLep.at(i).v.Pt(), missinget, delta_phi(goodLep.at(i).v.Phi(), metphi))};
	  h.mT_2L1T[i]->Fill(mT_2l1t[i]);
	}
	float mtT_2l1t = transv_mass(goodTau.at(0).v.Pt(), missinget, delta_phi(goodTau.at(0).v.Phi(), metphi));
	h.mtT_2L1T->Fill(mtT_2l1t);

	//dPhi's
	float dPhi_l0l1 = goodLep.at(0).v.DeltaPhi(goodLep.at(1).v);
	float dPhi_l0T = goodLep.at(0).v.DeltaPhi(goodTau.at(0).v);
	float dPhi_l1T = goodLep.at(1).v.DeltaPhi(goodTau.at(0).v);
	h.dPhi_l0l1_2L1T->Fill(dPhi_l0l1);
	h.dPhi_l0T_2L1T->Fill(dPhi_l0T);
	h.dPhi_l1T_2L1T->Fill(dPhi_l1T);

	//maxdPhi out of above 3
	float maxdPhi = maxfloat(dPhi_l0l1, dPhi_l0T, dPhi_l1T);
	h.maxdPhi_2L1T->Fill(maxdPhi);
	//mindPhi out of the above 3
	float mindPhi = minfloat(dPhi_l0l1, dPhi_l0T, dPhi_l1T);
	h.mindPhi_2L1T->Fill(mindPhi);
	
	//dR's
	float dR_l0l1 = goodLep.at(0).v.DeltaR(goodLep.at(1).v); //a
	float dR_l0T = goodLep.at(0).v.DeltaR(goodTau.at(0).v); //b
	float dR_l1T = goodLep.at(1).v.DeltaR(goodTau.at(0).v); //c
	h.dR_l0l1_2L1T->Fill(dR_l0l1);
	h.dR_l0T_2L1T->Fill(dR_l0T);
	h.dR_l1T_2L1T->Fill(dR_l1T);

	//dRMin out of the above 3    
	float dRmin = minfloat(dR_l0l1, dR_l0T, dR_l1T);
	h.dRmin_2L1T->Fill(dRmin);
	//dRmax out of the above 3
	float dRmax = maxfloat(dR_l0l1, dR_l0T, dR_l1T);
	h.dRmax_2L1T->Fill(dRmax);

	//dPhi's between L and MET
	float dPhi_l0MET = delta_phi(goodLep.at(0).v.Phi(), metphi);
	float dPhi_l1MET = delta_phi(goodLep.at(1).v.Phi(), metphi);
	float dPhi_TMET = delta_phi(goodTau.at(0).v.Phi(), metphi);
	h.dPhi_l0MET_2L1T->Fill(dPhi_l0MET);
	h.dPhi_l1MET_2L1T->Fill(dPhi_l1MET);
	h.dPhi_TMET_2L1T->Fill(dPhi_TMET);
	//maxdPhi_LMET
	float maxdPhi_LMET = maxfloat(dPhi_l0MET, dPhi_l1MET, dPhi_TMET);
	h.maxdPhi_LMET_2L1T->Fill(maxdPhi_LMET);
	float mindPhi_LMET = minfloat(dPhi_l0MET, dPhi_l1MET, dPhi_TMET);
	h.mindPhi_LMET_2L1T->Fill(mindPhi_LMET);

	//dPhi between leading lep and leading jet
	//dPhi between leading jet and met
	float dPhi_l0j0 = 0; float dPhi_j0met = 0;
	float dR_l0j0 = 0; float dR_j0met = 0;
	if((int)jetgen.size()>0){
	  dPhi_l0j0 = delta_phi(goodLep.at(0).v.Phi(), jetgen.at(0).v.Phi());
	  dPhi_j0met = delta_phi(jetgen.at(0).v.Phi(), metphi);
	  dR_l0j0 = goodLep.at(0).v.DeltaR(jetgen.at(0).v);
	  //dR_j0met = jetgen.at(0).v.DeltaR(mymet.Pt());
	}
	h.dPhi_l0j0_2L1T->Fill(dPhi_l0j0);
	h.dPhi_j0met_2L1T->Fill(dPhi_j0met);
	h.dR_l0j0_2L1T->Fill(dR_l0j0);
	h.dR_j0met_2L1T->Fill(dR_j0met);

	//dPhi betn tau and jet
	float dPhi_Tj0 = 0;
	if((int)jetgen.size()>0){
	  dPhi_Tj0 = delta_phi(goodTau.at(0).v.Phi(), jetgen.at(0).v.Phi());
	}
	h.dPhi_Tj0_2L1T->Fill(dPhi_Tj0);
	
	//pT's of leptons
	float pT0_2l1t = goodLep.at(0).v.Pt();
	float pT1_2l1t = goodLep.at(1).v.Pt();
	float pT_tau_2l1t = goodTau.at(0).v.Pt();
	h.pT0_2L1T->Fill(pT0_2l1t);
	h.pT1_2L1T->Fill(pT1_2l1t);
	h.pT_tau_2L1T->Fill(pT_tau_2l1t);

	  
	//Writing in output file
        outputFile2L1T << LT_2L1T << " " << HT_2L1T << " " << ((int)jetgen.size()) << " " << j0pT << " " << missinget << " " << ST_2L1T << " " << LTHT << " " << LTMET << " " << dilepMass_2l1T << " " << mT_2l1t[0] << " " << mT_2l1t[1] << " " << mtT_2l1t << " " << dPhi_l0l1 << " " << dPhi_l0T << " " << dPhi_l1T << " " << maxdPhi << " " << mindPhi << " " << dR_l0l1 << " " << dR_l0T << " " << dR_l1T << " " << dRmin << " " << dRmax << " " << dPhi_l0MET << " " << dPhi_l1MET << " " << dPhi_TMET << " " << maxdPhi_LMET << " " << mindPhi_LMET << " " << dPhi_l0j0 << " " << dPhi_j0met << " " << dR_l0j0 << " " << dR_j0met << " " << dPhi_Tj0 << " " << pT0_2l1t << " " << pT1_2l1t << " " << pT_tau_2l1t << endl;
	//file_2l1t.close();
      } //if 2L1T loop ends
      
    } //{goodLep.size()>0 && goodLep.at(0).v.Pt()>30} loop ends
      
  } //main loop over number of events end
  
} //void loop ends

void delZAna::Sort(int opt)
{
  //Sort selected objects by pT (always descending).
  //option 1 sorts the gooMu, goodTau, goodEle array
  if(opt==1){
    for(int i=0; i<(int)goodMu.size()-1; i++){
      for(int j=i+1; j<(int)goodMu.size(); j++){
	if( goodMu[i].v.Pt() < goodMu[j].v.Pt() ) swap(goodMu.at(i),goodMu.at(j)); }}
    for(int i=0; i<(int)goodTau.size()-1; i++){
      for(int j=i+1; j<(int)goodTau.size(); j++){
	if( goodTau[i].v.Pt() < goodTau[j].v.Pt() ) swap(goodTau.at(i),goodTau.at(j)); }}
    for(int i=0; i<(int)goodEle.size()-1; i++){
      for(int j=i+1; j<(int)goodEle.size(); j++){
	if( goodEle[i].v.Pt() < goodEle[j].v.Pt() ) swap(goodEle.at(i),goodEle.at(j)); }}
    for(int i=0; i<(int)goodLep.size()-1; i++){
      for(int j=i+1; j<(int)goodLep.size(); j++){
	if( goodLep[i].v.Pt() < goodLep[j].v.Pt() ) swap(goodLep.at(i),goodLep.at(j)); }}
  }
  if(opt==2){
    for(int i=0; i<(int)jetgen.size()-1; i++){
      for(int j=i+1; j<(int)jetgen.size(); j++){
	if( jetgen[i].v.Pt() < jetgen[j].v.Pt() ) swap(jetgen.at(i),jetgen.at(j)); }}
  }
  //Once you have other arrays (goodEle, goodPho), write code here to sort them.
}

//Maximum number
float delZAna::maxfloat(float a, float b, float c)
{
  float maxim = 0;
  if(a>b){ //a<b
    if(a>c){ //a<c
      maxim = a;  //a is the minimum
    }
    else{ //if c<a
      maxim = c; //c in the min
    }
  }
  else if(b>c){ //b<c
    maxim = b;
  }
  else{
    maxim = c;
  }
  return maxim;
}

//minimum number
float delZAna::minfloat(float a, float b, float c)
{
  float minim = 0;
  if(a<b){ //a<b
    if(a<c){ //a<c
      minim = a;  //a is the minimum
    }
    else{ //if c<a
      minim = c; //c in the min
    }
  }
  else if(b<c){ //b<c
    minim = b;
  }
  else{
    minim = c;
  }
  return minim;
}

float delZAna::delta_phi(float phi1, float phi2)
{
  phi1 = TVector2::Phi_0_2pi(phi1);
  phi2 = TVector2::Phi_0_2pi(phi2);
  float dphi = fabs(phi1 - phi2);
  if(dphi>TMath::Pi()) dphi = 2*TMath::Pi() - dphi;
  return dphi;
}

float delZAna::transv_mass(float pT_lep, float MET, float dphi)
{
  //Inputs are pT of lepton, MET and dphi between lepton_pT and MET
  float mT = sqrt(2 * pT_lep * MET * (1-cos(dphi)));
  return mT;
}

Double_t delZAna::delR(TLorentzVector v1,  TLorentzVector v2)
{ 
  double del_eta = abs(v1.Eta() - v2.Eta());
  return sqrt( pow(del_eta,2) + pow(delta_phi(v1.Phi(),v2.Phi()),2) );
}
float delZAna::calc_iso(float dR, TLorentzVector v1, int skip)
{
  // Sum the pT of all stable particles with deltaR<dR of particle v1
  // Skip the particle itself (skip) and skip neutrinos.
  float isol = 0;
  for(int no=0; no<brParticle->GetEntries(); no++){
    GenParticle *par = (GenParticle*)brParticle->At(no);
    if(no==skip) continue;
    if(par->Status==1){
      if(abs(par->PID)==12 || abs(par->PID)==14 || abs(par->PID)==16) continue;
      TLorentzVector v2; v2.SetPtEtaPhiM(par->PT,par->Eta,par->Phi,par->Mass);
      if(v1.DeltaR(v2)<dR) isol+=v2.Pt();
    }
  }
  return isol;

}
void delZAna::TauBuilder()
{
  //First we get the list of taus and keep them in tauind
  // we only keep those taus which dont decay to leptons
  // i.e. we want only those taus which decay hadronically
  vector<int> tauind;
  for(int no=0; no<brParticle->GetEntries(); no++){
    GenParticle *par = (GenParticle*)brParticle->At(no);
    if( abs(par->PID)==15 ){
      bool gottau=true;
      for(int da=par->D1;da<par->D2+1;da++){
	GenParticle *parda = (GenParticle*)brParticle->At(da);
	if(abs(parda->PID)==15 || abs(parda->PID)==11 || abs(parda->PID)==13){
	  gottau=false; break;
	}
      }
      if(gottau) tauind.push_back(no);
    }
  }
  //cout<<(int)tauind.size()<<endl;

  // Now for the taus which decay hadronically, we
  // loop over the visible daughters (dav) and build the visible 4-vector (vistau)
  // We only keep those taus with pT>20 GeV in goodTau.
  for(int i=0; i<(int)tauind.size(); i++){
    GenParticle *tau = (GenParticle*)brParticle->At(tauind.at(i));
    TLorentzVector vistau(0,0,0,0);
    for(int da=tau->D1;da<tau->D2+1;da++){
      GenParticle *dt = (GenParticle*)brParticle->At(da);
      if(abs(dt->PID)>10 && abs(dt->PID)<17) continue;
      //cout<<"NonLeptonic Daughter"<<endl;
      TLorentzVector dav; dav.SetPtEtaPhiM(dt->PT,dt->Eta,dt->Phi,dt->Mass);
      vistau += dav;
    }
    Lepton temp; temp.v = vistau; temp.id=tau->PID; temp.ind=tauind.at(i);
    if(temp.v.Pt()>20 && fabs(temp.v.Eta())<2.4)goodTau.push_back(temp);
      
  }
      
  
}

void delZAna::BookHistograms()
{
  // Booking syntax for histogram pointers (that we declared in struct Hists in header)
  // obj = new CLASS("localname","Histogram title",NumberOfBins,LowerEdge,HigherEdge)


  h.etmiss = new TH1F("MET","Missing E_{T}",500,0,500);// h.etmiss->Sumw2();

  h.ngoodtau = new TH1F("ngoodtau","NGoodTau",5,0,5);
  h.goodtaupt = new TH1F("goodtaupt","GoodTau pT",200,0,200);

  h.ngoodele = new TH1F("ngoodele", "NGoodEle",5,0,5);
  h.goodelept = new TH1F("goodelept", "GoodElePt",200,0,200);

  h.HT = new TH1F("HTjetgen","Sum of genjet pT = hT",200,0,200);
  h.ngenjets = new TH1F("nGenJets","number of genjets",5,0,5);
  
  h.massjj[0] = new TH1F("mass_dd","Mass of Z daughters", 200,0,200);
  h.massjj[1] = new TH1F("mass_mm","Mass of leading dimuons", 200,0,200); //h.massjj[1]->Sumw2();
  h.massEE = new TH1F("mass_ee","Mass of leading electrons",200,0,200);

  h.cat_3l = new TH1F("cats_3l","category count (>=3l) (>=0T)(mmm,mme,mee,eee,eem,emm,mem,eme)",8,0,8);
  h.cat_2l1T = new TH1F("cats_2l1T","category count (=2l) (>=1T)(mm,me,em,ee)",4,0,4);
  h.lt_3l = new TH1F("LT_3L","LT_3L(lep0+lep1+lep2([if]+tau0))",100,0,1000);
  h.ht_3l = new TH1F("HT_3L","HT_3L",100,0,1000);
  h.lt_2l1t = new TH1F("LT_2L1T","LT_2L1T(lep0+lep1+tau0))",100,0,1000);
  h.ht_2l1t = new TH1F("HT_2L1T","HT_2l1T",100,0,1000);
  h.bestMass_3l = new TH1F("bestMOSSF_3L","bestMass_3L",100,0,1000);
  h.triLepM_3L = new TH1F("triLepM_3L", "triLepM_3L((lep0+lep1+lep2).M())",200,0,2000);
  h.triLepM_2L1T = new TH1F("triLepM_2L1T","triLepM_2L1T((lep0+lep1+tau0).M())",200,0,2000);
  h.dilepMass_2l1t = new TH1F("dilepMOSSF_2l1t","DilepMass of 2L(OSSF) in 2L1T",100,0,1000);
  h.mT_3L[0] = new TH1F("mT0_3L","mT of leading lep in 3L",50,0,500);
  h.mT_3L[1] = new TH1F("mT1_3L","mT of subleading lepin 3L",50,0,500);
  h.mT_3L[2] = new TH1F("mT2_3L","mT of sub-subleading lep in 3L",50,0,500);
  h.mtT_3L = new  TH1F("mtT_2L","mT of leading tau(if goodTau.size>0) in 3L",50,0,500);
  h.MET_3L = new TH1F("MET_3L","MET_3L",100,0,1000);
  h.MET_2L1T = new TH1F("MET_2L1T","MET_2L1T",100,0,1000);
  h.mT_2L1T[0] = new TH1F("mT0_2L1T","mT of leading lep in 2L1T",50,0,500);
  h.mT_2L1T[1] = new TH1F("mT1_2L1T","mT of subleading lep in 2L1T",50,0,500);
  h.mtT_2L1T = new TH1F("mtT_2L1T","mT of leading tau in 2L1T",50,0,500);
  h.st_3L = new TH1F("ST_3L","ST_3L(LT+HT+MET)",100,0,1000);
  h.st_2L1T = new TH1F("ST_2L1T","ST_2L1T(LT+HT+MET)",100,0,1000);

  h.ltht_3L = new TH1F("LTHT_3L","(LT+HT)_3L",100,0,1000);
  h.ltmet_3L = new TH1F("LTMET_3L","(LT+MET)_3L",100,0,1000);
  h.ltht_2L1T = new TH1F("LTHT_2L1T","(LT+HT)_2L1T",100,0,1000);
  h.ltmet_2L1T = new TH1F("LTMET_2L1T","(LT+MET)_2L1T",100,0,1000);
  
  h.dPhi_l0l1_3L = new TH1F("dPhi_l0l1_3L","dPhi_l0l1_3L",64,-3.2,3.2);
  h.dPhi_l1l2_3L = new TH1F("dPhi_l1l2_3L","dPhi_l1l2_3L",64,-3.2,3.2);
  h.dPhi_l0l2_3L = new TH1F("dPhi_l0l2_3L","dPhi_l0l2_3L",64,-3.2,3.2);
  h.maxdPhi_3L = new TH1F("maxdPhi_3L","maxdPhi_3L",64,-3.2,3.2);
  h.mindPhi_3L = new TH1F("mindPhi_3L","mindPhi_3L",64,-3.2,3.2);

  h.dPhi_l0l1_2L1T = new TH1F("dPhi_l0l1_2L1T","dPhi_l0l1_2L1T",64,-3.2,3.2);
  h.dPhi_l0T_2L1T = new TH1F("dPhi_l0T_2L1T","dPhi_l0T_2L1T",64,-3.2,3.2);
  h.dPhi_l1T_2L1T = new TH1F("dPhi_l1T_2L1T","dPhi_l1T_2L1T",64,-3.2,3.2);
  h.maxdPhi_2L1T = new TH1F("maxdPhi_2L1T","maxdPhi_2L1T",64,-3.2,3.2);
  h.mindPhi_2L1T = new TH1F("mindPhi_2L1T","mindPhi_2L1T",64,-3.2,3.2);

  h.dPhi_l0MET_3L = new TH1F("dPhi_l0MET_3L","dPhi_l0MET_3L",64,-3.2,3.2);
  h.dPhi_l1MET_3L = new TH1F("dPhi_l1MET_3L","dPhi_l1MET_3L",64,-3.2,3.2);
  h.dPhi_l2MET_3L = new TH1F("dPhi_l2MET_3L","dPhi_l2MET_3L",64,-3.2,3.2);
  h.maxdPhi_LMET_3L = new TH1F("maxdPhi_LMET_3L","maxdPhi_LMET_3L",64,-3.2,3.2);
  h.mindPhi_LMET_3L = new TH1F("mindPhi_LMET_3L","mindPhi_LMET_3L",64,-3.2,3.2);

  h.dPhi_l0MET_2L1T = new TH1F("dPhi_l0MET_2L1T","dPhi_l0MET_2L1T",64,-3.2,3.2);
  h.dPhi_l1MET_2L1T = new TH1F("dPhi_l1MET_2L1T","dPhi_l1MET_2L1T",64,-3.2,3.2);
  h.dPhi_TMET_2L1T = new TH1F("dPhi_TMET_2L1T","dPhi_TMET_2L1T",64,-3.2,3.2);
  h.maxdPhi_LMET_2L1T = new TH1F("maxdPhi_LMET_2L1T","maxdPhi_LMET_2L1T",64,-3.2,3.2);
  h.mindPhi_LMET_2L1T = new TH1F("mindPhi_LMET_2L1T","mindPhi_LMET_2L1t",64,-3.2,3.2);
  h.dPhi_Tj0_2L1T = new TH1F("dPhi_Tj0_2L1T","dPhi_Tj0_2L1T",64,-3.2,3.2);

  h.dPhi_l0j0_3L = new TH1F("dPhi_l0j0_3L","dPhi_l0j0_3L",64,-3.2,3.2);
  h.dPhi_l0j0_2L1T = new TH1F("dPhi_l0j0_2L1T","dPhi_l0j0_2L1T",64,-3.2,3.2);
  h.dPhi_j0met_3L = new TH1F("dPhi_j0met_3L","dPhi_j0met_3L",64,-3.2,3.2);
  h.dPhi_j0met_2L1T = new TH1F("dPhi_j0met_2L1T","dPhi_j0met_2L1T",64,-3.2,3.2);
  h.dR_l0j0_3L = new TH1F("dR_l0j0_3L", "dR_l0j0_3L",100,0,10);
  h.dR_l0j0_2L1T = new TH1F("dR_l0j0_2L1T", "dR_l0j0_2L1T",100,0,10);
  h.dR_j0met_3L = new TH1F("dR_j0met_3L", "dR_j0met_3L",100,0,10);
  h.dR_j0met_2L1T = new TH1F("dR_j0met_2L1T", "dR_j0met_2L1T",100,0,10);
  
  h.njets_3l = new TH1F("njets_3L","njets_3L",20,0,20);
  h.njets_2l1t = new TH1F("njets_2L1T","njets_2L1T",20,0,20);

  h.j0pT_3L = new TH1F("j0pT_3L","j0pT_3L",50,0,500);
  h.j0pT_2L1T = new TH1F("j0pT_2L1T","j0pT_2L1T",50,0,500);

  h.dR_l0l1_3L = new TH1F("dR_l0l1_3L", "dR_l0l1_3L",100,0,10);
  h.dR_l1l2_3L = new TH1F("dR_l1l2_3L","dR_l1l2_3L",100,0,10);
  h.dR_l0l2_3L = new TH1F("dR_l0l2_3L","dR_l0l2_3L",100,0,10);
  h.dRmin_3L = new TH1F("dRmin_3L","dRmin_3L",100,0,10);
  h.dRmax_3L = new TH1F("dRmax_3L","dRmax_3L",100,0,10);

  h.dR_l0l1_2L1T = new TH1F("dR_l0l1_2L1T", "dR_l0l1_2L1T",100,0,10);
  h.dR_l0T_2L1T = new TH1F("dR_l0T_2L1T","dR_l0T_2L1T",100,0,10);
  h.dR_l1T_2L1T = new TH1F("dR_l1T_2L1T","dR_l1T_2L1T",100,0,10);
  h.dRmin_2L1T = new TH1F("dRmin_2L1T","dRmin_2L1T",100,0,10);
  h.dRmax_2L1T = new TH1F("dRmax_2L1T","dRmax_2L1T",100,0,10);

  h.pT0_3L = new TH1F("pT0_3L","pT_leading lep_3L",50,0,500);
  h.pT1_3L = new TH1F("pT1_3L","pT_subleading lep_3L",50,0,500);
  h.pT2_3L = new TH1F("pT2_3L","pT_sub-subleading lep_3L",50,0,500);
  h.pT_tau_3L = new TH1F("pT0_tau_3L","pT_leading tau(if goodTau>1)_3L",50,0,500);
  h.pT0_2L1T = new TH1F("pT0_2L1T","pT_leading lep_2L1T",50,0,500);
  h.pT1_2L1T = new TH1F("pT1_2L1T","pT_subleading lep_2L1T",50,0,500);
  h.pT_tau_2L1T = new TH1F("pT_2L1T","pT_leading tau_2L1T",50,0,500);
  

  
  // Note the calling of Sumw2() for each histogram we declare.
  // This is needed so that if scale the histograms later, the
  // errors on the points are still correct after scaling.
  // This is relevant.. so ask someone if you don't understand.
  
}






















































/*

  Code written before not needed anymore, tho let it be for reference:

  
  
  //SAVING THE DAUGHTERS OF Z
  // Example of getting daughters of a particle.
  //if particle is Z, save its daughters.
  if(par->PID==23){
  GenParticle *dau1 = (GenParticle*)brParticle->At(par->D1);
  GenParticle *dau2 = (GenParticle*)brParticle->At(par->D2);
  if(dau1->PID==23 || dau2->PID==23) continue;
  q1.SetPtEtaPhiM(dau1->PT,dau1->Eta,dau1->Phi,dau1->Mass);
  q2.SetPtEtaPhiM(dau2->PT,dau2->Eta,dau2->Phi,dau2->Mass);	
  if(q1.Pt()>0 && q2.Pt()>0){	
  float massqq = (q1+q2).M();
  h.massjj[0]->Fill(massqq);
  }
  }

  //DAUGHTERS OF W
  if(par->PID==24){
  GenParticle *d1 = (GenParticle*)brParticle->At(par->D1);
  GenParticle *d2 = (GenParticle*)brParticle->At(par->D2);
  if(abs(d1->PID)>10 && abs(d1->PID)<17 && abs(d2->PID)<17 && abs(d2->PID)>10){
  h.WtoLNu_d1->Fill(d1->PID);
  h.WtoLNu_d2->Fill(d2->PID);
  }
  }

  
     
      
  //Find a lep-nu pair coming from one W, find its dR(dR1). Find jets coming from other W, find its dR(dR2).
  //plot 2D histogram: dR1 v/s dR2
  //float dR1;
  //float dR2;
  if(abs(par->PID)==24){
  GenParticle *d1 = (GenParticle*)brParticle->At(par->D1);
  GenParticle *d2 = (GenParticle*)brParticle->At(par->D2);
  //TLorentzVector q1,q2;
  float dR1;
  float dR2;
  float pt_lnu;
  float pt_jj;
  //leptons
  if(abs(d1->PID)<17 && abs(d1->PID)>10 && abs(d2->PID)<17 && abs(d2->PID)>10){
  q1.SetPtEtaPhiM(d1->PT,d1->Eta,d1->Phi,d1->Mass);
  q2.SetPtEtaPhiM(d2->PT,d2->Eta,d2->Phi,d2->Mass);
  dR1=q1.DeltaR(q2);
  h.delR1->Fill(dR1);
  //h.pTd1_l->Fill(d1->PT);
  pt_lnu = (q1+q2).Pt();
  h.pT_lnu->Fill(pt_lnu);
  h.d1_PID_lep->Fill(d1->PID);
  }
  //jets
  if(abs(d1->PID)<10 && abs(d1->PID)>0 && abs(d2->PID)<10 && abs(d2->PID)>0){ 
  q1.SetPtEtaPhiM(d1->PT,d1->Eta,d1->Phi,d1->Mass);
  q2.SetPtEtaPhiM(d2->PT,d2->Eta,d2->Phi,d2->Mass);
  dR2 = q1.DeltaR(q2);
  h.delR2->Fill(dR2);
  //h.pTd1_j->Fill(d1->PT);
  pt_jj=(q1+q2).Pt();
  h.pT_jj->Fill(pt_jj);
  h.d1_PID_j->Fill(d1->PID);
  }
  h.dR12_2D->Fill(dR1,dR2);
  h.pT_WW->Fill(pt_lnu,pt_jj);
  if(pt_lnu>80 || pt_jj>80){
  h.dR12_bstW->Fill(dR1,dR2); //2D plot of dR's when W (going to LNu) is boosted (bst) i.e pT>80
  }
  }

  float mT=0;
  if(par->Status==1 && abs(par->PID)==13 || abs(par->PID)==11){ //
  Lepton temp;
  temp.v.SetPtEtaPhiM(par->PT,par->Eta,par->Phi,par->Mass);
  temp.id=par->PID; temp.ind=no;
  float p=par->PT;
  float q=mymet.Pt();
  float dphi=delta_phi(p,q);
  mT=sqrt(2*p*q*(1-cos(dphi)));
  //h.WmT->Fill(mT);
  }

  h.WtoLNu_d1 = new TH1F("WtoLNu_d1","WtoLNu_d1_PID",40,-20,20);
  h.WtoLNu_d2 = new TH1F("WtoLNu_d2","WtoLNu_d2_PID",40,-20,20);
      
  h.delR1 = new TH1F("delR1","delR1",200,0,10);
  h.pT_lnu = new TH1F("pT_lnu","pT_LepNu",200,0,200);
  h.delR2 = new TH1F("delR2","delR2",200,0,10);
  h.pT_jj = new TH1F("pT_jj","pT_jj",200,0,200);
  h.d1_PID_j = new TH1F("d1_PID_j","d1_PID_j",10,0,10);
  h.d1_PID_lep = new TH1F("d1_PID_lep","d1_PID_lep",20,0,20);

      
  //2D-Histograms
  h.dR12_2D = new TH2F("dR12_2D","dR12_2D",60,0,6,60,0,6);
  h.pT_WW = new TH2F("pT_WW","pT_WW",500,0,500,500,0,500);
  h.dR12_bstW = new TH2F("dR12_bstW","dR12_boostedW",60,0,6,60,0,6);
      
      
*/     
