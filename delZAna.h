//////////////////////////////////////////////////
// Author: Sourabh Dube
// Date: May 30th, 2014
// Run on Delphes output and have same structure
// as makeclass code
//////////////////////////////////////////////////

#ifndef delZAna_h
#define delZAna_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>

#include <TH2.h>
#include <TMath.h>
#include "TLorentzVector.h"
#include "TVector.h"
#include "TVector2.h"
#include "TVector3.h"
#include "TClonesArray.h"
// Header file for the classes stored in the TTree if any.
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
// external header files needed from Delphes
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "classes/DelphesClasses.h"

//declaration of the delphes classes we shall use
class Particle;
class ExRootTreeReader;
class GenParticle;
class Photon;
class Electron;
class Muon;
class Jet;
class Track;
class MissingET;


// Fixed size dimensions of array or collections stored in the TTree if any.
using namespace std;

class delZAna {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // List of branches

  ExRootTreeReader *treeReader;// = new ExRootTreeReader(&chain);
  //declaring the branches which hold the classes
  TClonesArray  *brMissingET;
  //TClonesArray  *brScalarHT;
  //TClonesArray  *brJet;
  TClonesArray  *brGenJet;
  TClonesArray  *brFatJet;
  //TClonesArray  *brPhoton;
  //TClonesArray  *brTrack;
  //TClonesArray  *brMuon;
  //TClonesArray  *brEle;
  TClonesArray  *brParticle;

  delZAna(TTree *tree=0);
  virtual ~delZAna();
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  //User added functions
  //You add functions here
  void Begin();
  void End();
  void BookHistograms();
  void SetHstFileName(const char *HstFileName){ _HstFileName = HstFileName;} //const 
  void SetSumFileName(const char *SumFileName){ _SumFileName = SumFileName;} //const
  void SetOpFile3L(const char *OpFileName3L){_OpFileName3L = OpFileName3L;} //const
  void SetOpFile2L1T(const char *OpFileName2L1T){_OpFileName2L1T = OpFileName2L1T;} //const
  void SetVerbose(int verbose){ _verbosity = verbose; }
  void Sort(int opt);

  float delta_phi(float phi1, float phi2);
  float transv_mass(float pT_lep, float MET, float dPhi);
  Double_t delR(TLorentzVector v1,  TLorentzVector v2);
  void TauBuilder();
  float calc_iso(float,TLorentzVector,int);
  float maxfloat(float a, float b, float c);
  float minfloat(float a, float b, float c);

public:
  struct Hists {
    //Declare the histograms you want in here.
    TH1F *etmiss,*ngoodtau,*goodtaupt;

    TH1F *ngoodele;
    TH1F *goodelept;
     
    TH1F *massjj[2];
    TH1F *massEE;

    TH1F *cat_3l;
    TH1F *cat_2l1T;
    TH1F *lt_3l;
    TH1F *lt_2l1t;
    TH1F *ht_3l, *njets_3l, *j0pT_3L;
    TH1F *ht_2l1t, *njets_2l1t, *j0pT_2L1T;
    TH1F *bestMass_3l, *triLepM_3L;
    TH1F *dilepMass_2l1t, *triLepM_2L1T;
    TH1F *mT_3L[3];
    TH1F *mtT_3L;
    TH1F *MET_3L;
    TH1F *mT_2L1T[2];
    TH1F *mtT_2L1T;
    TH1F *MET_2L1T;
    TH1F *st_3L;
    TH1F *ltht_3L, *ltmet_3L;
    TH1F *ltht_2L1T, *ltmet_2L1T;
    TH1F *dPhi_l0l1_3L, *dPhi_l1l2_3L, *dPhi_l0l2_3L, *mindPhi_3L, *maxdPhi_3L, *dPhi_l0j0_3L;
    TH1F *dR_l0l1_3L, *dR_l1l2_3L, *dR_l0l2_3L, *dRmin_3L, *dRmax_3L;
    TH1F *dPhi_l0l1_2L1T, *dPhi_l0T_2L1T, *dPhi_l1T_2L1T, *maxdPhi_2L1T, *mindPhi_2L1T, *dPhi_l0j0_2L1T, *dPhi_Tj0_2L1T;
    TH1F *dR_l0l1_2L1T, *dR_l0T_2L1T, *dR_l1T_2L1T, *dRmin_2L1T, *dRmax_2L1T;
    TH1F *pT0_3L, *pT1_3L, *pT2_3L, *pT_tau_3L;
    TH1F *pT0_2L1T, *pT1_2L1T, *pT_tau_2L1T;
    TH1F *st_2L1T;
    TH1F *dPhi_l0MET_3L, *dPhi_l1MET_3L, *dPhi_l2MET_3L;
    TH1F *maxdPhi_LMET_3L, *mindPhi_LMET_3L;
    TH1F *dPhi_l0MET_2L1T, *dPhi_l1MET_2L1T, *dPhi_TMET_2L1T;
    TH1F *maxdPhi_LMET_2L1T, *mindPhi_LMET_2L1T;
    TH1F *dPhi_j0met_3L, *dPhi_j0met_2L1T;
    TH1F *dR_l0j0_3L, *dR_l0j0_2L1T, *dR_j0met_3L, *dR_j0met_2L1T;
    TH1F *HT;
    TH1F *ngenjets;

    /*
      TH1F *WtoLNu_d1;
      TH1F *WtoLNu_d2;
      TH1F *delR1;
      //TH1F *pTd1_l;
      TH1F *pT_lnu;
      TH1F *delR2;
      //TH1F *pTd1_j;
      TH1F *pT_jj;
      TH1F *d1_PID_j;
      TH1F *d1_PID_lep;

      //2D-Histograms
      TH2F *dR12_2D;
      TH2F *pT_WW;
      TH2F *dR12_bstW;
    */
     
  };
  struct Lepton {
    TLorentzVector v;
    int id;
    int ind;
    float wt;
    int flavor;
  };
  std::ofstream  outputFile3L; 
  std::ofstream  outputFile2L1T; 
   
protected:
  Hists h;

private:
  TFile *_HstFile;
  const char *_HstFileName; //const 
  const char *_SumFileName; // const
  const char *_OpFileName3L; //const
  const char *_OpFileName2L1T; //const 
  int _verbosity;
  float GEV, MEV2GEV;
  int nEvtTotal, njet;
  vector<Lepton> goodMu;
  vector<Lepton> goodTau;
  vector<Lepton> goodEle;
  vector<Lepton> jetgen;
  vector<Lepton> goodLep; //includes e and mu
};

#endif

#ifdef delZAna_cxx
delZAna::delZAna(TTree *tree) : fChain(0) 
{
  Init(tree);
}
delZAna::~delZAna()
{
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}
void delZAna::Init(TTree *tree)
{
  if(!tree) return;
  fChain = tree;
  
  treeReader = new ExRootTreeReader(fChain);
  //brMissingET = treeReader->UseBranch("MissingET");
  //brScalarHT = treeReader->UseBranch("ScalarHT");
  //brJet = treeReader->UseBranch("Jet");
  brGenJet = treeReader->UseBranch("GenJet");
  //brFatJet = treeReader->UseBranch("FatJet");
  //brPhoton = treeReader->UseBranch("Photon");
  //brTrack = treeReader->UseBranch("Track");
  //brMuon= treeReader->UseBranch("Muon");
  //brEle = treeReader->UseBranch("Electron");
  brParticle= treeReader->UseBranch("Particle");
}
#endif
