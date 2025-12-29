#include <iostream>
#include <stdlib.h>
#include "TChain.h"
#include "delZAna.h"

//This is the driver script, which becomes our main program
//Here we set the options which we wish to use, which files
//we want to run over and so on.

//the argument decides what input sample we want to run over.
//we give separate names of output files for each set of
//input files.

int main(int argc, char *argv[])
{
  using std::cout;
  using std::endl;

  if(argc<2){
    cout<<" Please give one integer argument "<<endl;
    return 0;
  }
  int sample_id = atoi(argv[1]);

  
  const char *hstfilename, *sumfilename, *outputFile3Lname, *outputFile2L1Tname, *outputFile3L, *outputFile2L1T;

  TChain *chain = new TChain("Delphes");

  if(sample_id==0){ //M500_50k events
    chain->Add("VLLD_input/M500/VLLD_mu_M500_50k.root");
    hstfilename = "VLLD_output/codetest/VLLD_mu_M500_50k.root";//output histogram file
    sumfilename = "VLLD_output/codetest/sum_VLLD_mu_M500_50k.txt";//output text file
    outputFile3Lname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_0.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_2L1T_0.txt"; //text file for 2L1T
  }

  //---------------------------------------------------------------    M500 --------------------------------------------------------------------
  if(sample_id==1){ //M500_50k events
    chain->Add("VLLD_input/M500/VLLD_mu_M500_50k.root");
    hstfilename = "VLLD_output/M500/VLLD_mu_M500_50k.root";//output histogram file
    sumfilename = "VLLD_output/M500/sum_VLLD_mu_M500_50k.txt";//output text file
    outputFile3Lname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_0.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_2L1T_0.txt"; //text file for 2L1T
  }
  
  if(sample_id==2){ //M500_50k events
    chain->Add("VLLD_input/M500/VLLD_mu_M500_50k_1.root");
    hstfilename = "VLLD_output/M500/VLLD_mu_M500_50k_1.root";//output histogram file
    sumfilename = "VLLD_output/M500/sum_VLLD_mu_M500_50k_1.txt";//output text file
    outputFile3Lname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_1.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_2L1T_1.txt"; //text file for 2L1T
  }
  
  if(sample_id==3){ //M500_50k events
    chain->Add("VLLD_input/M500/VLLD_mu_M500_50k_2.root");
    hstfilename = "VLLD_output/M500/VLLD_mu_M500_50k_2.root";//output histogram file
    sumfilename = "VLLD_output/M500/sum_VLLD_mu_M500_50k_2.txt";//output text file
    outputFile3Lname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_3L_2.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M500/VLLD_mu_M500_50k_SOM_2L1T_2.txt"; //text file for 2L1T
  }

  //---------------------------------------------------------------    M750 --------------------------------------------------------------------
  if(sample_id==4){ //M750_50k events
    chain->Add("VLLD_input/M750/VLLD_mu_M750_50k.root");
    hstfilename = "VLLD_output/M750/VLLD_mu_M750_50k.root";//output histogram file
    sumfilename = "VLLD_output/M750/sum_VLLD_mu_M750_50k.txt";//output text file
    outputFile3Lname = "VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_0.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M750/VLLD_mu_M750_50k_SOM_2L1T_0.txt"; //text file for 2L1T
  }
  
  if(sample_id==5){ //M750_50k events
    chain->Add("VLLD_input/M750/VLLD_mu_M750_50k_1.root");
    hstfilename = "VLLD_output/M750/VLLD_mu_M750_50k_1.root";//output histogram file
    sumfilename = "VLLD_output/M750/sum_VLLD_mu_M750_50k_1.txt";//output text file
    outputFile3Lname = "VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_1.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M750/VLLD_mu_M750_50k_SOM_2L1T_1.txt"; //text file for 2L1T
  }
  
  if(sample_id==6){ //M750_50k events
    chain->Add("VLLD_input/M750/VLLD_mu_M750_50k_2.root");
    hstfilename = "VLLD_output/M750/VLLD_mu_M750_50k_2.root";//output histogram file
    sumfilename = "VLLD_output/M750/sum_VLLD_mu_M750_50k_2.txt";//output text file
    outputFile3Lname = "VLLD_output/M750/VLLD_mu_M750_50k_SOM_3L_2.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M750/VLLD_mu_M750_50k_SOM_2L1T_2.txt"; //text file for 2L1T
  }

  //---------------------------------------------------------------    M1000 --------------------------------------------------------------------
  if(sample_id==7){ //M1000_50k events
    chain->Add("VLLD_input/M1000/VLLD_mu_M1000_50k.root");
    hstfilename = "VLLD_output/M1000/VLLD_mu_M1000_50k.root";//output histogram file
    sumfilename = "VLLD_output/M1000/sum_VLLD_mu_M1000_50k.txt";//output text file
    outputFile3Lname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_0.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_2L1T_0.txt"; //text file for 2L1T
  }
  
  if(sample_id==8){ //M1000_50k events
    chain->Add("VLLD_input/M1000/VLLD_mu_M1000_50k_1.root");
    hstfilename = "VLLD_output/M1000/VLLD_mu_M1000_50k_1.root";//output histogram file
    sumfilename = "VLLD_output/M1000/sum_VLLD_mu_M1000_50k_1.txt";//output text file
    outputFile3Lname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_1.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_2L1T_1.txt"; //text file for 2L1T
  }
  
  if(sample_id==9){ //M1000_50k events
    chain->Add("VLLD_input/M1000/VLLD_mu_M1000_50k_2.root");
    hstfilename = "VLLD_output/M1000/VLLD_mu_M1000_50k_2.root";//output histogram file
    sumfilename = "VLLD_output/M1000/sum_VLLD_mu_M1000_50k_2.txt";//output text file
    outputFile3Lname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_2.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_2L1T_2.txt"; //text file for 2L1T
  }

  //---------------------------------------------------------------    M1500 --------------------------------------------------------------------
  if(sample_id==10){ //M1500_50k events
    chain->Add("VLLD_input/M1500/VLLD_mu_M1500_50k.root");
    hstfilename = "VLLD_output/M1500/VLLD_mu_M1500_50k.root";//output histogram file
    sumfilename = "VLLD_output/M1500/sum_VLLD_mu_M1500_50k.txt";//output text file
    outputFile3Lname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_0.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_2L1T_0.txt"; //text file for 2L1T
  }
  
  if(sample_id==11){ //M1500_50k events
    chain->Add("VLLD_input/M1500/VLLD_mu_M1500_50k_1.root");
    hstfilename = "VLLD_output/M1500/VLLD_mu_M1500_50k_1.root";//output histogram file
    sumfilename = "VLLD_output/M1500/sum_VLLD_mu_M1500_50k_1.txt";//output text file
    outputFile3Lname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_1.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_2L1T_1.txt"; //text file for 2L1T
  }
  
  if(sample_id==12){ //M1500_50k events
    chain->Add("VLLD_input/M1500/VLLD_mu_M1500_50k_2.root");
    hstfilename = "VLLD_output/M1500/VLLD_mu_M1500_50k_2.root";//output histogram file
    sumfilename = "VLLD_output/M1500/sum_VLLD_mu_M1500_50k_2.txt";//output text file
    outputFile3Lname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_3L_2.txt"; //text file for 3L
    outputFile2L1Tname = "VLLD_output/M1000/VLLD_mu_M1000_50k_SOM_2L1T_2.txt"; //text file for 2L1T
  }
  
  
  cout<<"Set Options... ";
  std::cout<<"Output files are "<<hstfilename<<", "<<sumfilename<<", "<<outputFile3Lname<<" and "<<outputFile2L1Tname<<std::endl;
  
  delZAna t(chain);//declared an instance of our class.
  t.SetHstFileName(hstfilename);
  t.SetSumFileName(sumfilename);
  t.SetOpFile3L(outputFile3Lname);
  t.SetOpFile2L1T(outputFile2L1Tname);
  t.SetVerbose(1500);//set verbosity level for output.
  cout<<" Processing... ";
  t.Begin();
  t.Loop();
  t.End();

  cout<<"Completed."<<endl;
  return 0;
}
