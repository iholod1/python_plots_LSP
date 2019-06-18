# returns cross scection data
# I. Holod, 07/06/17
import scipy.io
import numpy as np


def getHeBe():
  """Reads cross section data and returns particle energy and probability
     of creating neutron"""
# HeBe probability
  HeBe=scipy.io.loadmat('/g/g19/holod1/solid_target/AlphaBeToCarbonNeutronCrossSection.mat')
  Prob=HeBe.get('ThickTargetProb')
  pEn=HeBe.get('PEnergies')
  Prob=np.append(0,Prob);
  Prob=Prob.reshape((len(Prob),1));
  pEn=np.append(0,pEn);
  pEn=pEn.reshape((len(pEn),1));
  return(pEn,Prob)

def getDD():
  """Reads cross section data and returns particle energy and DD fusion cross section"""
# DD fusion cross section
  DDCross=scipy.io.loadmat('/g/g19/holod1/solid_target/DDCross.mat')
  dat=DDCross.get('DDCrossSection')
  csE=dat[0][0][0][:,0];
  csS=dat[0][0][1][:,0];
  csE=np.append(0,csE);
  csS=np.append(0,csS);
  csE=csE.reshape((len(csE),1));
  csS=csS.reshape((len(csS),1));
  return(csE,csS)

def getDCD2():
  """Reads cross section data and returns particle energy and probability
     of creating neutron"""
# DCD2 probability
  DCD2=scipy.io.loadmat('/g/g19/holod1/solid_target/D_on_CD2.mat')
  dat=DCD2.get('ThickTarget')
  Prob=dat[0][0][0][:,0];
  pEn=dat[0][0][1][:,0];
  Prob=np.append(0,Prob);
  Prob=Prob.reshape((len(Prob),1));
  pEn=np.append(0,pEn);
  pEn=pEn.reshape((len(pEn),1));
  return(pEn,Prob)
