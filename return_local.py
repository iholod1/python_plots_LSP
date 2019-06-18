# returns box-averaged value from sclr of flds .p4 file
# I. Holod, 06/27/17
import sys, argparse, os.path
import numpy as np



def return_local_sclr(dirname,num,ivar,x0,x1,z0,z1):
  """Reads sclr file and return average value of probe
     Output: time,value"""
  from read_xdr import readXDRsclr
  print "bounding box: (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1)
  fname=dirname+'sclr'+str(num)+'.p4'
  print fname;
  (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)
  dat=Var[ivar];
  dat=np.squeeze(dat[:,0,:]);

  indx=np.argwhere((X>=x0)&(X<=x1));
  indz=np.argwhere((Z>=z0)&(Z<=z1));
  indz=np.transpose(indz)
  val=np.mean(dat[indx,indz].flatten());
  return (time,val,VarNames[ivar],VarUnits[ivar])

def return_local_flds(dirname,num,ivar,x0,x1,z0,z1):
  """Reads flds file and return average value of probe
     Output: time,value"""
  from read_xdr import readXDRflds
  print "bounding box: (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1)
  fname=dirname+'flds'+str(num)+'.p4'
  print fname;
  (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRflds(fname)
  datx=Var[ivar][0];
  daty=Var[ivar][1];
  datz=Var[ivar][2];

  datx=np.squeeze(datx[:,0,:]);
  daty=np.squeeze(daty[:,0,:]);
  datz=np.squeeze(datz[:,0,:]);

  indx=np.argwhere((X>=x0)&(X<=x1));
  indz=np.argwhere((Z>=z0)&(Z<=z1));
  indz=np.transpose(indz)
  
  valx=np.mean(datx[indx,indz].flatten());
  valy=np.mean(daty[indx,indz].flatten());
  valz=np.mean(datz[indx,indz].flatten());
  return (time,valx,valy,valz,VarNames[ivar],VarUnits[ivar])

def return_local_beta(dirname,num,iB,iN,iT,x0,x1,z0,z1): # NOT COMPLETED YET
  """Reads sclr file and return average value of beta
     Output: time,value"""
  from read_xdr import readXDRsclr
  print "bounding box: (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1)
  fname=dirname+'sclr'+str(num)+'.p4'
  print fname;
  (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)
  Bfield=np.squeeze(Var[iB][:,0,:])
  Nele=np.squeeze(Var[iN][:,0,:])
  Tele=np.squeeze(Var[iT][:,0,:])

  dat=Nele*Tele/np.square(Bele);

  indx=np.argwhere((X>=x0)&(X<=x1));
  indz=np.argwhere((Z>=z0)&(Z<=z1));
  indz=np.transpose(indz)
  val=np.mean(dat[indx,indz].flatten());
  return (time,val,VarNames[ivar],VarUnits[ivar])

def return_local_ezden(dirname,num,x0,x1,z0,z1):
  """Reads flds file and return average value of probe
     Output: time,value"""
  from read_xdr import readXDRflds, readXDRsclr
  print "bounding box: (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1)
  fname=dirname+'flds'+str(num)+'.p4'
  print fname;
  (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRflds(fname)
  Ez=np.squeeze(Var[0][2][:,0,:]);
  fname=dirname+'sclr'+str(num)+'.p4'
  (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)
  dens=np.squeeze(Var[14][:,0,:]);

  indx=np.argwhere((X>=x0)&(X<=x1));
  indz=np.argwhere((Z>=z0)&(Z<=z1));
  indz=np.transpose(indz)
  
  ezden=np.mean(Ez[indx,indz].flatten()*dens[indx,indz].flatten());
  return (time,ezden,'EzDen','kV/cm/cc')

