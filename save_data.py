#!/usr/bin/env python
# save 2 d from sclr and flds
from sys import exit
import math
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
import pylab as pl

from read_xdr import readXDRsclr,readXDRflds
from read_xdr import readXDRstruct

def save_dat(fname,X,Z,dat):
  fid = open(fname, 'w')
  nx=len(X);nz=len(Z);
  fid.write("%d %d\n" %(nx,nz));
  for i in range(nx):
    fid.write("%.2f " %(X[i]));
  fid.write("\n");
  for i in range(nz):
    fid.write("%.2f " %(Z[i]));
  fid.write("\n");
  for j in range(nz):
    for i in range(nx):
      fid.write("%.2e " %(dat[i,j]));
    fid.write("\n");
  fid.close()


##### CONTROL PARAMETERS #####
num=17500
#dirname='/p/lscratche/holod1/miniDPF/c120_v40/wneutrals/initial/'
#dirname='/p/lscratche/holod1/miniDPF/c280_v30/neutrals/initial/'
dirname='./'
fname1=dirname+'sclr'+str(num)+'.p4'
fname2=dirname+'flds'+str(num)+'.p4'

(sX,sY,sZ,sVar,sVarNames,sVarUnits,stime)=readXDRsclr(fname1)
for i in range(len(sVar)): print i, sVarNames[i]
tstamp =  "%.2f" % stime

(vX,vY,vZ,vVar,vVarNames,vVarUnits,vtime)=readXDRflds(fname2)
for i in range(len(vVarNames)): print i, vVarNames[i]
tstamp =  "%.2f" % vtime

if True:
# ne,ni
 rbtheta=np.squeeze(sVar[1]);
 dens=np.squeeze(sVar[2]);
 densN=np.zeros(np.shape(dens));
 for i in range(np.shape(dens)[0]):
   for j in range(np.shape(dens)[1]):
    if rbtheta[i,j]<13000:
      densN[i,j]=dens[i,j];
      dens[i,j]=0.0;
 save_dat(dirname+'dense.in',sX,sZ,dens);
 save_dat(dirname+'densi.in',sX,sZ,dens);
 save_dat(dirname+'neutr.in',sX,sZ,densN);


 dat=sVar[3]
 save_dat(dirname+'tempe.in',sX,sZ,np.squeeze(dat[:,0,:]));
 dat=sVar[7]
 save_dat(dirname+'tempi.in',sX,sZ,np.squeeze(dat[:,0,:]));

 dat=vVar[2] # electron flow (beta)
 save_dat(dirname+'vxe.in',vX,vZ,np.squeeze(dat[0,:,0,:]));
 save_dat(dirname+'vze.in',vX,vZ,np.squeeze(dat[2,:,0,:]));

 dat=vVar[3] # ion flow (beta)
 save_dat(dirname+'vxi.in',vX,vZ,np.squeeze(dat[0,:,0,:]));
 save_dat(dirname+'vzi.in',vX,vZ,np.squeeze(dat[2,:,0,:]));

# conductivity
 dat=dens;
 for i in range(np.shape(dat)[0]):
   for j in range(np.shape(dat)[1]):
    if dat[i,j]<0.7e17: 
      dat[i,j]=0.0;
    else:
      dat[i,j]=1.0;
 save_dat(dirname+'cond.in',sX,sZ,dat);

 exit()








fig1, ax1 = pl.subplots()
ivar=2; # set variable to plot
dat=sVar[ivar]
ax1.contour(sZ, sX, dat[:,0,:],origin='lower',linewidths=1)
im1 = ax1.imshow(dat[:,0,:], interpolation='bilinear',
                origin='lower', extent=(min(sZ), max(sZ), min(sX), max(sX)))
CB = plt.colorbar(im1)  
ax1.set(title=sVarNames[ivar]+" at "+tstamp+"ns")

fig2, ax2 = pl.subplots()
ivar=2; # set variable to plot
dat=vVar[ivar]
print np.shape(dat)
ii=0;
ax2.contour(vZ, vX, dat[ii,:,0,:],origin='lower',linewidths=1)
im2 = ax2.imshow(dat[ii,:,0,:], interpolation='bilinear',
                origin='lower', extent=(min(vZ), max(vZ), min(vX), max(vX)))
CB = plt.colorbar(im2)  
ax2.set(title=vVarNames[ivar]+" at "+tstamp+"ns")

plt.show()






