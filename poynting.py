#!/usr/bin/env python
# script to plot from sclr*.p4
# I. Holod, 07/05/16


import sys, argparse
import os.path
import math
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
import pylab as pl
from PIL import Image

from read_xdr import readXDRflds
from read_xdr import readXDRstruct



##### CONTROL PARAMETERS #####
num=12000
ivar=1; # set variable to plot
flip=0; # flip plot symmetrically
dirname=''
num=6000

fname=dirname+'flds'+str(num)+'.p4'
print fname;


(X,Y,Z,Var,VarNames,VarUnits,time)=readXDRflds(fname)

tstamp =  "%.2f" % time
print "Time:", tstamp

Nx=X.shape[0]; Nz=Z.shape[0];
ex=np.zeros((Nx,Nz));
ez=np.zeros((Nx,Nz));
by=np.zeros((Nx,Nz));



for i in range(Nx):
  for j in range(Nz):
    ex[i,j]=Var[0][0][i,0,j];
    ez[i,j]=Var[0][2][i,0,j];
    by[i,j]=Var[1][1][i,0,j];

r1=0.45;
r2=0.76;
z1=1.2;
z2=1.9;

ix1=int(np.argwhere(X>r1)[0]);
ix2=int(np.argwhere(X<r2)[-1]);
iz1=int(np.argwhere(Z>z1)[0]);
iz2=int(np.argwhere(Z<z2)[-1]);


X1=X[ix1:ix2];
EX1=ex[ix1:ix2,0];
BY1=by[ix1:ix2,0];

Z2=Z[iz1:iz2];
EZ2=ez[ix2+1,iz1:iz2];
BY2=by[ix2+1,iz1:iz2];

print X[ix1], X[ix2]
print Z[iz1], Z[iz2]

nx=60; nz=60;
xspan=np.linspace(r1,r2,num=nx);
zspan=np.linspace(z1,z2,num=nz);
dx=(r2-r1)/(nx-1)
dz=(z2-z1)/(nz-1)

p1=np.sum([np.interp(xspan[i],X1,EX1)*np.interp(xspan[i],X1,BY1)*sp.pi*2.*dx*xspan[i] for i in range(len(xspan))]);
p2=np.sum([-np.interp(zspan[i],Z2,EZ2)*np.interp(zspan[i],Z2,BY2)*sp.pi*2.*dz*X[ix2+1] for i in range(len(zspan))]);
print "p1=", p1, "p2=", p2, "p2/p1=", p2/p1



xLowLim=min(X);


exit()
fig, ax = pl.subplots(figsize=(8,8))
im = ax.imshow(dat, interpolation='none',origin='lower', extent=(min(Z), max(Z), xLowLim, max(X)))
CB = plt.colorbar(im) 
 
ax.set(title=VarNames[ivar]+"("+str(xc)+","+str(yc)+","+str(zc)+")"+" (" + VarUnits[ivar] + ")" +" at "+tstamp+"ns")

# add structure
if os.path.isfile(dirname+'struct.p4'):
  (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
  lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
  lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
  ax.add_collection(lc)
  ax.set_aspect('auto')
  if flip:
    lines=[[(za[i],-xa[i]),(zb[i],-xb[i])] for i in range(len(xa))]
    lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
    ax.add_collection(lc)
    ax.set_aspect('equal')

plt.show();

exit()
fig.savefig(dirname + VarNames[ivar] + '_'+ str(num) + '.png',dpi=4*fig.dpi)

exit()

fid = open(dirname+'density.in', 'w')
dat=Var[6]
dat[:,0,:]
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
    fid.write("%.2e " %(dat[i,0,j]));
  fid.write("\n");
fid.close()

