#!/usr/bin/env python
# script to plot from two sclr*.p4 files
# I. Holod, 05/10/18


import argparse
import os.path
import numpy as np

import matplotlib

from read_xdr import readXDRsclr
from read_xdr import readXDRstruct


dirname=os.getcwd() + '/';
class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description='Process arguments');
parser.add_argument('-i0', type=int, help='time index initial');
parser.add_argument('-i1', type=int, help='time index final');
parser.add_argument('-p0', type=int, help='probe initial');
parser.add_argument('-p1', type=int, help='probe final');
parser.add_argument('-minx', type=float, help='minimum x');
parser.add_argument('-maxx', type=float, help='maximum x');
parser.add_argument('-minz', type=float, help='minimum z');
parser.add_argument('-maxz', type=float, help='maximum z');
parser.add_argument('-cmap', type=str, help='colormap');
parser.add_argument('-cmin', type=float, help='min value');
parser.add_argument('-cmax', type=float, help='max value');
parser.add_argument('-slicex', type=float, help='z-slice along given x');
parser.add_argument('-slicez', type=float, help='x-slice along given z');

parser.add_argument('-f', action='store_true', help="add horizontal mirror image");
parser.add_argument('-a', action='store_true', help="keep aspect ratio");
parser.add_argument('-log', action='store_true', help="log scale");
parser.add_argument('-save', action='store_true', help="save only");
parser.parse_args(namespace=arg)

num0 = int(arg.i0) if arg.i0 != None else 1;
num1 = int(arg.i1) if arg.i1 != None else 1;
ivar0 = int(arg.p0) if arg.p0 != None else 0;
ivar1 = int(arg.p1) if arg.p1 != None else 0;

x0 = float(arg.minx) if arg.minx != None else None 
x1 = float(arg.maxx) if arg.maxx != None else None 
z0 = float(arg.minz) if arg.minz != None else None 
z1 = float(arg.maxz) if arg.maxz != None else None 

cmap = arg.cmap  if arg.cmap != None else None 
cmin = float(arg.cmin) if arg.cmin != None else None 
cmax = float(arg.cmax) if arg.cmax != None else None 

sliceX = float(arg.slicex) if arg.slicex != None else None
sliceZ = float(arg.slicez) if arg.slicez != None else None  

flip = arg.f;
aspect= arg.a;
logScale = arg.log;
noShow = arg.save;

if noShow: matplotlib.use('Agg') # to run in pdebug (not using $DISPLAY)
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
from PIL import Image

#######################################################
fname=dirname+'sclr'+str(num0)+'.p4'
print(fname);
(X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)
if not noShow:
    for i in range(len(Var)): 
        print(str(i) + " " + VarNames[i] + " " + VarUnits[i]);
tstamp =  "%.2f" % time

dat=Var[ivar0];
dat=np.squeeze(dat[:,0,:]);

if flip: x0=min(X)
xmin=x0 if (x0!=None) and (x0>min(X)) and (x0<=max(X)) else min(X)
xmax=x1 if (x1!=None) and (x1>xmin) and (x1<=max(X)) else max(X)
zmin=z0 if (z0!=None) and (z0>min(Z)) and (z0<max(Z)) else min(Z)
zmax=z1 if (z1!=None) and (z1>zmin) and (z1<=max(Z)) else max(Z)

indx=np.argwhere((X>=xmin)&(X<=xmax));
indx=indx.reshape(indx.shape[0],1);
indz=np.argwhere((Z>=zmin)&(Z<=zmax));
indz=indz.reshape(1,indz.shape[0]);

#indz=np.transpose(indz)
X=X[indx];
Z=Z[indz];
dat0=dat[indx,indz];

#######################################################
fname=dirname+'sclr'+str(num1)+'.p4'
print(fname);
(X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)
if not noShow:
    for i in range(len(Var)): 
        print(str(i) + " " + VarNames[i] + " " + VarUnits[i]);
tstamp =  "%.2f" % time

dat=Var[ivar1];
dat=np.squeeze(dat[:,0,:]);

if flip: x0=min(X)
xmin=x0 if (x0!=None) and (x0>min(X)) and (x0<=max(X)) else min(X)
xmax=x1 if (x1!=None) and (x1>xmin) and (x1<=max(X)) else max(X)
zmin=z0 if (z0!=None) and (z0>min(Z)) and (z0<max(Z)) else min(Z)
zmax=z1 if (z1!=None) and (z1>zmin) and (z1<=max(Z)) else max(Z)

indx=np.argwhere((X>=xmin)&(X<=xmax));
indx=indx.reshape(indx.shape[0],1);
indz=np.argwhere((Z>=zmin)&(Z<=zmax));
indz=indz.reshape(1,indz.shape[0]);

#indz=np.transpose(indz)
X=X[indx];
Z=Z[indz];
dat1=dat[indx,indz];

########################################
# sweping efficiency
nX=X.size;
nZ=Z.size;
dX=np.diff(X,axis=0)
Xc = X[0:-1]+0.5*dX
#print Xc
mnorm=np.zeros((1,nZ));
dat=np.zeros((nX-1,nZ));
for i in range(0,nX-1):
    mnorm[0,:] = mnorm[0,:]+dat0[i,:]*Xc[i]*dX[i]*2.*np.pi
for i in range(0,nX-1):
    dat[i,:] = (dat1[i,:]*Xc[i]*dX[i]*2.*np.pi)/mnorm[0,:]

dat=np.cumsum(dat,axis=0)

fig, ax = pl.subplots(figsize=(8,8))

cm = matplotlib.cm.get_cmap(cmap)

if logScale:
    im = ax.pcolor(Z[0,:],Xc,dat,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)
else:
    im = ax.pcolor(Z[0,:],Xc,dat,vmin=cmin,vmax=cmax,cmap=cm)

ax.set_xlim([min(Z[0,:]),max(Z[0,:])])
ax.set_ylim([min(X[:,0]),max(X[:,0])])
 
ax.set(title=VarNames[ivar1]+" (" + VarUnits[ivar1] + ")" +" at "+tstamp+"ns",xlabel="Z (cm)",ylabel="R (cm)")

# add structure
if os.path.isfile(dirname+'struct.p4'):
    (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
    lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
    lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
    ax.add_collection(lc)
    if aspect: 
        ax.set_aspect('equal');
    else:
        ax.set_aspect('auto')
    if flip:
        lines=[[(za[i],-xa[i]),(zb[i],-xb[i])] for i in range(len(xa))]
        lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
        ax.add_collection(lc)
        ax.set_aspect('equal')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
CB =plt.colorbar(im, cax=cax)
#cax.yaxis.major.locator.set_params(nbins=8) 

if sliceX!= None and sliceX<=xmax and sliceX>=xmin: # plot slice along Z
    data2=np.zeros((1,len(Z[0,:])));
    for j in range(len(Z[0,:])):
        data2[0,j]=np.interp(sliceX,Xc[:,0],dat[:,j]);
    f2, ax2 = plt.subplots()
    ax2.plot(Z[0,:],data2[0,:],linewidth=2)
#     ax2.plot(Z[0,:],np.ones(len(data2[0,:]))*np.mean(data2[0,:]),linewidth=2)
    ax2.set(title="mass fraction " + VarNames[ivar1] +" within R = %.2fcm" %(sliceX) + " at "+tstamp+"ns")
    ax2.set_xlim([zmin,zmax]);
    ax2.set(xlabel='Z (cm)', ylabel="ratio")
    if logScale: ax2.set(yscale='log')
    if not os.path.exists(dirname+"plots"):  os.makedirs(dirname+"plots")
    f2.savefig(dirname +"plots/" + "sweeping_" + VarNames[ivar1] + "_%.1f" %(time) + '_sliceX.png',dpi=200, bbox_inches="tight")

if not os.path.exists(dirname+"plots"):
    os.makedirs(dirname+"plots")

fig.savefig(dirname + "plots/" + "sweeping_" + VarNames[ivar1] + "_%.1f" %(time) + '.png',dpi=200, bbox_inches="tight")
if not noShow: plt.show();


exit()

