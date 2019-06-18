#!/usr/bin/env python3
# script to plot from part*.p4
# massi and chargei must be adjusted for ion species
# I. Holod, 07/05/16

import sys, argparse
import os.path
import math
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
from PIL import Image

from read_xdr import readXDRpart1, readXDRsclr, readXDRstruct
from read_deck import readDeck
#from hist1d import hist1dlin, hist1dlog
from energy import kin_en

tStart = time.time()
erstm=0.511; #positron rest mass (MeV)

##### CONTROL PARAMETERS #####
dirname=os.getcwd() + '/';
class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description='Process arguments');
parser.add_argument('-i', type=int, help='time index');
parser.add_argument('-s', type=int, help='species');
parser.add_argument('-minx', type=float, help='minimum x');
parser.add_argument('-maxx', type=float, help='maximum x');
parser.add_argument('-minz', type=float, help='minimum z');
parser.add_argument('-maxz', type=float, help='maximum z');
parser.add_argument('-cmap', type=str, help='colormap');
parser.add_argument('-cmin', type=float, help='min value');
parser.add_argument('-cmax', type=float, help='max value');
parser.add_argument('-log', action='store_true', help="log scale");

parser.parse_args(namespace=arg)

num = int(arg.i) if arg.i != None else 1;
spc = int(arg.s)-1 if arg.s != None else 0;

minx = float(arg.minx) if arg.minx != None else None 
maxx = float(arg.maxx) if arg.maxx != None else None 
minz = float(arg.minz) if arg.minz != None else None 
maxz = float(arg.maxz) if arg.maxz != None else None 

cmap = arg.cmap  if arg.cmap != None else None 
cmin = float(arg.cmin) if arg.cmin != None else None 
cmax = float(arg.cmax) if arg.cmax != None else None 
logScale = arg.log;


fname=dirname+'part'+str(num)+'.p4'
print(fname);

maxSpecies = 11

# particle masses in positron units
masse=1.0
massi=3.672e3 # 7.292e3 # D/He mass
massn=1.825e+03; # neutron mass

# particle charges in positron units
chargee=-1.0;chargei=1.0;chargen=1.0;

mass=[massi]*maxSpecies;charge=[chargei]*maxSpecies;
sratio=[1.0]*maxSpecies;

indn=[9] # deck index for neutron species - 1
inde=[0,2,4] # electron deck indexes [3, 5] - 1
indi=[1,3,5] # ion deck indexes [4, 6] - 1

for i in inde: mass[i]=masse; charge[i]=chargee; sratio[i]=0.1;
for i in indi: mass[i]=massi; charge[i]=chargei; sratio[i]=0.1;
for i in indn: mass[i]=massn; charge[i]=chargen; sratio[i]=1.0;


str1="species"+str(spc+1)
mass[spc]=readDeck(dirname,str1,"mass");
print("mass =%f" %(mass[spc]))
sratio[spc]=readDeck(dirname,str1,"selection_ratio");
print("selection ratio = %f" %(sratio[spc]))

charge=[charge*1.609e-13 for charge in charge] # converts charge to mcoulombs
rmasse = 0.5110 # positron (electron) rest mass MeV
#############################

fname1='sclr'+str(num)+'.p4'
print(fname1)
(xs,ys,zs,sVar,sVarNames,VarUnits,tt)=readXDRsclr(dirname+fname1)
tstamp =  "%.2f" % tt
print("t=" + tstamp + "ns")

(tt,Part)=readXDRpart1(fname,spc+1)
nSpecies=len(Part); nsp=[0]*nSpecies;
nsp = [len(Part[i][0]) for i in range(nSpecies)] # total number of particles per species
print("numbers of particles per species", nsp)

npart=nsp[spc];
part = np.array(Part[spc]);
weight = part[0,:]/charge[spc];
ke=kin_en(part[4,:],part[5,:],part[6,:],mass[spc]);


# number of particles per cell
nx=len(xs); nz=len(zs);
dxs = np.diff(xs)
dxs = np.append(dxs,dxs[-1])
xs = xs + dxs

dzs = np.diff(zs)
dzs = np.append(dzs,dzs[-1])
zs = zs + dzs

minx=minx if (minx!=None) and (minx>min(xs)) and (minx<=max(xs)) else min(xs)
maxx=maxx if (maxx!=None) and (maxx>min(xs)) and (maxx<=max(xs)) else max(xs)
minz=minz if (minz!=None) and (minz>min(zs)) and (minz<=max(zs)) else min(zs)
maxz=maxz if (maxz!=None) and (maxz>min(zs)) and (maxz<=max(zs)) else max(zs)

ppc=np.zeros((nx,nz));
epc=np.zeros((nx,nz)); # energy per cell
j=0;
ntot = 0

for ip in range(npart):
    xp=part[1,ip];
    zp=part[3,ip];
#     if xp>maxx or xp<minx or zp>maxz or zp<minz: continue
    i=int(min(np.argwhere(xs>=xp)[0][:]));
    j=int(min(np.argwhere(zs>=zp)[0][:]));
    ppc[i,j] += 1/sratio[spc];
    ntot += 1
    epc[i,j] += ke[ip]*weight[ip]*1.6e-13/sratio[spc];

max_value=max(ppc.flatten()); max_index=np.where(ppc==max_value);
im=max_index[0][0]; jm=max_index[1][0];
xm=xs[im]; zm=zs[jm];
print("max ppc of specie %d is %d @ R = %f Z = %f" %(spc+1,max_value,xs[im],zs[jm]))
print("total number of particles in selected region = %d" %(ntot/sratio[spc]))
print("total kinetic energy (J) of specie %d KE =%.4e" %(spc+1,np.sum(epc.flatten())))

sDen=sVar[sVarNames.index('RhoT'+str(spc+1))].flatten()
#print "energy (J) of specie",spc+1," at n>1e17: ", np.sum(epc.flatten()[np.argwhere(sDen>1.e17)])
#print "energy (J) of specie",spc+1," at r>0.76: ", np.sum(epc[0,np.argwhere(xs>0.76),:].flatten())
#print "energy (J) of specie",spc+1," at r<0.76: ", np.sum(epc[0,np.argwhere(xs<0.76),:].flatten())



### plot the results ####
fig,ax = plt.subplots()
dat=ppc;

im = ax.pcolor(zs,xs,dat);
ax.set_xlim([minz,maxz])
ax.set_ylim([minx,maxx])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
CB = plt.colorbar(im, cax=cax)
ax.set(title="ppc("+str(spc+1)+") at "+tstamp+"ns")


# add structure
(xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
lc = mc.LineCollection(lines, linewidths=2)
ax.add_collection(lc)
ax.set_aspect('auto')
#ax.set_aspect('equal')

# plot energy per cell
f2,ax2 = plt.subplots()
dat=epc;
im2 = ax2.pcolor(zs,xs,dat[:,:]);
ax2.set_xlim([minz,maxz])
ax2.set_ylim([minx,maxx])
ax2.set(title="epc("+str(spc+1)+") (J) at "+tstamp+"ns")
# add structure
(xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
lc = mc.LineCollection(lines, linewidths=2)
ax2.add_collection(lc)
ax2.set_aspect('auto')

fig.savefig(dirname + 'ppc' + str(spc+1) + '_'+ str(num) + '.png',dpi=200, bbox_inches="tight")

tEnd = time.time()
print("elapsed time (s): %.2f" %(tEnd-tStart))

#f2.savefig(dirname + 'epc' + str(spc+1) + '_'+ str(num) + '.png',dpi=200)

plt.show()


