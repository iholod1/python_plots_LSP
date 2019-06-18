#!/usr/bin/env python
# script to plot from part*.p4
# massi and chargei must be adjusted for ion species
# I. Holod, 04/11/17

import sys, argparse
import os.path
import math
import numpy as np
import scipy as scp
import scipy.io


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import readXDRpart, readXDRsclr, readXDRstruct
from hist1d import hist1dlin, hist1dlog
from energy import kin_en, theta_ang, velocity, getMaxwellian, weighted_avg_and_std

class cell(object):
    __slots__=['x','z','vx','vz','w','en','ppc','nSigmaE']

# load Cross Section data
DDCross=scipy.io.loadmat('/g/g19/holod1/solid_target/DDCross.mat')
ddc=DDCross.get('DDCrossSection')
ddc[0].dtype
csE=ddc[0][0][0][:,0];
csS=ddc[0][0][1][:,0];
csE=np.append(0,csE);
csS=np.append(0,csS);

##### CONTROL PARAMETERS #####
num=1
sp = 1
ivar=1; # set variable to plot
flip=0; # flip plot symmetrically
dirname=''

class C(object):
  pass
arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-i', type=int, help='time index');
parser.add_argument('-s', type=int, help='species');
parser.add_argument('-f', action='store_true');
parser.parse_args(namespace=arg)
if arg.i: num = arg.i; dirname='';
if arg.s != None: sp = int(arg.s)-1;
if arg.f: flip = True;

fname=dirname+'part'+str(num)+'.p4'
print fname;

maxSpecies = 11
migen = 15000e-6
erstm=0.511; #positron rest mass (MeV)
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

charge=[charge*1.609e-13 for charge in charge] # converts charge to mcoulombs

#############################

(tt,Part)=readXDRpart(fname)

tstamp =  "%.2f" % tt
print "t=", tstamp, "ns"

nSpecies=len(Part); nsp=[0]*nSpecies;
nsp = [len(Part[i][0]) for i in range(nSpecies)] # number of particles per species
print "numbers of particles per species", nsp

fname1='sclr'+str(num)+'.p4'
print fname1
(xs,ys,zs,Vars,VarNamess,VarUnitss,times)=readXDRsclr(dirname+fname1)

# particle weight
weight=[Part[sp][0][i]/charge[sp] for i in range(nsp[sp])]
# particle velocity
(vx,vy,vz)=velocity(Part[sp][4],Part[sp][5],Part[sp][6]); # beta
 # kinetic energy
ke=kin_en(Part[sp][4],Part[sp][5],Part[sp][6],mass[sp]);

# number of particles per cell
nx=len(xs); nz=len(zs);
ndim=2;
if nz<2: ndim=1;
dx=(xs[-1]-xs[0])/(nx-1); # assuming uniform cells
if ndim>1: dz=(zs[-1]-zs[0])/(nz-1);
grid=[];
for i in range(nx):
  grid.append([]);
  for j in range(nz):
    grid[i].append([]);
    grid[i][j]=cell();
    grid[i][j].ppc=0;
    grid[i][j].nSigmaE=0;
    grid[i][j].x=xs[i];
    grid[i][j].z=zs[j];
    grid[i][j].w=[];
    grid[i][j].vx=[];
    grid[i][j].vz=[];

j=0;
for ip in range(nsp[sp]):
  i=int(math.floor(Part[sp][1][ip]/dx));
  if ndim>1: j=int(math.floor(Part[sp][3][ip]/dz));
  grid[i][j].vx.append(vx[ip]);
  grid[i][j].vz.append(vz[ip]);
  grid[i][j].w.append(weight[ip]);
  grid[i][j].ppc += 1/sratio[sp];
  grid[i][j].nSigmaE += weight[ip]*np.interp(ke[ip],csE,csS)/sratio[sp];

  

#pen=weight[i]*0.5*erstm*mass[sp]*vz[i]**2 # MeV

### plot ppc ####
fig,ax = pl.subplots()
dat=np.zeros((nx,nz));
for i in range(nx): 
  for j in range(nz): dat[i,j]=grid[i][j].nSigmaE;

im = ax.imshow(dat[:,:], interpolation='nearest',
                origin='lower', extent=(min(zs), max(zs), min(xs), max(xs)))
CB = plt.colorbar(im)  
ax.set(title="ppc("+str(sp+1)+") at "+tstamp+"ns")

# add structure
(xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
lc = mc.LineCollection(lines, linewidths=2)
ax.add_collection(lc)
ax.set_aspect('auto')

plt.show()


exit()
