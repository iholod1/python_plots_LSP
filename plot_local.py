#!/usr/bin/env python
# script to plot from part*.p4
# massi and chargei must be adjusted for ion species
# I. Holod, 04/20/17

import sys, getopt
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import readXDRpart, readXDRsclr, readXDRstruct
from hist1d import hist1dlin, hist1dlog
from energy import kin_en, theta_ang, velocity, getMaxwellian, weighted_avg_and_std

erstm=0.511; #positron rest mass (MeV)

##### CONTROL PARAMETERS #####
# set step #
num=30258
dirname=''
sp = 3

opts, args = getopt.getopt(sys.argv[1:],"i:s:")
for opt, arg in opts: 
  if opt=="-i": num = arg; dirname='';
  if opt=="-s": sp = int(arg)-1;

# set diagnostics species index -1


fname=dirname+'part'+str(num)+'.p4'
print fname;

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

charge=[charge*1.609e-13 for charge in charge] # converts charge to mcoulombs
rmasse = 0.5110 # positron (electron) rest mass MeV
#############################


fname1='sclr'+str(num)+'.p4'
print fname1
(xs,ys,zs,sVar,sVarNames,VarUnits,tt)=readXDRsclr(dirname+fname1)
tstamp =  "%.2f" % tt
print "t=", tstamp, "ns"

sDen=np.squeeze(sVar[sVarNames.index('RhoT'+str(sp+1))]);
sTemp=np.squeeze(sVar[sVarNames.index('Temp'+str(sp+1))]);

dum=np.argwhere(sDen>0.8*np.max(sDen.flatten())); # set selection criterion

indxz=[(dum[i][0],dum[i][1]) for i in range (len(dum))];

nx=len(xs); nz=len(zs);
ndim=2;
if nz<2: ndim=1;
dx=(xs[-1]-xs[0])/(nx-1); 
if ndim>1: dz=(zs[-1]-zs[0])/(nz-1);

(tt,Part)=readXDRpart(fname)
nSpecies=len(Part); nsp=[0]*nSpecies;
nsp = [len(Part[i][0]) for i in range(nSpecies)] # total number of particles per species
print "numbers of particles per species", nsp

xp=Part[sp][1][:];
zp=Part[sp][3][:];
indp=[];
for ip in range(nsp[sp]):
  ix=int(math.floor(xp[ip]/dx));
  iz=int(math.floor(zp[ip]/dz));
  if ((ix,iz) in indxz): indp.append(ip);

print len(indp) #number of selected particles


(vx,vy,vz)=np.array(velocity(Part[sp][4],Part[sp][5],Part[sp][6]));
weight=np.array([Part[sp][0][i]/charge[sp] for i in range(nsp[sp])]);

vx=vx[indp];vy=vy[indp];vz=vz[indp];weight=weight[indp];

(mux,vtx)= weighted_avg_and_std(vx,weight);
(muy,vty)= weighted_avg_and_std(vy,weight);
(muz,vtz)= weighted_avg_and_std(vz,weight);

(x,y)=hist1dlin(vz,weight,200)

fm=getMaxwellian(x,vtz,muz)

norm=sum(y);norm1=sum(fm);
y1 = [dum*norm/norm1 for dum in fm]

f1, ax1 = plt.subplots()
ttl =  "f(vz) species %d at t=%.2f ns" %(sp+1,tt)

ax1.plot(x,y,x,y1,linewidth=2)
#ax1.plot(x,y,linewidth=2)
#ax1.set(title=ttl, xlabel='v/c', ylabel='')
ax1.set(title=ttl, xlabel='v/c', ylabel='',yscale='log')
ax1.text(0.95*ax1.get_xlim()[0],1.1*ax1.get_ylim()[0],fname)
ax1.set_ylim([1e12,1.05*np.max(y)])
#ax1.set_xlim([-0.005,0.005])
plt.show()
f1.savefig(dirname + ttl.replace(" ", "_") + '.png',dpi=4*f1.dpi)

exit()

