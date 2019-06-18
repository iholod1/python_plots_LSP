#!/usr/bin/env python
# script to plot PDE of particles in specific region (from part*.p4)
# massi, chargei, and selection ratio must be defined
# I. Holod, 06/09/17

import os, sys, getopt
import math
import numpy as np

import matplotlib

matplotlib.use('Agg') # to run in pdebug (not using $DISPLAY)

import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import readXDRpart
from read_deck import readDeck
from hist1d import hist1dlin, hist1dlog
from energy import kin_en

erstm=0.511; #positron rest mass (MeV)

##### CONTROL PARAMETERS #####
# set step #
num=1
dirname=os.getcwd() + '/';
spc = 3

x0=0; x1=0.1;
z0=30.0; z1=32.0;
beam=1;
emin=0.025;
emax=1.0;
ylim=[1e12,1e16]

opts, args = getopt.getopt(sys.argv[1:],"i:s:b:d:u:r:l:")
for opt, arg in opts: 
  if opt=="-i": num = arg;
  if opt=="-s": spc = int(arg)-1;
  if opt=="-b": beam = int(arg);
  if opt=="-d": x0 = float(arg);
  if opt=="-u": x1 = float(arg);
  if opt=="-l": z0 = float(arg);
  if opt=="-r": z1 = float(arg);

# set diagnostics species index -1

fname=dirname+"part"+str(num)+".p4"
print(fname)

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
print "mass =", mass[spc]
sratio[spc]=readDeck(dirname,str1,"selection_ratio");
print "selection ratio =", sratio[spc]

charge=[charge*1.609e-13 for charge in charge] # converts charge to mcoulombs
rmasse = 0.5110 # positron (electron) rest mass MeV
#############################


(tt,Part)=readXDRpart(fname)
nSpecies=len(Part); nsp=[0]*nSpecies;
nsp = [len(Part[i][0]) for i in range(nSpecies)] # total number of particles per species
print "numbers of particles per species", nsp

# select particle by location
xp=Part[spc][1][:];
zp=Part[spc][3][:];
indp=[];
for ip in range(nsp[spc]):
  if xp[ip]>x0 and xp[ip]<x1 and zp[ip]>z0 and zp[ip]<z1: indp.append(ip);

print "number of selected particles =", len(indp)
if len(indp)==0: 
  print fname,": no particles in the region"; exit()

weight=np.array([Part[spc][0][i]/charge[spc]/sratio[spc] for i in range(nsp[spc])]);
ke=np.array(kin_en(Part[spc][4],Part[spc][5],Part[spc][6],mass[spc])); # kinetic energy
# pick selected indexes
weight=weight[indp].flatten();
ke=ke[indp].flatten();

if beam:
  indp=np.argwhere(ke>emin);
  if not len(indp)>0:
    print fname,": no particles in the energy range"; exit();
  weight=weight[indp].flatten();
  ke=ke[indp].flatten();
  ke[np.argwhere(ke>emax)]=emax; # cap the energy
  (x,y)=hist1dlog(ke,weight,40,minVal=emin,maxVal=emax) # log binning
else:
  indp=np.argwhere(ke<emin);
  if not len(indp)>0:
    print fname,": no particles in the energy range"; exit();
  weight=weight[indp].flatten();
  ke=ke[indp].flatten();
  (x,y)=hist1dlin(ke,weight,20,minVal=0,maxVal=emin) # lin binning

ttl =  "pdf species %d at t=%.2f ns in x0=%.1f x1=%.1f z0=%.1f z1=%.1f" %(spc+1,tt,x0,x1,z0,z1)
f1, ax1 = plt.subplots()
#ax1.plot(x,y,linewidth=2)
ax1.set_ylim(ylim)

if beam:
#  ax1.bar(x,y,wdth,align="center")
  wdth=np.diff(10**(np.linspace(np.log10(emin),np.log10(emax),41)))
  ax1.bar(x,y,wdth,log=True,ec="k", align="center")
  ax1.set(title=ttl, xlabel='E (MeV)', ylabel='#/MeV',xscale='log',yscale='log')
  ax1.set_xlim([emin, emax])
else:
  wdth=emin/float(len(x));
  ax1.bar(x,y,wdth,align="center")
  ax1.set(title=ttl, xlabel='E (MeV)', ylabel='#/MeV',yscale='log')
  ax1.set_xlim([0.0,emin])


#plt.show(); exit()

if not os.path.exists(dirname+"plots"):
  os.makedirs(dirname+"plots")

f1.savefig(dirname +"plots/" + ttl.replace(" ", "_") + ".png",dpi=200)

fout = open(dirname + "plots/" + ttl.replace(" ", "_") + ".out", 'w')
fout.write("# Energy (eV) weight\n");
for i in range(len(ke)):
  fout.write("%.8e %.8e" %(ke[i], weight[i]));
  fout.write("\n");
fout.close()

exit()

