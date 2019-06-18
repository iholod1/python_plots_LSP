#!/usr/bin/env python
# script to plot PDE of particles in specific region (from part*.p4)
# massi, chargei, and selection ratio must be defined
# I. Holod, 06/09/17

import os, sys, getopt
import numpy as np
import matplotlib.pyplot as plt

dirname=os.getcwd() + '/';
fid=dirname+"probe66_hist.out"

headr=[];tt=[];dat=[];
with open(fid) as infile:
  for line in infile:
    if line[0]=="#": headr.append(line.split("\t"));
    else: 
      tt.append(line.split()[0]);
      dat.append(line.split()[1:][0]);
x=np.array(tt).astype(float);
y=np.array(dat).astype(float);

dt=min(np.diff(x));
nT=(max(x)-min(x))/dt+1;
newT=np.linspace(min(x),max(x),nT);
newY=np.interp(newT,x,y);
yfft=np.abs(np.fft.fft(newY))

maxf=1e9/dt;

freq=np.linspace(0,maxf,len(yfft));

f, ax = plt.subplots()
ax.plot(freq,yfft,linewidth=2)
ttl =  "FFT " + headr[0][1]
#ax.set(title=ttl, xlabel='E (MeV)', ylabel='#/MeV',yscale='log',xscale='log')
#ax.set(title=ttl, xlabel='time', ylabel=Units[probe])
ax.set_xlabel("f (Hz)")
ax.set_title(ttl);
ax.set_xlim([0, max(freq)/2]);

#ax.text(0.95*ax.get_xlim()[0],1.1*ax.get_ylim()[0],fname)
plt.show();
f.savefig(dirname + ttl.replace(" ", "_") + '.png',dpi=200)
