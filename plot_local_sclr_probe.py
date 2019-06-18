#!/usr/bin/env python
# script to plot from file
# I. Holod, 04/05/17

import sys, getopt, os
import math
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

dirname=os.getcwd() + '/';

probe=1;

f, ax = plt.subplots()


fname = dirname+"local_probe"+"%d" %(probe)+".out"
headr=[];tt=[]; vx=[]; vy=[]; vz=[];
with open(fname) as infile:
  for line in infile:
    if line[0]=="#": headr.append(line.split("\t"));
    else: 
      tt.append(line.split()[0]);
      vx.append(line.split()[1]);

x=np.array(tt).astype(float);
y=np.array(vx).astype(float);

ind=np.argwhere((x>0)&(x<10000))
x=x[ind].reshape((ind.shape[0],1))
y=y[ind].reshape((ind.shape[0],1))

ax.plot(x,y,linewidth=2)
ax.plot(x,np.ones(len(y))*np.mean(y),linewidth=2,label="%.2e" %(np.mean(y)))
ax.set_xlim([min(x),max(x)])

ttl =  headr[0][2]+" "+headr[0][4][:-1];
ax.set(title=ttl, xlabel=headr[0][1], ylabel=headr[0][3])
ax.legend(loc='best', shadow=True)

f.savefig(dirname + ttl.replace(" ", "_").replace(",", "_").replace("(", "_").replace(")", "_") + '.png',dpi=200)

plt.show()

exit()
