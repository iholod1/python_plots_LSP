#!/usr/bin/env python
# script to plot from history.p4
# I. Holod, 07/13/16

from sys import exit
import math
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import read_hist

##### CONTROL PARAMETERS #####
#fname='/p/lscratche/holod1/miniDPF/c120_v40/short_05/L70/cond2/history.p4'
fname='/p/lscratche/holod1/miniDPF/c120_v40/L70/anode2/100/history.p4'
#fname='history.p4'


# particle masses in positron units
#mass=1.0
mass=3.672e3 # D 
#mass=7.292e3 # He
#mass=1.825e+03; # neutron

# particle charges in positron units
#charge=-1.0;
charge=1.0;

charge=charge*1.609e-13; # converts charge to mcoulombs
#############################

(title,nProbes,Vars,Units,tt,dat)=read_hist(fname)
print title
print Units

probe=79;
x=dat[:,0];
i4=dat[:,probe];

v4=dat[:,78];

# derivative of current
#di4=[];
#for i in range(len(x)-1):
#  di4.append(abs((i4[i+1]-i4[i])/(x[i+1]-x[i])));
#x=x[:-1];y=di4;
##imax=y.tolist().index(max(y))
##print y[imax], x[imax], x[imax-1]
##y=[y[i]/max(abs(y)) for i in range(len(y))];
#l4=[1.0e3*v4[i]/di4[i] for i in range(len(di4))];
#y=l4;


# integrated
#y[0]=0.0;
#for i in range(1,len(x)-1):
#  y[i]=y[i-1]+0.5*(y[i+1]+y[i])*(x[i+1]-x[i]);
#y=y[0:-2];
#x=x[0:-2];

# integrated square
#y[0]=0.0;
#for i in range(1,len(x)-1):
#  y[i]=y[i-1]+(0.5*(y[i+1]+y[i]))**2*(x[i+1]-x[i]);
#y=y[0:-2];
#x=x[0:-2];

y=i4;

f,ax = plt.subplots()
ax.plot(x,y,linewidth=2)
ttl =  "f(E) from "+fname
ttl =  "History of probe %d" %(probe)
#ax.set(title=ttl, xlabel='E (MeV)', ylabel='#/MeV',yscale='log',xscale='log')
ax.set(title=ttl, xlabel='time', ylabel=Units[probe])
ax.set_xlim([0,600])
ax.text(0.95*ax.get_xlim()[0],1.1*ax.get_ylim()[0],fname)

probe1=51;
x1=dat[:,0];
y1=dat[:,probe1];
y1=[abs(y1[i])/max(abs(y1)) for i in range(len(y1))];
#ax.plot(x1,y1,linewidth=2)

fid=open('/g/g19/holod1/miniDPF/current_shot_2016101300033m.dat','r')
line=fid.readline();
dum=np.array(line.split(',')).astype(float);
tt1=dum[0];dat1=dum[1:];
for line in fid:
        dum=np.array(line.split(',')).astype(float);
        tt1=np.hstack((tt1,dum[0]));
        dat1=np.vstack((dat1,dum[1:]));
tt1=[tt1[i]*1.0e9-890 for i in range(len(tt1))]
#f,ax1 = plt.subplots()
ax.plot(tt1,dat1,linewidth=2)
#ax1.set_xlim([0,1200])

plt.show()



#fid = open('current.dat', 'w')
#for i in range(len(x)):
#  dum="%.2f %.2f\n" %(x[i],y[i])
#  fid.write(dum)
#fid.close()



