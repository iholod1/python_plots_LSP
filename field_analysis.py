#!/usr/bin/env python
# analysis of extended history file
# I. Holod, 06/26/17

import sys, getopt, os
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import read_hist
###############################
from scipy.signal import butter, filtfilt
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
################################



dirname=os.getcwd() + "/";
fname=dirname+"history.p4"

(title,nProbes,Vars,Units,tt,dat)=read_hist(fname)

t=np.array(dat[:,0]);
nT=len(t);

z0=3.6
z1=4.2
r0=0.05
r1=0.1

nZ=int((z1-z0)/0.01) + 1
nR=int((r1-r0)/0.02) + 1

rR=np.linspace(r0,r1,nR);
rZ=np.linspace(z0,z1,nZ);

densI=np.zeros((nT,nR,nZ));
densE=np.zeros((nT,nR,nZ));
Ez=np.zeros((nT,nR,nZ));
Bx=np.zeros((nT,nR,nZ));
By=np.zeros((nT,nR,nZ));
Bz=np.zeros((nT,nR,nZ));

p=3;
for i in range(nR):
  for j in range(nZ):
    densI[:,i,j]=np.array(dat[:,p+1]);
    densE[:,i,j]=np.array(dat[:,p+2]);
    Ez[:,i,j]=np.array(dat[:,p+3]);
    Bx[:,i,j]=np.array(dat[:,p+4]);
    By[:,i,j]=np.array(dat[:,p+5]);
    Bz[:,i,j]=np.array(dat[:,p+6]);
    p+=6;


#dat2d=np.squeeze(Ez[-1,:,:]);
#fig, ax = pl.subplots(figsize=(8,8))
#im = ax.imshow(dat2d, interpolation='none',origin='lower', extent=(min(rZ), \
#               max(rZ), min(rR), max(rR)))
#CB = plt.colorbar(im) 
#plt.show();


#y=np.squeeze(By[:,-1,-1]);
#y=np.mean(np.squeeze(Ez[:,0,:]),1) # mean value at fixed x
y=np.mean(Ez.reshape((Ez.shape[0], -1)),axis=1)
#y=np.squeeze(Ez[:,0,23])
cutoff = 1500; fs = 50000
ys1 = butter_lowpass_filtfilt(y, cutoff, fs)
#ys1=ys1/np.max(ys1)

#y=np.squeeze(Ez[:,0,24]-Ez[:,0,23])
#cutoff = 1500; fs = 50000
#ys1 = butter_lowpass_filtfilt(y, cutoff, fs)
#ys1=ys1/np.max(ys1)

#y=np.squeeze(densI[:,0,23]-densE[:,0,23])
#cutoff = 1500; fs = 50000
#ys2 = butter_lowpass_filtfilt(y, cutoff, fs)
#ys2=ys2/np.max(ys2)
#t2=t;

#y=np.squeeze(By[:,0,23])
#cutoff = 1500; fs = 50000
#ys2 = butter_lowpass_filtfilt(y, cutoff, fs)
#ys2=np.diff(ys2)/np.diff(t)
#ys2=ys2/np.max(ys2)
#t2 = t[1:];


fig, ax = plt.subplots()
#ax.plot(t,y,linewidth=0.5)
ax.plot(t,ys1,linewidth=1)
#ax.plot(t2,ys2,linewidth=2)
ax.set_xlim([440,500]);
plt.show();

maxEz=np.zeros((nZ));
y=np.squeeze(Ez[:,0,:])
dum=np.squeeze(y[-1,:]);

for i in range(nT):
  dum=np.squeeze(y[i,:]);
  [j]=np.argwhere(dum==np.max(dum))[0];
  maxEz[j]+=1

print "max j =", np.argwhere(maxEz==np.max(maxEz))[0]

#it=np.argwhere(t>501)[0][0];
#dat2d=np.squeeze(Ez[it,:,:]);
#fig, ax = pl.subplots(figsize=(8,8))
#im = ax.imshow(dat2d, interpolation='none',origin='lower', extent=(min(rZ), \
#               max(rZ), min(rR), max(rR)))
#CB = plt.colorbar(im) 
#plt.show();

## FFT
#dt=min(np.diff(t));
#nT=(max(t)-420)/dt+1;
#newT=np.linspace(420,max(t),nT);
#newY=np.interp(newT,t,y);
#yfft=np.abs(np.fft.fft(newY))
#maxf=1e9/dt;
#freq=np.linspace(0,maxf,len(yfft));

#f, ax = plt.subplots()
#ax.plot(freq,yfft,linewidth=2)
#ttl =  "FFT"
#ax.set_xlabel("f (Hz)")
#ax.set_title(ttl);
#ax.set_xlim([0, max(freq)/2]);
#plt.show();
