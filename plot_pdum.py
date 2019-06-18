#!/usr/bin/env python
# script to plot from part*.p4
# massi and chargei must be adjusted for ion species
# I. Holod, 04/11/17

import sys, argparse
import os.path
import math
import numpy as np
import scipy as scp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import readXDRpart, readXDRsclr
from hist1d import hist1d
from energy import energy, kin_en, theta_ang, velocity, getMaxwellian, weighted_avg_and_std
from read_deck import readDeck

# os.chdir(os.path.join("C:",os.sep, "Users","holod1","Documents","data","test_thermonuclear_fusion","beam"))
mydir=os.getcwd()
print(mydir)

#class species(object):
#    __slots__=['ind','charge','mass','temp','vt']
#erstm=0.511e6; #positron rest mass (eV)
#ele=species()
#ele.ind=0
#ele.mass=1.0 # positron units
#ele.charge=-1.0 # positron units
#ele.temp=1000.0 # eV
#ele.vt=np.sqrt(ele.temp/erstm/ele.mass)
#print ele.vt


##### CONTROL PARAMETERS #####
# num=1
spc = 1
ivar=1; # set variable to plot
flip=0; # flip plot symmetrically
dirname=mydir

class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments')
parser.add_argument('-i', type=int, help='time index')
parser.add_argument('-s', type=int, help='species')
parser.add_argument('-minx', type=float, help='minimum x')
parser.add_argument('-maxx', type=float, help='maximum x')
parser.add_argument('-minz', type=float, help='minimum z')
parser.add_argument('-maxz', type=float, help='maximum z')
parser.add_argument('-f', action='store_true');
parser.parse_args(namespace=arg)

if arg.i: num = arg.i
if arg.s != None: spc = int(arg.s)-1;
if arg.f: flip = True;

minx = float(arg.minx) if arg.minx != None else None 
maxx = float(arg.maxx) if arg.maxx != None else None 
minz = float(arg.minz) if arg.minz != None else None 
maxz = float(arg.maxz) if arg.maxz != None else None 

fname=os.path.join(dirname,'part'+str(num)+'.p4')
print(fname)

rmasse = 0.5110 # MeV
clight = 2.99e10 # cm/s
mev2j = 1.602e-13
mass=1.0
charge=1.0
sratio=1.0

str1="species"+str(spc+1)
mass=readDeck(dirname,str1,"mass");
charge=readDeck(dirname,str1,"charge");
sratio=readDeck(dirname,str1,"selection_ratio");
if abs(charge) < 0.5: charge = 1.0;

print("mass = %f" %(mass))
print("charge = %f" %(charge))

charge=charge*1.609e-13; # converts charge to mcoulombs

#############################

(tt,Part)=readXDRpart(fname)

tstamp =  "%.2f" % tt
print("t=" + tstamp + "ns")


nSpecies=len(Part) 
nsp=[0]*nSpecies
nsp = [len(Part[i][0]) for i in range(nSpecies)] # number of particles per species
print("numbers of particles per species {}".format(nsp))

Var = np.array(Part[spc])
print(Var.shape)

minx = minx if minx !=None else np.min(Var[1,:])
maxx = maxx if maxx !=None else np.max(Var[1,:])
minz = minz if minz !=None else np.min(Var[3,:])
maxz = maxz if maxz !=None else np.max(Var[3,:])

minz = 0.3
maxz = 0.6

indx=np.argwhere((Var[1,:]>=minx)&(Var[1,:]<=maxx)
                 &(Var[3,:]>=minz)&(Var[3,:]<=maxz))
Var = Var[:,indx[:,0]]

# particle weight
weight=Var[0,:]/charge

dum = energy(Var[4,:],Var[5,:],Var[6,:],mass,"ke","ang","vx","vy","vz");
ke = dum["ke"] #MeV
ang = dum["ang"]
vx = dum["vx"] # cm/s
vy = dum["vy"]
vz = dum["vz"]


fname1 = os.path.join(dirname,'sclr'+str(num)+'.p4')
print(fname1)
(Xs,Ys,Zs,sVar,sVarNames,sVarUnits,stime)=readXDRsclr(fname1)

indx=np.argwhere((Xs>=minx)&(Xs<=maxx))
indx=indx.reshape(indx.shape[0],1,1)
indz=np.argwhere((Zs>=minz)&(Zs<=maxz))
indz=indz.reshape(1,1,indz.shape[0])

sTemp=sVar[sVarNames.index('Temp'+str(spc+1))][indx,:,indz].flatten();
sDen=sVar[sVarNames.index('RhoT'+str(spc+1))][indx,:,indz].flatten();
sPres=sVar[sVarNames.index('Pres'+str(spc+1))][indx,:,indz].flatten();

saTemp=np.average(sTemp);
wTemp=np.sum([sTemp[i]*sDen[i] for i in range(len(sTemp))])/np.sum(sDen)
maxDen=np.max(sDen);
maxTemp=np.average([sTemp[np.where(sDen>0.75*maxDen)]]);

#for i in range(len(Var)): print i, VarNames[i]
print("*** from sclr ***")
print('mean T (eV) = %.4e' %(saTemp))
#print "average thermal energy (eV) =", 1.5*saTemp
print("mean density weighted temperature (eV) = %.4e" %(wTemp)) 
#print "temperature at max density (eV) =", sTemp[np.argmax(sDen)]
print("temperature at max density (eV) = %.4e" %(maxTemp))

(mux,vtx)= weighted_avg_and_std(vx,weight);
(muy,vty)= weighted_avg_and_std(vy,weight);
(muz,vtz)= weighted_avg_and_std(vz,weight);
print("mux,muy,muz = %.4e, %.4e, %.4e (cm/sec)" %(mux,muy,muz))

# totalE=[weight[i]*0.5*mass*(vx[i]**2+vy[i]**2+vz[i]**2) for i in range(len(vx))];
totalE = weight*0.5*mass*(vx**2 + vy**2 + vz**2)/clight**2
thermalE = weight*0.5*mass*((vx-mux)**2+(vy-muy)**2+(vz-muz)**2)/clight**2
directE=weight*0.5*mass*(mux**2+muy**2+muz**2)/clight**2

print("*** from part ***")
print("total particle KE (J) = %.4e" %(np.sum(totalE)*rmasse*mev2j/sratio))
# print("total particle KE (J) =", np.sum(weight*ke)*mev2j/sratio)

print("total thermal KE (J) = %.4e" %(np.sum(thermalE)*rmasse*mev2j/sratio))
print("average thermal E per particle (eV) %.4e" %(1.0e6*np.sum(thermalE)/np.sum(weight)))
print("total directional KE (J) = %.4e" %(np.sum(directE)*rmasse*mev2j/sratio))
# exit()
Tx = 1.0e6*rmasse*mass*(vtx)**2/clight**2 # eV
Ty = 1.0e6*rmasse*mass*(vty)**2/clight**2
Tz = 1.0e6*rmasse*mass*(vtz)**2/clight**2
print("fitted particle temp (x,y,z) (eV) %.4e, %.4e, %.4e" %(Tx,Ty,Tz))

#print "average directional E per particle (eV)",1.0e6*np.sum(directE)/np.sum(weight);
#print "total-(thermal+directional) (J) =",(np.sum(totalE)-np.sum(thermalE)-np.sum(directE))*erstm*1.6e-13/sratio[sp]


# vt=np.sqrt(saTemp*1.e-6/rmasse/mass) # thermal velocity based on mean scalar temperature
vt=np.sqrt(wTemp*1.0e-6/rmasse/mass) # thermal velocity based on weighted temp
# vt=np.sqrt((Tx+Ty+Tz)/3*1.e-6/rmasse/mass) # thermal velocity based on mean scalar temperature
vt=np.sqrt(Tz*1.e-6/rmasse/mass) # thermal velocity based on mean scalar temperature
#print "mean thermal velovity (c unit) = ", vt

(x,y)=hist1d(vz/clight,weight,nBin=500,minVal=None,maxVal=None,logScale=False)

#vtx=vt
fm=getMaxwellian(x,vt,muz/clight)

norm=np.sum(y)
norm1=np.sum(fm);
y1 = fm*norm/norm1

f1, ax1 = plt.subplots()
ttl =  "f(vz) species %d at t=%.2f ns" %(spc+1,tt)

ax1.plot(x,y,linewidth=2,label="actual")
ax1.plot(x,y1,linewidth=2,label="fit")
# ax1.set(title=ttl, xlabel='velocity (c)', ylabel='',xscale='log')
ax1.set(title=ttl, xlabel='velocity (c)', ylabel='')
ax1.text(0.95*ax1.get_xlim()[0],1.1*ax1.get_ylim()[0],fname)
# ax1.set_ylim([1e12,1.05*np.max(y)])
# ax1.set_xlim([-0.005,0.005])
f1.savefig(dirname + ttl.replace(" ", "_") + '.png',dpi=4*f1.dpi)
plt.show()

exit()
# plot angular distribution
ang=theta_ang(Part[spc][4],Part[spc][5],Part[spc][6]);

(ang_bin,ang_h)=hist1d(ang,weight,nBin=60)

ang_h = map(lambda x, y: y/np.sin(x), ang_bin, ang_h)
ang_bin = [ang_bin/np.pi*180 for ang_bin in ang_bin]
f, ax = plt.subplots()
ax.plot(ang_bin,ang_h,linewidth=2)
ax.set(title='f(ang)', xlabel='Ang (deg)', ylabel='#/Deg',yscale='log')
plt.show()

