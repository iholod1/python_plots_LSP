#!/usr/bin/env python3
# script to plot from pext*.p4
# massi and chargei must be adjusted for ion species
# I. Holod, 07/05/16

import argparse
import os.path

import math
import numpy as np
import random

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from read_xdr import readXDRstruct
from hist1d import hist1dlin, hist1dlog
from energy import kin_en, theta_ang
from read_deck import readDeck


def readXDRpext(fname, setMinT=0, setMaxT=5000, delT=0.1, minQ=0, myRank=0, nProc=1):
    """Reads extraction file 
     Input: file name
     Output: Var[0:nparticles-1][0:nQuantities-1]"""
    import time
    import xdrlib
    
    t1 = time.time()
    
    if (setMaxT - delT - setMinT)<=0: return()
    
    nBins=int((setMaxT-setMinT)/(delT))
    print("nBins=%d" %(nBins))
    binnedT=np.linspace(setMinT,setMaxT-delT,nBins)+delT/2
    print(binnedT[0])
    
    totalQ=np.zeros((nBins,1))
    lostQ=np.zeros((nBins,1))
    wAdj = np.ones((nBins,1))
    
    
    
    print(fname)
    
    with open(fname, mode='rb') as file:
        fileContent = file.read()
        u = xdrlib.Unpacker(fileContent)     
    

        ftype = u.unpack_int()
        fversion = u.unpack_int()

        u.unpack_int()
        title = u.unpack_string() # read title
        print(title)

        u.unpack_int()
        frev = u.unpack_string()
        print(frev)

        geo = u.unpack_int()
        nQuantities = u.unpack_int()

        Units = []
        for i in range(nQuantities):
            ns = u.unpack_int()
            Units.append(u.unpack_string())

        print("nQuantities = %d" %(nQuantities))
        print(Units)
        
        if setMinT>0: # approach based on time criterion
            Var=[]
            for i in range(nQuantities):
                Var.append([])
         
            nQuantities -= 1
            while True:
                try:
                    tt = u.unpack_float()
                    if tt<setMinT or tt>setMaxT:
                        u.set_position(u.get_position()+nQuantities*4)
                    else:
                        dum = u.unpack_farray(nQuantities,u.unpack_float)
                        it=int(divmod(tt-setMinT,delT)[0])
                        totalQ[it]+=dum[0]
                        if dum[0]<minQ:
                            lostQ[it]+=dum[0]
                            continue
                        Var[0].append(binnedT[it])
                        for i in range(nQuantities):
                            Var[i+1].append(dum[i])
                except EOFError:
                    break
            
            Var=np.array(Var)            
            for it in range(nBins):
                if lostQ[it]<totalQ[it]:
                    wAdj[it]=totalQ[it]/(totalQ[it]-lostQ[it])                      
        else: # mpi approach based on apriori knowledge of particle number   
            nPart = divmod(len(u.get_buffer())-u.get_position(),4*nQuantities)[0]
            print("Number of particle records in file = %d" %(nPart))
        
            Var = np.zeros((nQuantities,nPart))
        
            ranges={}
        
            [w,r]=divmod(nPart,nProc)
        
            ranges[0]=(0,w+int(r>0)-1)
        
            for j in range(1,nProc):
                ranges[j]=(ranges[j-1][1]+1,ranges[j-1][1]+w+int(j<r))
        
            print(ranges[myRank])
            u.set_position(u.get_position()+ranges[myRank][0]*nQuantities*4)
      
            for i in range(ranges[myRank][0],ranges[myRank][1]+1):
                dum = u.unpack_farray(nQuantities,u.unpack_float)
                Var[:,i]=dum[:]


        
    file.close()
    t2 = time.time()
    print("Elapsed time %.2f s" %(t2-t1))
    return Var, wAdj 


dirname=os.getcwd() + '/';

class C(object):
    pass

arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-p', type=int, help='probe');
parser.add_argument('-s', type=int, help='species');
parser.add_argument('-mint', type=float, help='start time');
parser.add_argument('-maxt', type=float, help='end time');
parser.add_argument('-mine', type=float, help='minimum energy');
parser.add_argument('-maxe', type=float, help='maximum energy');
parser.add_argument('-minn', type=float, help='minimum number');
parser.add_argument('-maxn', type=float, help='maximum number');
parser.add_argument('-nsigma', action='store_true', help="calculate nSigma");
parser.add_argument('-save', action='store_true', help="save only");

parser.parse_args(namespace=arg)
iExtr = int(arg.p) if arg.p != None else 1; 
spc = int(arg.s)-1 if arg.s != None else 9; # default:neutrons
t0 = float(arg.mint) if arg.mint != None else 0.0;
t1 = float(arg.maxt) if arg.maxt != None else None;
x0 = float(arg.mine) if arg.mine != None else 0.015;
x1 = float(arg.maxe) if arg.maxe != None else None;
y0 = float(arg.minn) if arg.minn != None else None;
y1 = float(arg.maxn) if arg.maxn != None else None;
noShow = arg.save;
nsigma = arg.nsigma;

if noShow: matplotlib.use('Agg') # to run in pdebug (not using $DISPLAY)
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl
if nsigma:
    from tools.csdata import getDD, getHeBe, getDCD2

fname=dirname+'pext'+str(iExtr)+'.p4'

# particle masses in positron units
if spc==2: mass=1.0
if spc==3: mass=3.672e3 # D 
#if spc==3: mass=7.292e3 # He
if spc==9: mass=1.825e+03; # neutron
# particle charges in positron units
if (spc==2): charge=-1.0;
if (spc==9) or (spc==3) :charge=1.0;


# read from the local deck
str1="species"+str(spc+1)
mass=readDeck(dirname,str1,"mass");
print("mass = %f" %(mass))
charge=readDeck(dirname,str1,"charge");
if abs(charge) < 0.5: charge = 1.0;
print("charge = %f" %(charge))

charge=charge*1.609e-13; # converts charge to mcoulombs
#############################

# Var=np.array(readXDRpext(fname))
Var, wAdj =readXDRpext(fname,setMinT=t0,setMaxT=t1,minQ=charge*0.0)

print(Var[0,0])
minT = np.max([t0,min(Var[0,:])]) if t0 != None else min(Var[0,:])
maxT = np.min([t1,max(Var[0,:])]) if t1 != None else max(Var[0,:])
minT = t0
maxT = t1

print("minT=%.2f " %(minT) + "maxT = %.2f" %(maxT));

#ind = np.argsort(Var[0,:]) #sort by time
ind = np.argwhere((Var[0,:]>=minT)&(Var[0,:]<=maxT))
ind = ind.reshape(1,ind.shape[0])[0];
Var = Var[:,ind]; # sort by time

npart=Var.shape[-1];
if npart==0: 
    print("NO DATA TO PLOT")
    exit()
print("npart = %d" %(npart));

# particle times
partT=Var[0,:].flatten(); 
# particle radial position
partR=Var[2,:].flatten();
partZ=Var[4,:].flatten();
# particle weight
weight = Var[1,:]/charge
print("minimum weight %e" %min(weight))
# kinetic energy
ke=kin_en(Var[5,:],Var[6,:],Var[7,:],mass); 

if nsigma:
# load DD Cross Section data
#  (csE,csS)=getDD();
#  nSigmaE = 0.0
#  for ip in range(npart):
#    nSigmaE += weight[ip]*np.interp(ke[ip],csE.flatten(),csS.flatten());
#  print "nSigma =", nSigmaE

# load AmBe thick target probability
#  (pE,pP)=getHeBe();

# load DCD2 thick target probability
    (pE,pP)=getDCD2();
    rTarget=0.1;
    nNeutrons = np.zeros((npart))
    for ip in range(npart):
        if partR[ip]<rTarget:
            nNeutrons[ip] = weight[ip]*np.interp(ke[ip],pE.flatten(),pP.flatten());
    print("nNeutrons = %.2e" %(np.sum(nNeutrons)))
    nB = 40
    (R,YR)=hist1dlin(partR,nNeutrons,nB,minVal=0,maxVal=1.0)
    YldDensR=YR[0:-1]/R[0:-1]/np.diff(R)
    nX=nB*2; nY=nB*2;
    newX=np.linspace(-rTarget,rTarget,nX);
    newY=np.linspace(-rTarget,rTarget,nY);
    YldDens=np.zeros((nX,nY));
    for i in range(nX):
        for j in range(nY):
            rdum=np.sqrt(newX[i]**2+newY[j]**2);
            YldDens[i,j]=np.interp(rdum,R[0:-1],YldDensR)*0.5/np.pi;




# time analysis
# delT=0.1
# nT=int(1+math.floor((maxT-minT)/delT)) 
# print("nT=%d" %(nT))
# tt=np.linspace(minT,maxT,num=nT,endpoint=True)

delT = 0.1
nT=int((t1-t0)/(delT))
tt=np.linspace(t0,t1-delT,nT)+delT/2

numberT=np.zeros((nT,1));
numberE=np.zeros((nT,1));
indexT=np.searchsorted(tt,partT);

setMaxR = 1.
setMaxZ = 4.
nBins = 100
delR = setMaxR/float(nBins)
Rbin = np.linspace(0,setMaxR-delR,nBins)+delR/2
numberR=np.zeros((nBins,1));

for i in range(npart):
    weight[i]*=wAdj[indexT[i]-1]
    numberT[indexT[i]-1]=numberT[indexT[i]-1]+weight[i];
    numberE[indexT[i]-1]=numberE[indexT[i]-1]+weight[i]*ke[i];
    if partR[i]<setMaxR and partZ[i]<setMaxZ:
        iR = int(divmod(partR[i],delR)[0])
        numberR[iR]+=weight[i]

   
yld=np.cumsum(numberT);
beamE=np.cumsum(numberE);

if max(yld)>500:
    f, ax = plt.subplots()
    ax.plot(tt,yld,linewidth= 2)
    ax.set(title="Yield", xlabel='time (ns)', ylabel='#',yscale='log')
    ax.set_ylim([500,1.05*max(yld)])
    ax.text(ax.get_xlim()[0],1.1*ax.get_ylim()[0],fname)
    
    f1, ax1 = plt.subplots()
    ax1.plot(Rbin,np.cumsum(numberR)/yld[-1],linewidth= 2)
    ax1.set(title="Yield", xlabel='R (cm)', ylabel='fraction of total')
    ax1.set_ylim([0,1])
    f1.savefig(dirname + 'Rlocation.png',dpi=200,  bbox_inches="tight")
    fout = open(dirname+'pdf_extract'+str(iExtr)+'_Rlocation.out', 'w')
    fout.write("#\tR (cm)" + "\tYield (num)" + "\tZmax = " + str(setMaxZ) + "\n");
    for i in range(len(Rbin)):
        fout.write("%.8e %.8e" %(Rbin[i], numberR[i]));
        fout.write("\n");
    fout.close()
      

print("max cum. particle num. = %.2e" %(yld[-1]));
print("max cum. particle energy (J) = %.2e" %(beamE[-1]*1.602e-13));

#plt.show()
#exit();

### spatial analysis
#partR=[]; partZ=[]; dotSize=[];
#for i in range(500):
#  j=random.randint(0,len(Var[2])-1);
#  partR.append(Var[2][j]);
#  partZ.append(Var[4][j]);
#  dotSize.append(100*weight[j]/max(weight));
if spc==9:    
    rnds=random.sample(range(0,len(Var[2])-1),10000)
    partR=Var[2,rnds]
    partZ=Var[4,rnds]
    partW=Var[1,rnds]/charge
    mxw=max(partW)/10
    dotSize=partW[:]/mxw
    pitch=np.array([weight[j]*Var[7,j]/np.sqrt(Var[5,j]**2+Var[6,j]**2+Var[7,j]**2) for j in rnds]);
    print("average pitch %.4f" %(np.mean(pitch)/np.mean(weight)))
    
    f2,ax2=plt.subplots();
    ax2.scatter(partZ,partR,c='black',s=dotSize);
    #ax2.scatter(Var[4],Var[2],c='black',s=dotSize);
    # add structure
    (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
    lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
    lc = mc.LineCollection(lines, linewidths=2)
    ax2.add_collection(lc);
    ax2.set_aspect('auto');
    ax2.set_xlim([min(za),max(zb)]); ax2.set_ylim([min(xa),max(xb)]);
    ax2.set(title="Birth locations")
    plt.show()
    f2.savefig(dirname + 'n_birth.png',dpi=200,  bbox_inches="tight")
    exit();

if spc!=9:
    # plot energy distribution
    # dum=np.ones(len(partR))
    # (x,y)=hist1dlin(partR,dum,100)
    # f21,ax21 = plt.subplots()
    # ax21.plot(x,y)
    # ax21.set(xlabel='R (cm)', ylabel='num')

    (x,y)=hist1dlog(ke,weight,80,minVal=x0) # log binning
#  (x,y)=hist1dlog(ke,weight,80) # log binning
    #(x,y)=hist1dlin(ke,weight,80)
    #y[np.argwhere(y<1e-12)[0]] = 1e-12
    f3, ax3 = plt.subplots()
    ax3.plot(x,y,linewidth=2)
    ttl =  "f(E) species "+str(spc+1)+" maxt = %.1f ns" %(maxT)
    ax3.set(title=ttl, xlabel='E (MeV)', ylabel='Num/MeV',yscale='log',xscale='log')

    xlims = [min(x), max(x)] 
    if x0: xlims[0] = min([x0, xlims[1]])
    if x1: xlims[1] = max([x1, xlims[0]])
    ax3.set_xlim(xlims);

    ylims = [min(y), max(y)] 
    if y0: ylims[0] = min([y0, ylims[1]])
    if y1: ylims[1] = max([y1, ylims[0]])
    ax3.set_ylim(ylims);    
    #ax3.set_ylim([1e14,1e17]); ax3.set_xlim([0.025,10]);
    ax3.set(title=ttl, xlabel='E (MeV)', ylabel='num/MeV')
    #ax3.text(0.95*ax.get_xlim()[0],1.1*ax.get_ylim()[0],fname)


# distribution of useful beam
    if nsigma:
        f5, ax5 = pl.subplots(figsize=(8,8))
        im5 = ax5.pcolor(newX,newY,YldDens)
        ax5.set_xlim([min(newX),max(newY)])
        ax5.set_ylim([min(newY),max(newY)])
        ax5.set(xlabel='X (cm)', ylabel='Y (cm)',title="num. of n per cm^2")
        ax5.set_aspect('equal')
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        CB =plt.colorbar(im5, cax=cax)
        
    if not noShow: plt.show()
    if not os.path.exists(dirname+"plots"):  os.makedirs(dirname+"plots")
    f3.savefig(dirname +"plots/" + "pdf_extract"+str(iExtr)+ "_%.1f" %(maxT) + ".png",dpi=200, bbox_inches="tight")
    if nsigma: f5.savefig(dirname +"plots/" + "yield_density_%.1f" %(maxT) + ".png",dpi=200, bbox_inches="tight")
    fout = open(dirname+'pdf_extract'+str(iExtr)+'.out', 'w')
    fout.write("#\tEnergy (MeV)"+"\t#/MeV\n");
    for i in range(len(x)):
        fout.write("%.8e %.8e" %(x[i], y[i]));
        fout.write("\n");
    fout.close()
    exit()









#######################################################
