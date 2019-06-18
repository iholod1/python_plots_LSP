#!/usr/bin/env python
# script to plot from pext*.p4
# massi and chargei must be adjusted for ion species
# I. Holod, 04/24/18
from mpi4py import MPI

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());
root = 0


import argparse
import os.path

import numpy as np

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from read_xdr import readXDRstruct, readXDRpextMPI
from energy import energy
from read_deck import readDeck
from tools.fitting import linReg


dirname=os.getcwd() + '/';

class C(object):
    pass

def fsave(x,y,fttl,**kwargs):
    fout = open(fttl + ".out", 'w')
    if "comment" in kwargs:
        fout.write("#" + kwargs["comment"] + "\n");
    for i in range(len(x)):
        fout.write("%.8e %.8e" %(x[i], y[i]));
        fout.write("\n");
    fout.close()

def plot2d(R,Z,dat,**kwargs):
    f, ax = pl.subplots(figsize=(8,8))
    cm = matplotlib.cm.get_cmap(cmap)
    if np.sum(dat.flatten())<1e-6: # no data to plot
        return()
    if linN: 
        im = ax.pcolor(Z,R,dat,vmin=cmin,vmax=cmax,cmap=cm)
    else:
        dat[np.nonzero(dat<1)]=1
        im = ax.pcolor(Z,R,dat,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)

    ax.set_xlim([Z[0],Z[-1]])
    ax.set_ylim([R[0],R[-1]])
    if ("fname" in kwargs):
        title = kwargs["title"]
    else:
        title = r"Birth location (num/cm$^2$)"
    ax.set(title=title,xlabel="Z (cm)",ylabel="R (cm)")
    if os.path.isfile(os.path.join(dirname,'struct.p4')):
        (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
        lines = [[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
        lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
        ax.add_collection(lc)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        CB = plt.colorbar(im, cax=cax) 
    if aspect: 
        ax.set_aspect('equal');
    if ("fname" in kwargs):
        fttl = kwargs["fname"]
    else:
        fttl = os.path.join(dirname,"plots","pdf_extract{:d}_RZlocation".format(iExtr))
    f.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    pl.close(f)
    return()

arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-p', type=int, help='probe');
parser.add_argument('-s', type=int, help='species');
parser.add_argument('-mint', type=float, help='start time');
parser.add_argument('-maxt', type=float, help='end time');
parser.add_argument('-nt', type=int, help='number of time bins');
parser.add_argument('-mine', type=float, help='minimum energy');
parser.add_argument('-maxe', type=float, help='maximum energy');
parser.add_argument('-minn', type=float, help='minimum number');
parser.add_argument('-maxn', type=float, help='maximum number');
parser.add_argument('-minr', type=float, help='minimum R');
parser.add_argument('-maxr', type=float, help='maximum R');
parser.add_argument('-maxz', type=float, help='maximum Z');
parser.add_argument('-minz', type=float, help='minimum Z');
parser.add_argument('-linn', action='store_true', help="linear scale for yield");
parser.add_argument('-line', action='store_true', help="linear scale for energy");
parser.add_argument('-solid', action='store_true', help="distribution in solid angle")
parser.add_argument('-cumr', action='store_true', help="cumulative distribution in r")
parser.add_argument('-volume', action='store_true', help="1/r factor in radial distribution")
parser.add_argument('-kev', action='store_true', help="keV units")
parser.add_argument('-nsigma', action='store_true', help="calculate nSigma");
parser.add_argument('-temp', action='store_true', help="fit temperature");
parser.add_argument('-save', action='store_true', help="save only");
parser.add_argument('-cmap', type=str, help='colormap');
parser.add_argument('-cmin', type=float, help='min value');
parser.add_argument('-cmax', type=float, help='max value')
parser.add_argument('-ntof', type=float, help='ntof distance')
parser.add_argument('-a', action='store_true', help="keep aspect ratio")


parser.parse_args(namespace=arg)
iExtr = int(arg.p) if arg.p != None else 1 
spc = int(arg.s)-1 if arg.s != None else 9 # default:neutrons
minT = float(arg.mint) if arg.mint != None else None
maxT = float(arg.maxt) if arg.maxt != None else None
nT = arg.nt  if arg.nt != None else 100 
minE = float(arg.mine) if arg.mine != None else None
maxE = float(arg.maxe) if arg.maxe != None else None
minN = float(arg.minn) if arg.minn != None else None
maxN = float(arg.maxn) if arg.maxn != None else None
setMinR = float(arg.minr) if arg.minr != None else None
setMaxR = float(arg.maxr) if arg.maxr != None else None
setMaxZ = float(arg.maxz) if arg.maxz != None else None
setMinZ = float(arg.minz) if arg.minz != None else None
noShow = arg.save
nsigma = arg.nsigma
temp = arg.temp
linN = arg.linn
linE = arg.line
cumr = arg.cumr
volume = arg.volume
solid = arg.solid
kev = arg.kev
aspect= arg.a

cmap = arg.cmap  if arg.cmap != None else None 
cmin = float(arg.cmin) if arg.cmin != None else None 
cmax = float(arg.cmax) if arg.cmax != None else None 
ntof = float(arg.ntof) if arg.ntof != None else 0.0

if noShow: matplotlib.use('Agg') # to run in pdebug (not using $DISPLAY)
import matplotlib.pyplot as plt
if 'classic' in plt.style.available: plt.style.use('classic')
from matplotlib.colors import LogNorm
import matplotlib.collections as mc
import pylab as pl

if nsigma:
    from tools.csdata import getDD, getHeBe, getDCD2

fname=dirname+'pext'+str(iExtr)+'.p4'

# particle mass in positron units
mass=1.825e+03; # neutron
charge=1.0;

# read from the local deck
str1="species"+str(spc+1)
mass=readDeck(dirname,str1,"mass");
charge=readDeck(dirname,str1,"charge");
if abs(charge) < 0.5: charge = 1.0;

if myRank ==0:
    print("mass = %f" %(mass))
    print("charge = %f" %(charge))

charge=charge*1.609e-13; # converts charge to mcoulombs
#############################


     
# world_group = MPI.Comm.Get_group(comm)
# world_group = comm.Get_group()
# if myRank == 0: print(world_group)
# new_group = world_group.Excl([1])
# if myRank == 0: print(new_group.Get_size())
# 
# comm1 = MPI.Comm.Create_group(new_group)
# myRank1=comm1.Get_rank()
# nProc1=int(comm1.Get_size());
# 
# print(myRank,myRank1)    
# print(nProc,nProc1)
# comm1.Free()
# 
# exit()
##########################
# if myRank == root:
#     nPart = 0
# else:
#     nPart = 1
# 
# if nPart == 0:
#     color = 22
#     key = -1
# else:
#     color = 11
#     key = myRank
#     
# comm1 = comm.Split(color,key)
# myRank1=comm1.Get_rank()
# nProc1=int(comm1.Get_size());
# 
# print(myRank,myRank1)    
# print(nProc,nProc1)
# comm1.Free()
#    
# exit()



Var = readXDRpextMPI(fname,myRank=myRank,nProc=nProc)
nQuantities = int(Var.shape[1])

comm.Barrier()

minT = np.max([minT,min(Var[:,0])]) if minT != None else min(Var[:,0])
minT = comm.allreduce(minT,op=MPI.MIN)

maxT = np.min([maxT,max(Var[:,0])]) if maxT != None else max(Var[:,0])
maxT = comm.allreduce(maxT,op=MPI.MAX)

setMinR = setMinR if setMinR !=None else np.min(Var[:,2])
setMinR = comm.allreduce(setMinR,op=MPI.MIN)

setMaxR = setMaxR if setMaxR !=None else np.max(Var[:,2])
setMaxR = comm.allreduce(setMaxR,op=MPI.MAX)

setMinZ = setMinZ if setMinZ !=None else np.min(Var[:,4])
setMinZ = comm.allreduce(setMinZ,op=MPI.MIN)

setMaxZ = setMaxZ if setMaxZ !=None else np.max(Var[:,4])
setMaxZ = comm.allreduce(setMaxZ,op=MPI.MAX)

indx=np.argwhere((Var[:,0]>=minT)&(Var[:,0]<=maxT)
                 &(Var[:,2]>=setMinR)&(Var[:,2]<=setMaxR)
                 &(Var[:,4]>=setMinZ)&(Var[:,4]<=setMaxZ))
# apply selection criteria
Var=Var[indx[:,0],:]
nPart=Var.shape[0]

#redistribute new data array
balance = np.zeros((1,nProc))
balance[0,myRank] = nPart
balance = comm.allreduce(balance,op=MPI.SUM)

giveto = np.zeros((nProc,nProc))
[w,r]=divmod(np.sum(balance.flatten()),nProc)
mnN = w
mxN = w + int(r>0)

for i in range(nProc):
    if balance[0,i]>=mxN:
        continue
    for j in range(nProc):
        if balance[0,j]>mnN:
            n = min(balance[0,j]-mxN, mxN - balance[0,i])
            giveto[j,i] = n
            balance[0,j] -= n
            balance[0,i] += n
        if balance[0,i]>=mxN:
            break

comm.barrier()
# check if anyone to give to
if sum(giveto[myRank,:])>0:
    for i in range(nProc):
        if giveto[myRank,i]==0:
            continue
        else:
            send_size = int(giveto[myRank,i])
            send_data = Var[-send_size:,:]
            Var = Var[:-send_size,:]
            comm.Send(send_data,dest=i)
# check if anyone to take from
if sum(giveto[:,myRank])>0:
    for i in range(nProc):
        if giveto[i,myRank]==0:
            continue
        else:
            recv_size = int(giveto[i,myRank])
            recv_data = np.zeros((recv_size,nQuantities),dtype=np.float64)
            comm.Recv(recv_data,source=i)
            Var = np.concatenate((Var,recv_data),axis=0)

nPart=Var.shape[0]
nPartTot = comm.reduce(nPart,root=root,op=MPI.SUM)
if myRank==root: 
    print("minT=%.2f, maxT=%.2f " %(minT,maxT))
    print("Total number of markers %d" %(nPartTot))

if nPart==0: 
    print("%d: no particles" %(myRank))
    exit()
    
# particle times
partT=Var[:,0].flatten(); 
# particle radial and axial positions
partR=Var[:,2].flatten();
partZ=Var[:,4].flatten();

# particle weight
weight = Var[:,1]/charge
# print("sum of weights %.2e" %(np.sum(weight)))
# kinetic energy

dum = energy(Var[:,5],Var[:,6],Var[:,7],mass,"ke","ang","vz");
ke = dum["ke"]
ang = dum["ang"]
vz = dum["vz"]

meanvz = np.mean(vz)
dum = comm.allreduce(meanvz,op=MPI.SUM)
meanvz = dum/nProc    
if myRank ==0:
    print("average vz %.4e cm/s" %(meanvz)) 

if ntof>0:
    indt = np.nonzero(vz<=0.0)
    weight[indt] = 0.0
    indt = np.nonzero(vz>0.0)
    partT[indt] = partT[indt] + 1.e9*ntof/vz[indt]
    
    meanvz = np.mean(vz[indt])
    dum = comm.allreduce(meanvz,op=MPI.SUM)
    meanvz = dum/nProc 
    minT += 0.25*1.e9*ntof/meanvz
    maxT += 2*1.e9*ntof/meanvz     

#     if nsigma:
# # load DD Cross Section data
# #  (csE,csS)=getDD();
# #  nSigmaE = 0.0
# #  for ip in range(nPart):
# #    nSigmaE += weight[ip]*np.interp(ke[ip],csE.flatten(),csS.flatten());
# #  print "nSigma =", nSigmaE
# 
# # load AmBe thick target probability
# #  (pE,pP)=getHeBe();
# 
# # load DCD2 thick target probability
#         (pE,pP)=getDCD2();
#         rTarget=0.1;
#         nNeutrons = np.zeros((nPart))
#         for ip in range(nPart):
#             if partR[ip]<rTarget:
#                 nNeutrons[ip] = weight[ip]*np.interp(ke[ip],pE.flatten(),pP.flatten());
#         
#         print("nNeutrons = %.2e" %(np.sum(nNeutrons)))
#         nB = 40
#         (R,YR)=hist1dlin(partR,nNeutrons,nB,minVal=0,maxVal=1.0)
#         YldDensR=YR[0:-1]/R[0:-1]/np.diff(R)
#         nX=nB*2; nY=nB*2;
#         newX=np.linspace(-rTarget,rTarget,nX);
#         newY=np.linspace(-rTarget,rTarget,nY);
#         YldDens=np.zeros((nX,nY));
#         for i in range(nX):
#             for j in range(nY):
#                 rdum=np.sqrt(newX[i]**2+newY[j]**2);
#                 YldDens[i,j]=np.interp(rdum,R[0:-1],YldDensR)*0.5/np.pi;




tbin=np.linspace(minT,maxT,nT+1,endpoint=True)
delt = 0.5*(tbin - np.roll(tbin,1))[1:].reshape((nT,1))
tt = 0.5*(tbin + np.roll(tbin,1))[1:].reshape((nT,1))
indexT=np.searchsorted(tbin[1:],partT);



if ntof>0:
    indt = np.nonzero(indexT>=nT)
    indexT[indt]=nT-1
    weight[indt]=0.0

numberT=np.zeros((nT,1));
numberE=np.zeros((nT,1));

nRBin = 100
delR = (setMaxR-setMinR)/float(nRBin)
Rbin = np.linspace(setMinR,setMaxR,nRBin+1)[:-1] + delR/2
# Rbin = Rbin.reshape((len(Rbin),1))

nZBin = 100
delZ = (setMaxZ-setMinZ)/float(nZBin)
Zbin = np.linspace(setMinZ,setMaxZ,nZBin+1)[:-1] + delZ/2
# Zbin = Zbin.reshape((len(Zbin),1))

numberRZ=np.zeros((nRBin,nZBin));
numberRZT=np.zeros((nRBin,nZBin,nT));
numberR=np.zeros((nRBin));


delTheta = 10*np.pi/180.
nThetaBin = int(np.pi/delTheta)
ThetaBin = np.linspace(0,np.pi-delTheta,nThetaBin)+0.5*delTheta
if solid:
    dOmega = np.array([4.*np.pi*(np.sin(0.5*(ThetaBin[i]+0.5*delTheta))**2-
                             np.sin(0.5*(ThetaBin[i]-0.5*delTheta))**2) for i in range(len(ThetaBin))]) 
else:
    dOmega = np.array([2.*delTheta for i in range(len(ThetaBin))]) 


numberTheta=np.zeros((nThetaBin,1));

maxE = maxE if maxE!=None else np.max(ke)
maxE = comm.allreduce(maxE,op=MPI.MAX)
minE = minE if minE!=None else np.min(ke)
minE = comm.allreduce(minE,op=MPI.MIN)

if linE:
    nEBin = 100
    Ebin = np.linspace(minE,maxE,nEBin+1)
else:    
    nEBin = 40
    Ebin = 10**(np.linspace(np.log10(minE),np.log10(maxE),nEBin+1))

delE = 0.5*(Ebin - np.roll(Ebin,1))[1:]
EbinC = 0.5*(Ebin + np.roll(Ebin,1))[1:]    

fE=np.zeros((nEBin,1));
fEE=np.zeros((nEBin,1)); # energy contributions binned
fET=np.zeros((nEBin,nT));


# print(Ebin)
# if True:
#     Ebin = 10**(np.linspace(np.log10(minE),np.log10(maxE),nEBin+1))
# else:
#     Ebin = np.linspace(minE,maxE,nEBin+1)

# delR = setMaxR/float(nRBin)
# Rbin = np.linspace(0,setMaxR-delR,nRBin)+delR/2


for i in xrange(nPart):
    if ke[i]<maxE and ke[i]>minE:
        iE = np.min([np.max([0,np.digitize(ke[i], Ebin)-1]),nEBin-1])
        fE[iE] += weight[i]/delE[iE] 
        fEE[iE] += weight[i]*ke[i]
        fET[iE,indexT[i]] += weight[i]/delE[iE]

        numberT[indexT[i]] += weight[i];
        numberE[indexT[i]] += weight[i]*ke[i];
        iTheta=int(divmod(ang[i],delTheta)[0])
        numberTheta[iTheta] += weight[i]/dOmega[iTheta]
    
        iR = min(nRBin-1,int(divmod(partR[i]-setMinR,delR)[0]))
        iZ = min(nZBin-1,int(divmod(partZ[i]-setMinZ,delZ)[0]))
        numberRZ[iR,iZ] += weight[i]/(2.*np.pi*Rbin[iR]*delR*delZ)
        numberR[iR] += weight[i]/delR
        numberRZT[iR,iZ,indexT[i]] += weight[i]/(2.*np.pi*Rbin[iR]*delR*delZ)

dum=np.zeros((nT,1));
comm.Reduce(numberT, dum, op=MPI.SUM, root=root)
if myRank == root:
    numberT=dum/delt

dum=np.zeros((nT,1));
comm.Reduce(numberE, dum, op=MPI.SUM, root=root)
if myRank == root: 
    numberE=dum
    
dum=np.zeros((nRBin,nZBin));
comm.Reduce(numberRZ, dum, op=MPI.SUM, root=root)
if myRank == root: 
    numberRZ=dum

dum=np.zeros((nRBin,nZBin,nT));
comm.Reduce(numberRZT, dum, op=MPI.SUM, root=root)
if myRank == root:
    numberRZT=dum/delt

dum=np.zeros((nRBin));
comm.Reduce(numberR, dum, op=MPI.SUM, root=root)
if myRank == root: 
    numberR=dum    
    
dum=np.zeros((nThetaBin,1));
comm.Reduce(numberTheta, dum, op=MPI.SUM, root=root)
if myRank == root: 
    numberTheta=dum
    
dum=np.zeros((nEBin,1));
comm.Reduce(fE, dum, op=MPI.SUM, root=root)
if myRank == root: 
    fE=dum      
    
dum=np.zeros((nEBin,1));
comm.Reduce(fEE, dum, op=MPI.SUM, root=root)
if myRank == root: 
    fEE=dum  

dum=np.zeros((nEBin,nT));
comm.Reduce(fET, dum, op=MPI.SUM, root=root)
if myRank == root: 
    fET=dum   

if myRank==root:
    if not os.path.exists(dirname+"plots"):  os.makedirs(dirname+"plots")
    
    yld=np.cumsum(numberT*delt);
    minN = minN if minN!=None else min(yld)
    maxN = maxN if maxN!=None else max(yld)
        
    beamE=np.cumsum(numberE);

#    Time history
    f, ax = plt.subplots()
    
    x = tt
    y = yld
    
    ax.plot(x,y,linewidth= 2)
    yscale = 'linear' if linN else 'log'
    ax.set(title="Time evolution", xlabel='time (ns)', ylabel='num',yscale=yscale)
    ylims = [minN, 1.05*maxN] 
    ax.set_ylim(ylims)
#     ax.text(ax.get_xlim()[0],1.1*ax.get_ylim()[0],fname)
    fttl = dirname + "plots/" + 'pdf_extract'+str(iExtr)+'_history'
    f.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    fsave(x,y,fttl,comment="\ttime (ns)" + "\tYield (num)")
    
    f0, ax0 = plt.subplots()
    x = tt
    y = numberT
    ax0.plot(x,y,linewidth= 2)
    yscale = 'linear' if linN else 'log'
    ax0.set(title="Time evolution", xlabel='time (ns)', ylabel='num/ns',yscale=yscale)
    f0.savefig(fttl + "_diff.png", dpi=200,  bbox_inches="tight")
    fsave(x,y,fttl + "_diff",comment="\ttime (ns)" + "\tYield (num/ns)")


        
#    Radial distribution
    f1, ax1 = plt.subplots()
    x = Rbin
    
    if cumr:
#         y = np.cumsum(numberR.flatten()*2.*np.pi*Rbin.flatten()*delR)/yld[-1]
        y = np.cumsum(numberR)/np.sum(numberR)
        ylabel = 'fraction of total'
    else:
        if volume: 
            y = np.sum(numberRZ,axis=1)*delZ; ylabel = 'num/cm^2'
        else:
            y = numberR; ylabel = 'num/cm' 

    ax1.plot(x,y,linewidth= 2)    
    ax1.set(title="Radial distribution", xlabel='R (cm)', ylabel=ylabel)
    ax1.set_ylim([0,1.05*np.max(y)])
    ax1.set_xlim([setMinR,setMaxR])
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_Rlocation'
    f1.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    fsave(x,y,fttl,comment="\tR (cm)" + "\tYield (fraction)" + "\tZmax = " + str(setMaxZ))

#    axial distribution
    f11, ax11 = plt.subplots()
    x = Zbin
#         y = np.cumsum(numberR.flatten()*2.*np.pi*Rbin.flatten()*delR)/yld[-1]
    y = numberRZ.transpose().dot(Rbin)*2.*np.pi*delR; ylabel = 'num/cm'
    y = np.cumsum(y)/np.sum(y)
    # print(np.nonzero(y>0.9)[0][0])
    print("Axial extend {:.3f} cm".format(x[np.nonzero(y>0.9)[0][0]]-x[np.nonzero(y>0.1)[0][0]]))
    ax11.plot(x,y,linewidth= 2)
    ax11.set(title="Axial distribution", xlabel='Z (cm)', ylabel=ylabel)
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_Zlocation'
    f11.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    
    
    if False:
        j = np.nonzero(Zbin>=0)[0][0]
        x = Zbin[j:]    
        dum = np.zeros(nT)
        for i in range(nT):
    #
            fttl = os.path.join(dirname,"plots","pdf_extract{:d}_T{:06.1f}_RZlocation".format(iExtr,tt[i]))
            title = r"Birth location (num/cm$^2$) T{:06.1f}".format(tt[i])
            plot2d(Rbin,Zbin,numberRZT[:,:,i],fname=fttl,title=title)
    #         
            y = numberRZT[:,j:,i].transpose().dot(Rbin)*2.*np.pi*delR; ylabel = 'num/cm'
            ysum=np.sum(y)
            if (ysum<1): continue
            y = np.cumsum(y)/ysum
            dum[i]=x[np.nonzero(y>0.9)[0][0]]
        fttl = os.path.join(dirname,"plots","pdf_extract{:d}_RZlocation".format(iExtr))
        fsave(tt,dum,fttl,comment="\ttime (ns)" + "\tdZ (cm)" + "\tRmax = " + str(setMaxR))

    
#    Angular distributions
    f2, ax2 = plt.subplots()
    x = ThetaBin/np.pi*180
    y = numberTheta
    if solid: 
        ax2.set_ylabel(r'dN/d$\Omega$') # num per solid angle
    else:
        ax2.set_ylabel(r'dN/d$\theta$') # num per angle    
    ax2.plot(x,y,marker='s',linewidth= 2)
    ax2.set(title="Angular distribution")
    ax2.set_xlabel(r'$\theta$ (deg)')
    ax2.set_ylim([0,ax2.get_ylim()[1]])
    ax2.set_xlim([0,10])
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_ThetaDistr'
    f2.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    fsave(ThetaBin,numberTheta,fttl,comment="\tTheta (rad)" + "\tYield (num)")
  
    f3 = plt.figure()
    y = np.concatenate((np.flipud(numberTheta),numberTheta),axis=0);
    x = np.concatenate([-ThetaBin[::-1],ThetaBin])
    ax3 = plt.subplot(111,projection='polar')
    ax3.plot(x, y, marker='o', linewidth= 2, linestyle='None')
#     ax3.set_rmax(2)
#     ax3.set_rticks([1e6])  # less radial ticks
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
#     ax3.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax3.grid(True)
    if solid: 
        ax3.set_title(r'dN/d$\Omega$') # num per solid angle
    else:
        ax3.set_title(r'dN/d$\theta$') # num per angle        
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_ThetaDistr_polar'
    f3.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")    
    

    
#    Energy distribution
    f4, ax4 = plt.subplots()
    x = EbinC
    y = fE
    xlabel = 'E (MeV)'
    ylabel='f(E) (num/MeV)'
        
    if kev:
        x*=1.e3
        y*=1.e-3
        xlabel = 'E (keV)'
        ylabel='f(E) (num/keV)'
    
    xscale = 'linear' if linE else 'log'
    yscale = 'linear' if linN else 'log'
    
    if not linN: y[np.nonzero(y<1)]=1
    
    
    ax4.plot(x,y,linewidth= 2,label="Pext")
    ax4.set_ylim([minN, maxN*(maxE-minE)])
#     if kev:
#         ax4.set_xlim([1.e3*minE, 1.e3*maxE])
#     else:
#         ax4.set_xlim([minE, maxE])
    
    
    
#     x = x[np.nonzero((x>ax4.get_xlim()[0])&(x<ax4.get_xlim()[-1]))]

    ax4.set(title="Energy distribution", xlabel=xlabel, ylabel=ylabel,yscale=yscale,xscale=xscale)
    
    if temp:
        regC = linReg(x,np.log10(y))
        print("Beam temperature %.2f MeV" %(-1/regC[0][0]))
        label = "Fit T=%.2fMeV" %(-1/regC[0][0])
        ax4.plot(x,10**np.polyval(regC,x),label=label)
        ax4.legend(loc='best', shadow=True)
    
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_EnDistr'
    f4.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    fsave(EbinC,fE,fttl,comment="\tEnergy (MeV)" + "\t num/MeV")
    
    if nsigma:
# load DD Cross Section data
        (csE,csS)=getDD();
        nSigmaE = np.zeros(len(x))
        for i in range(len(x)):
            nSigmaE[i] = y[i]*np.interp(x[i],csE.flatten(),csS.flatten())
#     for ip in range(npart):
#         nSigmaE += weight[ip]*np.interp(ke[ip],csE.flatten(),csS.flatten());
        print("nSigmaE = %.2e" %(np.trapz(nSigmaE,x)))
#### temporal dependence
        dum = np.zeros(nT)
        for j in range(nT):            
            for i in range(len(EbinC)):
                nSigmaE[i] = fET[i,j]*np.interp(EbinC[i],csE.flatten(),csS.flatten())
            dum[j] = np.trapz(nSigmaE,EbinC)
        fttl = os.path.join(dirname,"plots","pdf_extract{:d}_nSigma".format(iExtr))
        fsave(tt,dum,fttl,comment="\ttime (ns)" + "\tnSigma" + "\tRmax = " + str(setMaxR) + "\tZmax = " + str(setMaxZ))     
    
#     2D birth location plot
    f5, ax5 = pl.subplots(figsize=(8,8))
    cm = matplotlib.cm.get_cmap(cmap)
    if linN: 
        im = ax5.pcolor(Zbin,Rbin,numberRZ,vmin=cmin,vmax=cmax,cmap=cm)
    else:
        numberRZ[np.nonzero(numberRZ<1)]=1
        im = ax5.pcolor(Zbin,Rbin,numberRZ,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)

    ax5.set_xlim([Zbin[0],Zbin[-1]])
    ax5.set_ylim([Rbin[0],Rbin[-1]])
    ax5.set(title=r"Birth location (num/cm$^2$)",xlabel="Z (cm)",ylabel="R (cm)")
    if os.path.isfile(dirname+'struct.p4'):
        (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
        lines = [[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
        lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
        ax5.add_collection(lc)
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        CB = plt.colorbar(im, cax=cax) 
    if aspect: 
        ax5.set_aspect('equal');   
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_RZlocation'
    f5.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")
    

#    Energy distribution binned
    f4, ax4 = plt.subplots()
    x = EbinC
    y = fEE/beamE[-1]
    xlabel = 'E (MeV)'
    ylabel='fE'
        
    xscale = 'linear' if linE else 'log'
    ax4.plot(x,y,"ob",linewidth= 2,label="Pext")
    ax4.set(title="Energy distribution", xlabel=xlabel, ylabel=ylabel,xscale=xscale)
    fttl = dirname  + "plots/" + 'pdf_extract'+str(iExtr)+'_EnDistr2'
    f4.savefig(fttl + ".png", dpi=200,  bbox_inches="tight")


    print("Beam num. = %.2e" % max(yld));
    print("Beam energy (J) = %.2e" %(beamE[-1]*1.602e-13));
    
    if not noShow: plt.show()
    
exit()



#     if spc==9:
#         rnds=random.sample(range(0,len(Var[2])-1),10000)
#         partR=Var[2,rnds]
#         partZ=Var[4,rnds]
#         partW=Var[1,rnds]/charge
#         mxw=max(partW)/10
#         dotSize=partW[:]/mxw
#         pitch=np.array([weight[j]*Var[7,j]/np.sqrt(Var[5,j]**2+Var[6,j]**2+Var[7,j]**2) for j in rnds]);        pitch=np.array([weight[j]*Var[7][j]/np.sqrt(Var[5][j]**2+Var[6][j]**2+Var[7][j]**2) for j in rnds]);
#         print("average pitch %.4f" %(np.mean(pitch)/np.mean(weight)))
#     
#         f2,ax2=plt.subplots();
#         ax2.scatter(partZ,partR,c='black',s=dotSize);
#     
#     # add structure
#         (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
#         lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
#         lc = mc.LineCollection(lines, linewidths=2)
#         ax2.add_collection(lc);
#         ax2.set_aspect('auto');
#         ax2.set_xlim([min(za),max(zb)]); ax2.set_ylim([min(xa),max(xb)]);
#         ax2.set(title="Birth locations")
#         plt.show()
#         f2.savefig(dirname + 'n_birth.png',dpi=200)
# 
#     else:
#         (x,y)=hist1dlog(ke,weight,80,minVal=0.015) # log binning
#         f3, ax3 = plt.subplots()
#         ax3.plot(x,y,linewidth=2)
#         ttl =  "f(E) species "+str(spc+1)+" maxt = %.1f ns" %(maxT)
#         ax3.set(title=ttl, xlabel='E (MeV)', ylabel='Num/MeV',yscale='log',xscale='log')
# 
#         xlims = [min(x), max(x)] 
#         if minE: xlims[0] = min([minE, xlims[1]])
#         if maxE: xlims[1] = max([maxE, xlims[0]])
#         ax3.set_xlim(xlims);
# 
#         ylims = [min(y), max(y)] 
#         if y0: ylims[0] = min([y0, ylims[1]])
#         if y1: ylims[1] = max([y1, ylims[0]])
#         ax3.set_ylim(ylims);    
#     
#         ax3.set(title=ttl, xlabel='E (MeV)', ylabel='num/MeV')
#     
# # distribution of useful beam
#         if nsigma:
#             f5, ax5 = pl.subplots(figsize=(8,8))
#             im5 = ax5.pcolor(newX,newY,YldDens)
#             ax5.set_xlim([min(newX),max(newY)])
#             ax5.set_ylim([min(newY),max(newY)])
#             ax5.set(xlabel='X (cm)', ylabel='Y (cm)',title="num. of n per cm^2")
#             ax5.set_aspect('equal')
#             divider = make_axes_locatable(ax5)
#             cax = divider.append_axes("right", size="5%", pad=0.05)
#             CB =plt.colorbar(im5, cax=cax)
#         
#         if not noShow: plt.show()
#         if not os.path.exists(dirname+"plots"):  os.makedirs(dirname+"plots")
#         f3.savefig(dirname +"plots/" + "pdf_extract"+str(iExtr)+ "_%.1f" %(maxT) + ".png",dpi=200)
#         if nsigma: f5.savefig(dirname +"plots/" + "yield_density_%.1f" %(maxT) + ".png",dpi=200)
#         fout = open(dirname+'pdf_extract'+str(iExtr)+'.out', 'w')
#         fout.write("#\tEnergy (MeV)"+"\t#/MeV\n");
#         for i in range(len(x)):
#             fout.write("%.8e %.8e" %(x[i], y[i]));
#             fout.write("\n");
#         fout.close()
# 
# print("%d success" %(myRank))
# comm.Barrier()
# exit()
