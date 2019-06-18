#!/usr/bin/env python
"""Script to write input files from sclr*.p4 and flds"""
__author__ = "Ihor Holod"
__email__ = "holod1@llnl.gov"
__version__ = "081117"

import argparse
import os.path
import numpy as np
from read_xdr import readXDRsclr, readXDRflds

def plot2d(x,y,dat):
    import matplotlib.pyplot as plt
    import pylab as pl
    fig, ax = pl.subplots(figsize=(8,8))
    im = ax.pcolor(x,y,dat)
    plt.colorbar(im)
    ax.set_aspect('equal');
    plt.show()

dirname=os.getcwd() + '/';
class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description=__doc__);
parser.add_argument('-i', type=int, help='time index');
parser.parse_args(namespace=arg)

num = int(arg.i) if arg.i != None else 1;

# read scalar
fname=dirname+'sclr'+str(num)+'.p4'
print(fname);
(X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)

nx=len(X);
nz=len(Z);

for probe in range(len(VarNames)):
    fid = open(dirname + VarNames[probe].replace(" ", "") + ".in", "w")
    fid.write("#\t"+VarNames[probe]+"\t"+VarUnits[probe]+"\t"+"%.4f ns" %(time) +"\n" );
    dat=np.squeeze(Var[probe])
    if probe==VarNames.index("RhoT2"):
        dat*=1+(0.5-np.random.rand(dat.shape[0],dat.shape[1]))/10
    fid.write("%d %d\n" %(nx,nz));
    for i in range(nx):
        fid.write("%.4f " %(X[i]));
        fid.write("\n");
    for i in range(nz):
        fid.write("%.4f " %(Z[i]));
        fid.write("\n");
    for j in range(nz):
        for i in range(nx):
            fid.write("%.4e " %(dat[i,j]));
        fid.write("\n");
    fid.close()

# conductivity
fid = open(dirname + "cond.in", "w")
probe=VarNames.index("RhoT1")
dat=np.squeeze(Var[probe])
ind = np.nonzero(dat<1e18)
dat[:] = 1.0
dat[ind] = 0.0

fid.write("%d %d\n" %(nx,nz));
for i in range(nx):
    fid.write("%.4f " %(X[i]));
    fid.write("\n");
for i in range(nz):
    fid.write("%.4f " %(Z[i]));
    fid.write("\n");
for j in range(nz):
    for i in range(nx):
        fid.write("%.4e " %(dat[i,j]));
    fid.write("\n");
fid.close()

plot2d(Z,X,dat)


# read fields
fname=dirname+'flds'+str(num)+'.p4'
print(fname);
(X,Y,Z,Var,VarNames,VarUnits,time)=readXDRflds(fname)

nx=len(X);
nz=len(Z);

coord = {0:'x',1:'y',2:'z'}

for comp in range(3):
    for probe in range(len(VarNames)):
        fid = open(dirname + VarNames[probe].replace(" ", "") + "_" + coord[comp] + ".in", "w")
        fid.write("#\t"+VarNames[probe]+coord[comp]+"\t"+VarUnits[probe]+"\t"+"%.4f ns" %(time) +"\n" );
        dat=np.squeeze(Var[probe][comp])
        fid.write("%d %d\n" %(nx,nz));
        for i in range(nx):
            fid.write("%.4f " %(X[i]));
            fid.write("\n");
        for i in range(nz):
            fid.write("%.4f " %(Z[i]));
            fid.write("\n");
        for j in range(nz):
            for i in range(nx):
                fid.write("%.4e " %(dat[i,j]));
            fid.write("\n");
        fid.close()



