#!/usr/bin/env python
"""Script to plot from sclr*.p4"""
__author__ = "Ihor Holod"
__credits__ = ["I. Holod", "D. Higginson", "A. Link"]
__email__ = "holod1@llnl.gov"
__version__ = "081117"

import sys

def rhol(**kwargs):
    import argparse
    import os.path
    import numpy as np
    import matplotlib
    from read_xdr import readXDRsclr, readXDRstruct
    import gc
    
    
    if not os.path.exists(os.path.join(os.getcwd(),"plots")):
        os.makedirs(os.path.join(os.getcwd(),"plots"))
    dirname=os.getcwd()+"/"
        
    
    class C(object):
        pass
    arg=C()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', type=int, help='time index')
    parser.add_argument('-si', type=int, help='ion species')
    parser.add_argument('-se', type=int, help='electron species')
    parser.add_argument('-minx', type=float, help='minimum x')
    parser.add_argument('-maxx', type=float, help='maximum x')
    parser.add_argument('-minz', type=float, help='minimum z')
    parser.add_argument('-maxz', type=float, help='maximum z')
    parser.add_argument('-slicex', type=float, action='append', help='z-slice along given x')
    parser.add_argument('-save', action='store_true', help="save only")
    
    if "arglist" in kwargs:
        print(kwargs["arglist"])
        parser.parse_args(kwargs["arglist"],namespace=arg)
    else:
        parser.parse_args(namespace=arg)       
    
    num = int(arg.i) if arg.i != None else 1
    si = int(arg.si) if arg.si != None else 6
    se = int(arg.se) if arg.se != None else 5
    
    x0 = float(arg.minx) if arg.minx != None else None 
    x1 = float(arg.maxx) if arg.maxx != None else None 
    z0 = float(arg.minz) if arg.minz != None else None 
    z1 = float(arg.maxz) if arg.maxz != None else None
    
    sliceX = arg.slicex if arg.slicex != None else None
    silent = arg.save
    
    #######################################################
    
    fname=dirname+'sclr'+str(num)+'.p4'
    print(fname)
    
    if "sdata" in kwargs:
        (X,Y,Z,Var,VarNames,VarUnits,time)=kwargs["sdata"]
    else:
        (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname,silent=silent)    

    tstamp =  "%.2f" % time
    
    ivar = VarNames.index("RhoT" + str(si))
    densi = Var[ivar]
    ivar = VarNames.index("Temp" + str(se))
    tempe = Var[ivar]
    ivar = VarNames.index("RhoT" + str(se))
    dense = Var[ivar]
    
    xmin=x0 if (x0!=None) and (x0>min(X)) and (x0<=max(X)) else min(X)
    xmax=x1 if (x1!=None) and (x1>xmin) and (x1<=max(X)) else max(X)
    zmin=z0 if (z0!=None) and (z0>min(Z)) and (z0<max(Z)) else min(Z)
    zmax=z1 if (z1!=None) and (z1>zmin) and (z1<=max(Z)) else max(Z)
    
    indx=np.argwhere((X>=xmin)&(X<=xmax))
    indx=indx.reshape(indx.shape[0],1,1)
    indz=np.argwhere((Z>=zmin)&(Z<=zmax))
    indz=indz.reshape(1,1,indz.shape[0])
    indy=np.arange(len(Y)).reshape(1,len(Y),1)
    
    #indz=np.transpose(indz)
    X=X[indx].flatten()
    Y=Y[indy].flatten()
    Z=Z[indz].flatten()
    
    densi=np.squeeze(densi[indx,indy,indz])
    dense=np.squeeze(dense[indx,indy,indz])
    tempe=np.squeeze(tempe[indx,indy,indz])
    
    rhol=np.zeros((1,len(Z)));
    tave=np.zeros((1,len(Z)));
    dave=np.zeros((1,len(Z)));
    
    for i in range(len(sliceX)):
        xx = sliceX[i]
        f1 = open(dirname +"plots/rhol_" + '{:.3f}'.format(xx) + ".out", "a")
        f2 = open(dirname +"plots/tave_" + '{:.3f}'.format(xx) + ".out", "a")
        for j in range(len(Z)):
            rhol[0,j]=np.interp(xx,X[:],densi[:,j]);
            tave[0,j]=np.interp(xx,X[:],dense[:,j]*tempe[:,j]);
            dave[0,j]=np.interp(xx,X[:],dense[:,j]);
    
        rL=np.trapz(rhol[0,:],Z)
        tA=np.trapz(tave[0,:],Z)/np.trapz(dave[0,:],Z)
        print("R = %.2f: <RhoL> (1/cm^2) %.2e" %(xx, rL))
        print("R = %.2f: <Temp> (eV) %.2e" %(xx, tA))
        
        f1.write("%.8e %.8e\n" %(time, rL));
        f2.write("%.8e %.8e\n" %(time, tA));
    
        f1.close()
        f2.close()
        
    return()
    
if __name__=="__main__":
    rhol(arglist=sys.argv[1:])
    exit()    



