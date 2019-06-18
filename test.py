#!/usr/bin/env python
# script to plot from sclr*.p4
# I. Holod, 07/05/16


import numpy as np
import xdrlib
import time

# from read_xdr import readXDRsclr
# (X,Y,Z,Var,VarNames,VarUnits,tt)=readXDRsclr('sclr28884.p4')
#    
# print(len(X), len(Z), Var.shape)
# exit()

t1 = time.time()
fileName="sclr28884.p4"
with open(fileName, mode='rb') as file:
    fileContent = file.read()
    u = xdrlib.Unpacker(fileContent) 
        
    ftype = u.unpack_int()
    if(ftype!=3):
        print('Not a Sclr File')

    fversion = u.unpack_int()
    u.unpack_int()
    title = u.unpack_string().decode() # read title
    print(title)

    u.unpack_int();
    frev = u.unpack_string().decode()
    print(frev)

    tStep = u.unpack_float()
    print("Time %.2f" %(tStep))

    geo = u.unpack_int()
    nDomains = u.unpack_int()
    nQuantities = u.unpack_int()
    print("nQuantities = %d" %(nQuantities))

    VarNames = []
    for i in range(nQuantities):
        u.unpack_int()
        VarNames.append(u.unpack_string().decode())

    VarUnits = []
    for i in range(nQuantities):
        u.unpack_int()
        VarUnits.append(u.unpack_string().decode())

    Var=[];X=[];Y=[];Z=[];
    for i in range(nQuantities): Var.append([]);

    for iDom in range(nDomains):
        iR = u.unpack_int()
        jR = u.unpack_int()
        kR = u.unpack_int()
        nI = u.unpack_int()
        XT=u.unpack_farray(nI,u.unpack_float)
        nJ = u.unpack_int()
        YT=u.unpack_farray(nJ,u.unpack_float)
        nK = u.unpack_int()
        ZT=u.unpack_farray(nK,u.unpack_float)

        XL=[];YL=[];ZL=[];
        print(nI,len(XT))
        for k in range(nK):
            for j in range(nJ):
                for i in range(nI):
                    XL.append(XT[i])
                    YL.append(YT[j])
                    ZL.append(ZT[k])
        
        X.extend(XL)
        Y.extend(YL)
        Z.extend(ZL)

        for k in range(nQuantities):
            Var[k].extend(u.unpack_farray(nI*nJ*nK,u.unpack_float))
            
UX, indexX = np.unique(X, return_inverse=True);
UY, indexY = np.unique(Y, return_inverse=True);
UZ, indexZ = np.unique(Z, return_inverse=True);

nX=np.size(UX);nY=np.size(UY);nZ=np.size(UZ);
newVar=np.zeros((nQuantities,nX,nY,nZ))
MaxIndex=np.size(X);

Var=np.array(Var).reshape((nQuantities,MaxIndex));
newVar[:,indexX[:],indexY[:],indexZ[:]]=Var[:,:];
          
t2 = time.time()
print(t2-t1)     
exit()




