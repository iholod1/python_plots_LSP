# python script to read *.p4 files
# I. Holod, 05/29/18

import xdrlib
import numpy as np
import time

def readXDRpart(fname):
    """Reads particle dump file 
     Input: file name
     Output: time, Part[0:nSpecies-1][0:nPart-1][0:nQuantities-1]"""
    t1 = time.time()
    f = open(fname,'rb'); # open file
    u = xdrlib.Unpacker(f.read()) 
    f.close()

    ftype = u.unpack_int()
    fversion = u.unpack_int()

    ns = u.unpack_int();title = u.unpack_string() # read title
    print(title)

    ns = u.unpack_int();frev = u.unpack_string()
    print(frev)

    tt = u.unpack_float()
    print("Time =",time)
    geo = u.unpack_int() 
    SymFlag = u.unpack_farray(3,u.unpack_int)
    nSpecies = u.unpack_int()
    nParticles = u.unpack_int()
    nQuantities = u.unpack_int()

    Units = []
    for i in range(nQuantities):
        ns = u.unpack_int()
        Units.append(u.unpack_string())

    print("nSpecies =",nSpecies, "nQuantities =",nQuantities,"nParticles =",nParticles)
    print("Units: ", Units)

    Part = [];
    for s in range(nSpecies):
        Part.append([])
        for i in range(nQuantities):
            Part[s].append([])

    test = 1
    while test:
        try:
            s = u.unpack_int()
            dum = u.unpack_farray(nQuantities,u.unpack_float)
            for i in range(nQuantities):
                Part[s-1][i].append(dum[i])
        except EOFError:
            test = 0
    t2 = time.time()
    print(t2-t1)

    return (tt,Part)

def readXDRpext(fname):
    """Reads extraction file 
     Input: file name
     Output: Var[0:nparticles-1][0:nQuantities-1]"""
    t1 = time.time()
    print(fname)
    f = open(fname,'rb'); # open file
    u = xdrlib.Unpacker(f.read()) 
    f.close()

    ftype = u.unpack_int()
    fversion = u.unpack_int()

    ns = u.unpack_int();title = u.unpack_string() # read title
    print(title)

    ns = u.unpack_int();frev = u.unpack_string()
    print(frev)

    geo = u.unpack_int()
    nQuantities = u.unpack_int()

    Units = []
    for i in range(nQuantities):
        ns = u.unpack_int()
        Units.append(u.unpack_string())

    print("nQuantities = %d" %(nQuantities))
    print(Units)

    Var=[]
    for i in range(nQuantities):
        Var.append([])

    test = 1
    while test:
        try:
            dum = u.unpack_farray(nQuantities,u.unpack_float)
            for i in range(nQuantities):
                Var[i].append(dum[i])
        except EOFError:
            test = 0

    t2 = time.time()
    print(t2-t1)
    return Var

def readXDRpext_new(fname, t0=0, myRank=0, nProc=1):
    """Reads extraction file 
     Input: file name
     Output: Var[0:nparticles-1][0:nQuantities-1]"""
    t1 = time.time()
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
        
        if t0>0: # approach based on time criterion
            Var=[]
            for i in range(nQuantities):
                Var.append([])
         
            nQuantities -= 1
            while True:
                try:
                    tt = u.unpack_float()
                    if tt<t0:
                        u.set_position(u.get_position()+nQuantities*4)
                    else:
                        dum = u.unpack_farray(nQuantities,u.unpack_float)
                        Var[0].append(tt)
                        for i in range(nQuantities):
                            Var[i+1].append(dum[i])
                except EOFError:
                    break
            
            Var=np.array(Var)            
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
    return Var


def readXDRpextMPI(fname, myRank=0, nProc=1):
    """Reads extraction file 
     Input: file name
     Output: Var[0:nparticles-1,0:nQuantities-1]"""
    import time
    import struct
    import os
         
    t1 = time.time()
               
    print(fname)
    statinfo = os.stat(fname)
    fsize = statinfo.st_size
   

    
    with open(fname, mode='rb') as file:
        file.read(12)
        
        data = file.read(4) # read string length
        x = struct.unpack('>L', data)[0]
        nb = 4*divmod(x+3,4)[0]
        title = file.read(nb)
        print(title)
        
        file.read(4) # integer        
        data = file.read(4) # read string length
        x = struct.unpack('>L', data)[0]
        nb = 4*divmod(x+3,4)[0]
        frev = file.read(nb)
        print(frev)
        
        data = file.read(4)
        geo = struct.unpack('>L', data)[0]
        data = file.read(4)
        nQuantities = struct.unpack('>L', data)[0]
       
        Units = []
        for i in range(nQuantities):
            file.read(4)
            data = file.read(4)
            x = struct.unpack('>L', data)[0]
            nb = 4*divmod(x+3,4)[0]
            Units.append(file.read(nb).decode('ascii').rstrip('\x00'))

        print("nQuantities = %d" %(nQuantities))
        print(Units)
        nPart = divmod(fsize-file.tell(),4*nQuantities)[0]   
        if nPart==0: print("NO PARTICLE RECORDS FOUND"); exit()
        print("Number of particle records in file = %d" %(nPart))        
        
        ranges={}
        
        [w,r]=divmod(nPart,nProc)
        
        ranges[0]=(0,w+int(r>0)-1)
        
        for j in range(1,nProc):
            ranges[j]=(ranges[j-1][1]+1,ranges[j-1][1]+w+int(j<r))
        
       
#             print(ranges[myRank])
        nb = file.tell() + ranges[myRank][0]*nQuantities*4
        file.seek(nb)
            
        istart = ranges[myRank][0]
        ifinish = ranges[myRank][1]
        myPart = ifinish - istart + 1
        print("My rank %d, my particle records %d" %(myRank,myPart))

        Var=np.zeros((myPart,nQuantities))
      
        for i in range(istart,ifinish+1):
            dum = []
            for j in range(nQuantities):
                data = file.read(4)
                x = struct.unpack('>f', data)[0]
                dum.append(x)
            Var[i-istart,:]=dum[:]        

    t2 = time.time()
    print("File reading time %.2f s" %(t2-t1))
    return Var   
# ####
#     while True:
#         try:
#             s = u.unpack_int()
#             if not s in spc:
#                 u.set_position(u.get_position()+nQuantities*4)
#             else:
#                 dum = u.unpack_farray(nQuantities,u.unpack_float)
#                 for i in range(nQuantities):
#                     Part[s-1][i].append(dum[i])
#         except EOFError:
#             break
# ###

def readXDRstruct(fname):
    """Read structure file
    returns coordinate of segments"""

    f = open(fname,'rb');
    u = xdrlib.Unpacker(f.read()) 
    f.close()
    
    title = u.unpack_string()
    #print title
    u.unpack_int()
    geo = u.unpack_int()

    dim = u.unpack_int()
    #print 'dim=',dim

    nty=[];mty=[];nid=[];mid=[]
    xa,xb,ya,yb,za,zb=[],[],[],[],[],[]

    test = 1
    while test:
        try:
            nty.append(u.unpack_int())
            mty.append(u.unpack_int())
            nid.append(u.unpack_int())
            mid.append(u.unpack_int())
            xa.append(u.unpack_float())
            ya.append(u.unpack_float())
            za.append(u.unpack_float())
            xb.append(u.unpack_float())
            yb.append(u.unpack_float())
            zb.append(u.unpack_float())
        except EOFError:
            test = 0

#import matplotlib.pyplot as plt
#import matplotlib.collections as mc
#import pylab as pl
#lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
#lc = mc.LineCollection(lines, linewidths=1)
#fig, ax = pl.subplots()
#ax.add_collection(lc)
#ax.autoscale()
#plt.show()

    return(xa,ya,za,xb,yb,zb)

def readXDRsclr(fname,**kwargs):
    """Reads scalar file 
     Input: file name
     Output: time, Part[0:nSpecies-1][0:nPart-1][0:nQuantities-1]
     Based on Tony's Matlab script"""
    t1 = time.time()
    pout= not kwargs["silent"] if "silent" in kwargs else True

    with open(fname, mode='rb') as file:
        fileContent = file.read()
        u = xdrlib.Unpacker(fileContent) 
        
        ftype = u.unpack_int()
        if(ftype!=3):
            print('Not a Sclr File')
            return()

        fversion = u.unpack_int()
        u.unpack_int()
        title = u.unpack_string().decode() # read title
        if pout: print(title)

        u.unpack_int();
        frev = u.unpack_string().decode()
        if pout: print(frev)

        tStep = u.unpack_float()
        if pout: print("Time %.2f" %(tStep))

        geo = u.unpack_int()
        nDomains = u.unpack_int()
        nQuantities = u.unpack_int()
#         print("nQuantities = %d" %(nQuantities))

        VarNames = []
        for i in range(nQuantities):
            u.unpack_int()
            VarNames.append(u.unpack_string().decode())

        VarUnits = []
        for i in range(nQuantities):
            u.unpack_int()
            VarUnits.append(u.unpack_string().decode())
        
        for i in range(nQuantities):
            if pout: print(str(i) + " " + VarNames[i] + " " + VarUnits[i]);                

        Var=[];X=[];Y=[];Z=[];
        for i in range(nQuantities):
            Var.append([])
        

        for iDom in range(nDomains):
            iR = u.unpack_int()
            jR = u.unpack_int()
            kR = u.unpack_int()
            nI = u.unpack_int()
            XT = u.unpack_farray(nI,u.unpack_float)
            nJ = u.unpack_int()
            YT = u.unpack_farray(nJ,u.unpack_float)
            nK = u.unpack_int()
            ZT = u.unpack_farray(nK,u.unpack_float)

            XL=[];YL=[];ZL=[];
            for k in range(nK):
                for j in range(nJ):
                    for i in range(nI):
                        # XL.append(XT[i])
                        YL.append(YT[j])
                        ZL.append(ZT[k])
            XL=XT*nJ*nK
            
            X.extend(XL)
            Y.extend(YL)
            Z.extend(ZL)
            
            for k in range(nQuantities):
                Var[k].extend(u.unpack_farray(nI*nJ*nK,u.unpack_float))

    UX, indexX = np.unique(X, return_inverse=True);
    UY, indexY = np.unique(Y, return_inverse=True);
    UZ, indexZ = np.unique(Z, return_inverse=True);

    nX=np.size(UX); nY=np.size(UY); nZ=np.size(UZ);
    newVar=np.zeros((nQuantities,nX,nY,nZ))
    MaxIndex=np.size(X);

    Var=np.array(Var).reshape((nQuantities,MaxIndex));
    newVar[:,indexX[:],indexY[:],indexZ[:]]=Var[:,:];

    file.close    
    del X, Y, Z, Var
    t2 = time.time()

    print("readXDRsclr elapsed time = {:.3f} s".format(t2-t1))

    return (UX,UY,UZ,newVar,VarNames,VarUnits,tStep) 

def read_hist(fname):
    """Reads history file """

    headr=[];tt=[]; dat=[];
    with open(fname) as infile:
        for line in infile:
            if line[0]=="#": headr.append(line);
            else: 
                tt.append(line.split()[0]);
                dat.append(line.split()[1:]);

    title=headr[0].strip();         
    nProbes=int(headr[3].split()[-1]);
    Vars=[];Units=[];
    for i in range(nProbes):
        Vars.append(headr[4+i].strip().split(':')[1][1:]); # removes empty space at the beginning
        Units.append(headr[4+i].strip().split(':')[2]);

    tt=np.array(tt).astype(float);
    dat=np.array(dat).astype(float);

    return(title,nProbes,Vars,Units,tt,dat)

def readXDRflds(fname,**kwargs):
    """Reads vector file 
     Input: file name
     Output: Coordinates,Variables[nQuantities,3,nX,nY,nZ,Units,Time]
     Based on Tony's Matlab script"""
    t1 = time.time()
    
    pout= not kwargs["silent"] if "silent" in kwargs else True
    
    f = open(fname,'rb'); # open file
    u = xdrlib.Unpacker(f.read()) 
    f.close()

    ftype = u.unpack_int()
    if(ftype!=2): 
        print('Not a Flds File')

    fversion = u.unpack_int()
    ns = u.unpack_int();title = u.unpack_string() # read title
    if pout: print(title)

    ns = u.unpack_int();frev = u.unpack_string()
    if pout: print(frev)

    tStep = u.unpack_float()
    if pout: print("Time = %.2f" %(tStep))

    geo = u.unpack_int(); nDomains = u.unpack_int(); nQuantities = u.unpack_int();

    VarNames = []
    for i in range(nQuantities):
        ns = u.unpack_int()
        VarNames.append(u.unpack_string().decode())

    VarUnits = []
    for i in range(nQuantities):
        ns = u.unpack_int()
        VarUnits.append(u.unpack_string().decode())

    for i in range(nQuantities):
        if pout: print(str(i) + " " + VarNames[i] + " " + VarUnits[i]);    

    Var=[];X=[];Y=[];Z=[];
    for i in range(nQuantities): Var.append([]);

    Check=1;
    rdCounter = 1;

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
            Var[k].extend(u.unpack_farray(3*nI*nJ*nK,u.unpack_float))


    UX, indexX = np.unique(X, return_inverse=True);
    UY, indexY = np.unique(Y, return_inverse=True);
    UZ, indexZ = np.unique(Z, return_inverse=True);

    nX=np.size(UX);nY=np.size(UY);nZ=np.size(UZ);
    newVar=np.zeros((nQuantities,3,nX,nY,nZ))

    MaxIndex=np.size(X);

    Var=np.array(Var).reshape((nQuantities,MaxIndex,3));	
    newVar[:,0,indexX[:],indexY[:],indexZ[:]]=Var[:,:,0];  
    newVar[:,1,indexX[:],indexY[:],indexZ[:]]=Var[:,:,1];
    newVar[:,2,indexX[:],indexY[:],indexZ[:]]=Var[:,:,2];

    t2 = time.time()
    print("readXDRflds elapsed time = {:.3f} s".format(t2-t1))

    return (UX,UY,UZ,newVar,VarNames,VarUnits,tStep)  

def readXDRpart1(fname,spc):
    """Reads particle dump file (new version)
     Input: file name (string), species (integer or list)
     Output: time, Part[0:nSpecies-1][0:nPart-1][0:nQuantities-1]"""
    t1 = time.time()
    f = open(fname,'rb'); # open file
    u = xdrlib.Unpacker(f.read()) 
    f.close()

    ftype = u.unpack_int()
    fversion = u.unpack_int()

    u.unpack_int();
    title = u.unpack_string() # read title
    print(title)

    u.unpack_int();
    frev = u.unpack_string()
    print(frev)

    tt = u.unpack_float()
    print("Time =",tt)
    geo = u.unpack_int() 
    SymFlag = u.unpack_farray(3,u.unpack_int)
    nSpecies = u.unpack_int()
    nParticles = u.unpack_int()
    nQuantities = u.unpack_int()

    Units = []
    for i in range(nQuantities):
        u.unpack_int()
        Units.append(u.unpack_string())

    print("nSpecies =",nSpecies, "nQuantities =",nQuantities,"nParticles =",nParticles)
    print("Units: ", Units)

    Part = [];
    for s in range(nSpecies):
        Part.append([])
        for i in range(nQuantities):
            Part[s].append([])
    
    if type(spc)!=list: spc=[spc];
    
    while True:
        try:
            s = u.unpack_int()
            if not s in spc:
                u.set_position(u.get_position()+nQuantities*4)
            else:
                dum = u.unpack_farray(nQuantities,u.unpack_float)
                for i in range(nQuantities):
                    Part[s-1][i].append(dum[i])
        except EOFError:
            break

    t2 = time.time()
    print("readXDRpart1 time:", (t2-t1))

    return (tt,Part)

def get_t(fname):
    """Returns time step from *.p4 file"""
    
    import os, struct
#     print(fname)
    statinfo = os.stat(fname)
    fsize = statinfo.st_size
    with open(fname, mode='rb') as file:
        file.read(12)
        
        data = file.read(4) # read string length
        x = struct.unpack('>L', data)[0]
        nb = 4*divmod(x+3,4)[0]
        title = file.read(nb).decode()
#         print(title)
        
        file.read(4) # integer        
        data = file.read(4) # read string length
        x = struct.unpack('>L', data)[0]
        nb = 4*divmod(x+3,4)[0]
        frev = file.read(nb).decode()
#         print(frev)
        
        data = file.read(4)
        tStep = struct.unpack('>f', data)[0]
#         print(tStep)
        return(float(tStep))

def get_ind(t,**kwargs):
    import os, glob
    if "dirname" in kwargs:
        os.chdir(kwargs["dirname"])
    dirname=os.getcwd()
    prefix = 'sclr'
    fileList = glob.glob(prefix + '*.p4')
    if len(fileList)==0:
        prefix = 'flds'
        fileList = glob.glob(prefix + '*.p4')
    
    indt = sorted([int(f[:-3][4:]) for f in fileList])
    for i, ind in enumerate(indt[1:]) :
        if get_t(prefix + str(ind) + '.p4')>t: 
            break
    return(indt[i])
