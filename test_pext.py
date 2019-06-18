import numpy as np

def readXDRpextMPI1(fname, myRank=0, nProc=1):
    """Reads extraction file 
     Input: file name
     Output: Var[0:nparticles-1][0:nQuantities-1]"""
    import time
    import struct
    import os
         
    t1 = time.time()
               
    print(fname)
    statinfo = os.stat(fname)
    fsize = statinfo.st_size
   

    
    with open(fname, mode='rb') as file:
        file.read(8)
        file.read(4) # integer
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
        print(file.tell())
            
        istart = ranges[myRank][0]
        ifinish = ranges[myRank][1]
        myPart = ifinish - istart + 1
        print("My rank %d, my particle number %d" %(myRank,myPart))
        Var=np.zeros((myPart,nQuantities))
      
        for i in range(istart,ifinish+1):
            dum = []
            for j in range(nQuantities):
                data = file.read(4)
                x = struct.unpack('>f', data)[0]
                dum.append(x)
            Var[i-istart,:]=dum[:]        

    file.close()
    t2 = time.time()
    print("File reading time %.2f s" %(t2-t1))
    return Var, myPart, nQuantities                
        

def readXDRpextMPI(fname, myRank=0, nProc=1):
    """Reads extraction file 
     Input: file name
     Output: Var[0:nparticles-1][0:nQuantities-1]"""
    import time
    import xdrlib
         
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

        nPart = divmod(len(u.get_buffer())-u.get_position(),4*nQuantities)[0]
        if nPart==0: print("NO PARTICLE RECORDS FOUND"); exit()
        print("Number of particle records in file = %d" %(nPart))
        
        ranges={}
        
        [w,r]=divmod(nPart,nProc)
        
        ranges[0]=(0,w+int(r>0)-1)
        
        for j in range(1,nProc):
            ranges[j]=(ranges[j-1][1]+1,ranges[j-1][1]+w+int(j<r))
        
#             print(ranges[myRank])
        u.set_position(u.get_position()+ranges[myRank][0]*nQuantities*4)
            
        istart = ranges[myRank][0]
        ifinish = ranges[myRank][1]
        myPart = ifinish - istart + 1
        print("My rank %d, my particle number %d" %(myRank,myPart))
        Var=np.zeros((myPart,nQuantities))
      
        for i in range(istart,ifinish+1):
            dum = u.unpack_farray(nQuantities,u.unpack_float)
            Var[i-istart,:]=dum[:]
                
            
        
    file.close()
    t2 = time.time()
    print("File reading time %.2f s" %(t2-t1))
    return Var, myPart, nQuantities


Var1, nPart, nQuantities = readXDRpextMPI1("pext1.p4",myRank=0,nProc=1)
Var2, nPart, nQuantities = readXDRpextMPI("pext1.p4",myRank=0,nProc=1)

print(Var1==Var2)