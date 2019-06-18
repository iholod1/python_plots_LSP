#!/usr/bin/env python
# NOT COMPLETED YET
# I. Holod, 06/09/17

import sys, argparse, os, os.path
from mpi4py import MPI
from return_local import return_local_sclr
import numpy as np

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';
class C(object):
  pass
arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-x0', type=float, help="min x");
parser.add_argument('-x1', type=float, help="max x");
parser.add_argument('-z0', type=float, help="min z");
parser.add_argument('-z1', type=float, help="max z");


parser.parse_args(namespace=arg)
x0 = float(arg.x0) if arg.x0 != None else 0;
x1 = float(arg.x1) if arg.x1 != None else 3.0;        
z0 = float(arg.z0) if arg.z0 != None else 0;
z1 = float(arg.z1) if arg.z1 != None else 6.0;        


if myRank==0:
  fnames=[];
  lst_dir=os.listdir(dirname)
  for fid in lst_dir:
    if 'sclr' in fid and '.p4' in fid: 
      fnames.append(int(fid[4:-3]))
  print "Number of PEs = ", nProc
  print "Total number of files = ", len(fnames)
  if not os.path.exists(dirname+"plots"):
    os.makedirs(dirname+"plots")
else:
  fnames=None;

comm.Barrier()
fnames=comm.bcast(fnames, root=0);


nFiles=int(len(fnames));

dat = np.zeros((nFiles,2));
newDat = np.zeros((nFiles,2));

# distribute tasks
j=myRank
myList=[]; myInd=[];
while j<nFiles:
  myList.append(fnames[j])
  myInd.append(j)
  j+=nProc

#print "my list:", myList

for i in myInd:
  (tdum,vdum,vname,vunit)=return_local_beta(dirname,fnames[i],x0,x1,z0,z1);
  dat[i,0]=tdum
  dat[i,1]=vdum

comm.Barrier()
comm.Reduce(dat,newDat,op=MPI.SUM,root=0);

if myRank==0: 
  ind = np.argsort(newDat[:,0]); newDat = newDat[ind]; # sort by first column
  fout = open(dirname+"local_probe"+"%d" %(probe)+'.out', 'w')
  fout.write("#\t" + "time\t" + vname + "\t" + vunit + "\t" + \
             "value at (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1) + "\n");
  for i in range(newDat.shape[0]):
    fout.write("%.8e %.8e" %(newDat[i,0], newDat[i,1]));
    fout.write("\n");
  fout.close()

exit()
