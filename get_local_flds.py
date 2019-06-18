#!/usr/bin/env python
# scan all flds file in directory and write box-averaged value 
# of probe quantity (all components) into local_probe.out file
# I. Holod, 07/26/17

import sys, argparse, os, os.path
from mpi4py import MPI
from return_local import return_local_flds
import numpy as np

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';
class C(object):
  pass
arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-p', type=int, help="probe");
parser.add_argument('-minx', type=float, help="min x");
parser.add_argument('-maxx', type=float, help="max x");
parser.add_argument('-minz', type=float, help="min z");
parser.add_argument('-maxz', type=float, help="max z");

parser.parse_args(namespace=arg)
probe = int(arg.p) if arg.p != None else 0;
x0 = float(arg.minx) if arg.minx != None else 0;
x1 = float(arg.maxx) if arg.maxx != None else 3.0;        
z0 = float(arg.minz) if arg.minz != None else 0;
z1 = float(arg.maxz) if arg.maxz != None else 6.0;  

if myRank==0:
  fnames=[];
  lst_dir=os.listdir(dirname)
  for fid in lst_dir:
    if 'flds' in fid and '.p4' in fid: 
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

dat = np.zeros((nFiles,4));
newDat = np.zeros((nFiles,4));

# distribute tasks
j=myRank
myInd=[];
while j<nFiles:
  myInd.append(j)
  j+=nProc

for i in myInd:
  (tdum,vx,vy,vz,vname,vunit)=return_local_flds(dirname,fnames[i],probe,x0,x1,z0,z1);
  dat[i,0]=tdum
  dat[i,1]=vx;
  dat[i,2]=vy;
  dat[i,3]=vz;


comm.Barrier()
comm.Reduce(dat,newDat,op=MPI.SUM,root=0);

if myRank==0: 
  ind = np.argsort(newDat[:,0]); newDat = newDat[ind]; # sort by first column
  fout = open(dirname+"local_probe"+"%d" %(probe)+'.out', 'w')
  fout.write("#\t" + "time\t" + vname + "\t" + vunit + "\t" + \
             "(x,y,z)-components at (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1) + "\n");
  for i in range(newDat.shape[0]):
    fout.write("%.8e %.8e %.8e %.8e" %(newDat[i,0], newDat[i,1], newDat[i,2], newDat[i,3]));
    fout.write("\n");
  fout.close()
  print "All done"



exit()
