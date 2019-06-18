#!/usr/bin/env python
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, argparse, os, os.path
from mpi4py import MPI

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';
class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description='Process arguments');
parser.add_argument('-a', type=str, help="arguments for plot_sclr.py");
parser.add_argument('-i', type=int, help="initial index");
parser.add_argument('-f', type=int, help="final index");

parser.parse_args(namespace=arg)
arg_str = str(arg.a) if arg.a != None else " -p 1 -save"; 
istart = int(arg.i) if arg.i != None else None
ifinish = int(arg.f) if arg.f != None else None

if myRank==0:
    fnames=[];
    lst_dir=os.listdir(dirname)
    for fid in lst_dir:
        if fid.startswith('sclr') and fid.endswith('.p4'):
            dum = int(fid[4:-3])
            if (istart and dum<istart) or (ifinish and dum>ifinish):
#             if (istart and dum<istart):                
                continue
            else:
                fnames.append(dum)
    print("Number of PEs = %d" %(nProc))
    print("Total number of files = %d" %(len(fnames)))
    if not os.path.exists(dirname+"plots"):
        os.makedirs(dirname+"plots")
else:
    fnames=None;

comm.Barrier()
fnames=comm.bcast(fnames, root=0);

nFiles=int(len(fnames));

# distribute tasks
j=myRank
myList=[];
while j<nFiles:
    myList.append(fnames[j])
    j+=nProc

print("my list: " + str(myList))

for i in myList:
  fid="/g/g19/holod1/python/plot_flds.py -i "+str(i) + " " + arg_str;
  os.system(fid); # call plotting script

comm.Barrier()
exit()
