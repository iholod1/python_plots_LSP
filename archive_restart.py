#!/usr/bin/env python
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, argparse, os, os.path
from mpi4py import MPI
import time

t1 = time.time()

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';
class C(object):
  pass
arg=C()
parser = argparse.ArgumentParser(description='Process arguments');
parser.add_argument('-a', type=str, help="arguments for plot_sclr.py");

parser.parse_args(namespace=arg)
arg_str = str(arg.a) if arg.a != None else ""; 

if myRank==0:
  fnames=[];
  lst_dir=os.listdir(dirname)
  for fid in lst_dir:
    if fid.startswith('restart') and fid.endswith('.dat'): 
      fnames.append(int(fid[7:-4]))
  print "Number of PEs = ", nProc
  print "Total number of files = ", len(fnames)
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

print "my list:", myList
if myRank==0:
  os.system("tar -cvf test.tar history.p4 pext* log.txt")

for i in myList:
  cmd="tar --append --file=test.tar " + "restart" + str(i) + ".dat"
  os.system(cmd);

comm.Barrier()

t2 = time.time()
if myRank==0: print(t2-t1)
exit()
