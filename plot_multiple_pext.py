#!/usr/bin/env python
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, getopt, os, os.path
from mpi4py import MPI
import numpy as np

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';

fnames = np.arange(400, 600, 0.5)
nFiles=int(len(fnames));

# distribute tasks
j=myRank
myList=[];
while j<nFiles:
  myList.append(fnames[j])
  j+=nProc

print "my list:", myList

for i in myList:
  fid="/g/g19/holod1/python/plot_pext.py -p 3 -s 4 -maxt "+str(i)+" -save";
  os.system(fid);

comm.Barrier()
exit()
