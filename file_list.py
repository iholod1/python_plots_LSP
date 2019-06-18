#!/usr/bin/env python
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, getopt, os, os.path
from mpi4py import MPI

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';

if myRank==0:
  fnames=[];
  lst_dir=os.listdir(dirname)
  for fid in lst_dir:
    if 'part' in fid and '.p4' in fid: 
      fnames.append(int(fid[4:-3]))
  print "Number of PEs = ", nProc
  print "Total number of files = ", len(fnames)
  if not os.path.exists(dirname+"plots"):
    os.makedirs(dirname+"plots")
else:
  fnames=None;

fnames=comm.bcast(fnames, root=0);

#for i in range(1,5000000):
#  fid = "part"+str(i)+".p4";
##  if fid in lst_dir: fnames.append(fid);
#  if fid in lst_dir: fnames.append(i);

nFiles=int(len(fnames));

# distribute tasks
j=myRank
myList=[];
while j<nFiles:
  myList.append(fnames[j])
  j+=nProc

print "my list:", myList

for i in myList:
  fid="/g/g19/holod1/python/plot_local_pdf.py -i "+str(i)+" -s 4 -b 0 -d 0 -u 0.1 -l 30.0 -r 32.0";
  os.system(fid); # call plotting script

exit()
