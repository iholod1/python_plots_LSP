#!/usr/bin/env python
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, argparse, os, os.path
from mpi4py import MPI
from read_xdr import get_ind

comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd();
class C(object):
    pass
arg=C()
parser = argparse.ArgumentParser(description='Process arguments');
parser.add_argument('-i', type=int, help="initial index");
parser.add_argument('-f', type=int, help="final index");

parser.parse_args(namespace=arg)
istart = int(arg.i) if arg.i != None else None
ifinish = int(arg.f) if arg.f != None else None

nEntries = 0
tt=[]; dat = [];
if myRank==0:
    fid=os.path.join(dirname,'plots','pdf_extract1_Zlocation.out')
    print(fid)
    headr = [];
    with open(fid) as infile:
        for line in infile:
            if len(line)==0: continue;
            if line[0]=="#": headr.append(line.split("\t"));
            else:
                tt.append(float(line.split()[0]))
                dat.append(float(line.split()[1]))

tt = comm.bcast(tt, root=0);
dat = comm.bcast(dat, root=0);
nEntries = len(tt)
iEntries = [i for i in range(nEntries)]

j=myRank
myList=[];
while j<nEntries:
    myList.append(iEntries[j])
    j+=nProc

print("my list: " + str(myList))


for j in myList:
    i =  get_ind(tt[j])
    fid = os.path.join(dirname,"scalar.py -i {:d} -f1 {:f}".format(i,dat[j]))
    print(fid)
    os.system("python " + fid); # call python script

comm.Barrier()
exit()
