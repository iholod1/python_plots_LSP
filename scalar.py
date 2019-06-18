#!/usr/bin/env python
# script for multiple operations

import argparse
import sys
import time
from plot_sclr import plt_sclr
from plot_flds import plt_flds
from rhol import rhol
from read_xdr import readXDRsclr, readXDRflds
from read_xdr import get_ind

def arg2list(argline):
    argstring = " ".join(argline.split()) # remove double spaces
    return(["-"+ s.strip().replace(" ","=") for s in argstring.split("-") if s])

class C(object):
    pass
arg=C()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', type=int, help='file index')
parser.add_argument('-t', type=float, help='time')
parser.add_argument('-f1', type=float, help='float type argument 1')
parser.add_argument('-f2', type=float, help='float type argument 2')
parser.add_argument('-f3', type=float, help='float type argument 3')
parser.add_argument('-i1', type=int, help='integer type argument 1')
parser.add_argument('-i2', type=int, help='integer type argument 2')
parser.add_argument('-i3', type=int, help='integer type argument 3')


parser.parse_args(namespace=arg)

if arg.i is not None:
    num=arg.i
elif arg.t is not None:
    num=get_ind(arg.t)
else:
    exit()

f1 = arg.f1 if arg.f1 != None else None
f2 = arg.f2 if arg.f2 != None else None
f3 = arg.f3 if arg.f3 != None else None
i1 = arg.i1 if arg.i1 != None else None
i2 = arg.i2 if arg.i2 != None else None
i3 = arg.i3 if arg.i3 != None else None

    
############## PRE-LOADING ########################
fname='sclr'+str(num)+'.p4'
print(fname)
t1 = time.time()
sdata=readXDRsclr(fname,silent=1)
print("t1 = {:.3f}".format(time.time()-t1))

# fname='flds'+str(num)+'.p4'
# print(fname)
# t1 = time.time()
# fdata=readXDRflds(fname,silent=1)
# print("t1 = {:.3f}".format(time.time()-t1))
####################################################




# argline = "-i " + str(num) + " -p 1 -save"
# t1 = time.time()
# plt_sclr(arglist=arg2list(argline),sdata=sdata)
# t2 = time.time()
# print("plt_sclr time = {:.3f}".format(t2-t1))

# argline = "-i " + str(num) + " -p 1 -save"
# t1 = time.time()
# plt_flds(arglist=arg2list(argline),fdata=fdata)
# t2 = time.time()
# print("plt_flds time = {:.3f}".format(t2-t1))

argline = "-i {:d} -si 12 -se 1 -slicex {:f} -minz 0 -maxz {:f}".format(num,0,f1)
t1 = time.time()

rhol(arglist=arg2list(argline),sdata=sdata)
t2 = time.time()
print("rhol time = {:.3f}".format(t2-t1))

exit()