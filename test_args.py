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
    argstring = " ".join(argline.replace("="," ").split()) # remove double spaces
    print(argstring)
    return(["-"+ s.strip().replace(" ","=") for s in argstring.split("-") if s])

class C(object):
    pass
arg=C()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', type=int, help='file index')
parser.add_argument('-t', type=float, action='append', help='time')

print(arg2list("-i 1000 -t -10 -t 11"))
parser.parse_args(arg2list("-i=1000 -t -10 -t 11"),namespace=arg)
print(arg.t)

if arg.i is not None:
    num=arg.i
elif arg.t is not None:
    num=get_ind(arg.t)
else:
    exit()

print(num)
exit()

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


argline = "-i " + str(num) + " -p 1 -save"
t1 = time.time()
plt_sclr(arglist=arg2list(argline),sdata=sdata)
t2 = time.time()
print("plt_sclr time = {:.3f}".format(t2-t1))

# argline = "-i " + str(num) + " -p 1 -save"
# t1 = time.time()
# plt_flds(arglist=arg2list(argline),fdata=fdata)
# t2 = time.time()
# print("plt_flds time = {:.3f}".format(t2-t1))

argline = "-i " + str(num) + " -si 2 -se 1 -slicex 0 -slicex 0.1"
t1 = time.time()
rhol(arglist=arg2list(argline),sdata=sdata)
t2 = time.time()
print("rhol time = {:.3f}".format(t2-t1))

exit()