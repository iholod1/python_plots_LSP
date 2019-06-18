#!/usr/bin/env python
# script to plot from sclr*.p4
# I. Holod, 06/27/17


import sys, getopt, argparse
import os.path
import math
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
import pylab as pl

from read_xdr import readXDRsclr


dirname=os.getcwd() + '/';

class C(object):
  pass
arg=C()
parser = argparse.ArgumentParser(description='Process integer arguments');
parser.add_argument('-i', type=int, help='time index');
parser.add_argument('-p', type=int, help='probe');
parser.add_argument('-x0', type=float, help='min X');
parser.add_argument('-x1', type=float, help='max X');
parser.add_argument('-z0', type=float, help='min Z');
parser.add_argument('-z1', type=float, help='max Z');
parser.add_argument('-x', action='store_true', help='add X-component');
parser.add_argument('-y', action='store_true', help='add Y-component');
parser.add_argument('-z', action='store_true', help='add Z-component');

parser.parse_args(namespace=arg)
num = int(arg.i) if arg.i != None else 1;
ivar = int(arg.p) if arg.p != None else 0;
x0 = float(arg.x0) if arg.x0 != None else 0.0;
x1 = float(arg.x1) if (arg.x1 != None)&(arg.x1 > x0) else x0;
z0 = float(arg.z0) if arg.z0 != None else 0.0;
z1 = float(arg.z1) if (arg.z1 != None)&(arg.z1 > z0) else z0;
xc = 1.0 if arg.x else 0.0;
yc = 1.0 if arg.y else 0.0;
zc = 1.0 if arg.z else 0.0;



print "bounding box: (x0=%.3f,x1=%.3f,z0=%.3f,z1=%.3f)" %(x0,x1,z0,z1)
fname=dirname+'sclr'+str(num)+'.p4'
print fname;
#(X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname)
#dat=Var[ivar];
#dat=np.squeeze(dat[:,0,:]);
exit()


