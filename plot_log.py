#!/usr/bin/env python3

import os.path
import numpy as np
from tools.plotting import plot1d

mydir=os.getcwd()
print(mydir)
fid=os.path.join(mydir,'log.out')
print(fid)
tt=[]; CENet = []; headr = [];
with open(fid) as infile:
	for line in infile:
		if len(line)==0: continue
		line = line.strip()
		if "Number of particles=" in line:
			tt.append(float(line.split()[1][5:]))
			CENet.append(float(line0.split()[9][:-1]))
		if "dEnP" in line:
			line0 = line			

# print(tt)
# print(line0.split())
# print(float(line.split()[9][:-1]))
tt = np.array(tt)
CENet = np.array(CENet)

plot1d(tt,CENet,"ns","joules")

if not os.path.exists(os.path.join(mydir,"plots")):  os.makedirs(os.path.join(mydir,"plots"))
fout = open(os.path.join(mydir,'plots','cenet_hist.out'), 'w')
fout.write("#\tCumulative net energy\ttime\tns\tunits\tjoules\n");
for i in range(len(tt)):
	fout.write("%.8e %.8e" %(tt[i], CENet[i]));
	fout.write("\n");
fout.close()