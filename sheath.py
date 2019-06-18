#!/usr/bin/env python
# calculate sheath velocity from file containing position
# I. Holod, 07/25/17


import os
import numpy as np

import matplotlib.pyplot as plt

dirname=os.getcwd() + '/';

fname = dirname+"sheath_front.out"
headr=[];tt=[]; xx=[];
with open(fname) as infile:
    for line in infile:
        if line[0]=="#": 
            headr.append(line.split("\t"));
        else: 
            tt.append(line.split()[0]);
            xx.append(line.split()[1]);

dat=np.array((tt[:],xx[:])).astype(float)
print(dat)


ind = np.argsort(dat[0,:]); dat = dat[:,ind]; # sort by first column
x=dat[0,:].reshape(dat.shape[1],1);
y=dat[1,:].reshape(dat.shape[1],1);

# ind2=np.argwhere((x[:,0]>350)&(x[:,0]<480))
# x=x[ind2].reshape((ind2.shape[0],1))
# y=y[ind2].reshape((ind2.shape[0],1))

vs = np.polyfit(x.flatten(), y.flatten(), 1)
print("sheath velocity (m/s) = ", vs[0]*1e-2/1e-9)

f, ax = plt.subplots()
ax.plot(x,y,linewidth=2)
ax.plot(x,vs[1]+vs[0]*x,linewidth=1)
ax.set_xlim([min(x),max(x)])

ttl =  "sheat front position vs time";
ax.set(title=ttl, xlabel="ns", ylabel="cm")

#f.savefig(dirname + "sheath_position.png",dpi=200)

plt.show()

exit()
