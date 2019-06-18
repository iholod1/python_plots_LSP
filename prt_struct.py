#!/usr/bin/env python
# python script to read struct.p4

import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab as pl

from read_xdr import readXDRstruct

(xa,ya,za,xb,yb,zb)=readXDRstruct('struct.p4')

lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]

lc = mc.LineCollection(lines, linewidths=1)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.set_ylim([min(xa),max(xb)]);
ax.set_xlim([min(za),max(zb)]);
plt.show()


