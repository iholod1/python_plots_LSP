#!/usr/bin/env python3
import numpy as np
import time

def test(**kwargs):
    if ("fname" in kwargs):
        print(kwargs["fname"])
    return()

test(fname="test")

X=[0,1,2,3]; ni=4
Y=[0]; nj=1
Z=[0,1]; nk=2

t1 = time.clock()
XL=[]; YL=[]; ZL=[]
for k in range(nk):
    for j in range(nj):
        for i in range(ni):
            YL.append(Y[j])
            ZL.append(Z[k])

XL=X*nj*nk

print(XL)
print(YL)
print(ZL)
print("t1 = {:.4e}".format(time.clock()-t1))


t1 = time.clock()
LGrid=np.meshgrid(Z,Y,X)
XL=LGrid[2].flatten()
YL=LGrid[1].flatten()
ZL=LGrid[0].flatten()
print(XL)
print(YL)
print(ZL)
print("t1 = {:.4e}".format(time.clock()-t1))


dat=np.array([[1,2],[3,1],[2,4]])
# print(dat)
ind = [np.lexsort((dat[:, 1], dat[:, 0]))]
# print(ind)
dat = dat[ind]
# print(dat)


a=np.array([[1,2,3],[2,2,2]])
b=np.array([2,2]).reshape((2,1))
c = a*b
# print(a)
# print(c)

a = np.array([1,2,3])
print(a[::-1])


# print(0.5*(Ebin + np.roll(Ebin,1))[1:])
# print(0.5*(Ebin - np.roll(Ebin,1))[1:])

# delE = (maxE-minE)/float(nEBin)
# Ebin = np.linspace(minE,maxE,nEBin+1)[:-1] + delE/2
