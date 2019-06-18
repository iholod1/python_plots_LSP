#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from read_xdr import read_hist
from read_xdr import readXDRpart
from hist1d import hist1dlin, hist1dlog
from energy import kin_en, theta_ang

massp=1.67e-24; #proton mass (g)
amu=9.1e-28; #positron mass (g)

chargee=-1.0;chargei=1.0;

# dpf-like test
#masse=amu;ze=-1.0;massi=2.0*massp;zi=1.0;n=6.67e16;te0=2.0;ti0=1.0;
# Nanbu's paper test
masse=amu;ze=-1.0;massi=5*amu;zi=1.0;n=1.0e15;te0=1000.0;ti0=100.0;maxt=6000e-9; # sec
# Nanbu's test Te=120
#masse=amu;ze=-1.0;massi=5*amu;zi=1.0;n=1.0e15;te0=120.0;ti0=100.0;
# equal mass
#masse=amu;ze=-1.0;massi=1.01*amu;zi=1.0;n=1.0e15;te0=1000.0;ti0=100.0;
# Tony's case
#masse=amu;ze=-1.0;massi=2*massp;zi=1.0;n=1.0e18;te0=1000.0;ti0=100.0;

#maxt=500e-9; # sec
#maxt=10e-9; # sec

nt=100;
tt=np.linspace(0.0,maxt,nt);
dt=maxt/(nt-1);

ti=np.zeros(nt);te=np.zeros(nt);nu=np.zeros(nt);
ti[0]=ti0;
te[0]=te0;
lam=23.0-np.log((n/te[0]+n/ti[0])**0.5*(masse/massp+massi/massp)/(te[0]*massi/massp+ti[0]*masse/massp));
nu[0]=1.8e-19*(masse*massi)**0.5*n*lam/(masse*ti[0]+massi*te[0])**1.5;
print "lambda=",lam, ', tau (ns) =', 1/nu[0]*1e9;

for i in range(nt-1):
  ti[i+1]=ti[i]+dt*(te[i]-ti[i])*nu[i];
  te[i+1]=te[i]+dt*(ti[i]-te[i])*nu[i];
  nu[i+1]=1.8e-19*(masse*massi)**0.5*n*lam/(masse*ti[i]+massi*te[i])**1.5;
  lam=23.0-np.log((n/te[i]+n/ti[i])**0.5*(masse/massp+massi/massp)/(te[i]*massi/massp+ti[i]*masse/massp));
#  lam=24.0-np.log(n**0.5/(te[i]));

# binary
fname='/p/lscratche/holod1/test_collision/nanbu/nanbu_1dtest_b/fitted/history.p4'
#fname='/p/lscratche/holod1/test_collision/nanbu_equal_mass_b/history.p4'
#fname='/p/lscratche/holod1/test_collision/bb3050_nofield/history.p4'
#fname='/p/lscratche/holod1/test_collision/tony/bin/history.p4'
#fname='/p/lscratche/holod1/test_collision/nanbu/nanbu_1dtest_b_te120_ti100/history.p4'

(title,nProbes,Vars,Units,tt1,dat)=read_hist(fname)
x1=dat[:,0];
y11=dat[:,3]; # electron temp
y12=dat[:,4]; # ion temp

# jones
fname='/p/lscratche/holod1/test_collision/nanbu/nanbu_1dtest_j/3cells/history.p4'
#fname='/p/lscratche/holod1/test_collision/nanbu/nanbu_equal_mass_j/history.p4'
#fname='/p/lscratche/holod1/test_collision/jj3060_nofield/history.p4'
#fname='/p/lscratche/holod1/test_collision/tony/jones/history.p4'
#fname='/p/lscratche/holod1/test_collision/nanbu/nanbu_1dtest_j_te120_ti100/history.p4'
#fname='/p/lscratche/holod1/test_collision/nanbu/nanbu_1dtest_2b_te120_ti100/history.p4'


(title,nProbes,Vars,Units,tt1,dat)=read_hist(fname)
x2=dat[:,0];
y21=dat[:,3];
y22=dat[:,4];

f1, ax1 = plt.subplots(); f1, ax2 = plt.subplots()
ax1.plot(tt*1e9,te/te[0],'k',label='analitic electron (NRL)')
ax1.plot(x1,y11/y11[0],'b',label='LSP binary electron')
ax1.plot(x2,y21/y21[0],'r',label='LSP jones electron')
#plt.show();exit()



ax1.plot(tt*1e9,(ti[0]+te[0]-ti)/te[0],'--k',label='analytic ion (NRL) (Ti0+Te0-Ti)')
ax1.plot(x1,(y12[0]+y11[0]-y12)/y11[0],'--b',label='LSP binary ion (Ti0+Te0-Ti)')
ax1.plot(x2,(y22[0]+y21[0]-y22)/y21[0],'--r',label='LSP jones ion (Ti0+Te0-Ti)')
#ax1.get_xaxis().set_major_formatter(
#    matplotlib.ticker.FuncFormatter(lambda x, p: format(x, '.2e')))
ax1.set_xlim([0,maxt*1e9])

ax2.plot(tt*1e9,ti/te[0],'--k',label='analytic ion (NRL)')
ax2.plot(x1,y12/y11[0],'--b',label='LSP binary ion')
ax2.plot(x2,y22/y21[0],'--r',label='LSP jones ion')


ax1.set(xlabel='time (ns)', ylabel='T/T(0)')
legend = ax1.legend(loc='upper right', shadow=True)
ax2.set(xlabel='time (ns)', ylabel='T/T(0)')
legend = ax2.legend(loc='upper right', shadow=True)
ax2.set_xlim([0,maxt*1e9])


plt.show()

exit()


