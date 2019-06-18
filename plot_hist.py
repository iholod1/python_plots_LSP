#!/usr/bin/env python
"""script to plot from history.p4"""
__author__ = "Ihor Holod"
__credits__ = ["I. Holod", "D. Higginson", "A. Link"]
__email__ = "holod1@llnl.gov"
__version__ = "081117"

import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
if 'classic' in plt.style.available: plt.style.use('classic')
from read_xdr import read_hist


##### CONTROL PARAMETERS #####
class C(object):
	pass
arg=C()

parser = argparse.ArgumentParser(description=__doc__);
parser.add_argument('-p', type=int, action='append',help='probe');
parser.add_argument('-i', type=float, help='initial time');
parser.add_argument('-f', type=float, help='final time');
parser.add_argument('-l', help='directory location');
parser.add_argument('-s', action='store_true', help="save only");
parser.add_argument('-log', action='store_true', help="use log scale");
parser.add_argument('-diff', action='store_true', help="plot derivative");
parser.add_argument('-int', action='store_true', help="plot integated");
parser.add_argument('-cmin', type=float, action='append', help='min lim');
parser.add_argument('-cmax', type=float, action='append', help='max lim');
parser.add_argument('-smooth', action='store_true', help='smooth data');
parser.add_argument('-xfactor', type=float, help='x multiplier');
parser.add_argument('-yfactor', type=float, help='y multiplier');


parser.parse_args(namespace=arg)

probe = arg.p if arg.p != None else 1; 
if type(probe)!=list: probe=[probe];

t0 = float(arg.i) if arg.i != None else -1
t1 = float(arg.f) if arg.f != None else -1
dirname = str(arg.l) if arg.l != None else os.getcwd() + "/";
show_flag = not arg.s;
logscale = arg.log
deriv = arg.diff
integ = arg.int

smooth = False
if arg.smooth:
# 	from statsmodels.nonparametric.smoothers_lowess import lowess
# 	from tools.lowess import lowess 
	from tools.filter import butter_lowpass_filtfilt
	smooth = True
	
cmin = arg.cmin if arg.cmin != None else None
cmax = arg.cmax if arg.cmax != None else None

xfactor = float(arg.xfactor) if arg.xfactor != None else 1.0
yfactor = float(arg.yfactor) if arg.yfactor != None else 1.0

if not dirname[-1]=="/": dirname=dirname+"/";
fname=dirname+"history.p4"
print(fname);

# particle masses in positron units
#mass=1.0
mass=3.672e3 # D 
#mass=7.292e3 # He
#mass=1.825e+03; # neutron

# particle charges in positron units
#charge=-1.0;
charge=1.0;

charge=charge*1.609e-13; # converts charge to mcoulombs
#############################

(title,nProbes,Vars,Units,tt,dat)=read_hist(fname)
if show_flag: 
	for i in range(nProbes): print(str(i) + " " + Vars[i])


fig=[]; ax=[];

for ip, p in enumerate(probe):
	t=np.array(dat[:,0])*xfactor;
	y=np.array(dat[:,p])*yfactor;
	if ('extraction 1  charge' in Vars) and (p==Vars.index('extraction 1  charge')): 
		y=y/1.609e-13;
		Units[p]="#";


	indx=np.argwhere(t);

	if t1>0: indx=np.argwhere(t<=t1); t=t[indx[:,0]]; y=y[indx[:,0]]; 
	if t0>0: indx=np.argwhere(t>t0); t=t[indx[:,0]]; y=y[indx[:,0]];
	print("value at min t = %.4f " %(np.min(t)) + "ns: " + "%.4e" %(y[0])); 
	print("value at max t = %.4f " %(np.max(t)) + "ns: " + "%.4e" %(y[-1]));
	if y[0]!=0: print("change %.f%%" %(100*(y[-1]-y[0])/y[0]));
	print("averaged value %.2e" %(np.mean(y)));
	
	if smooth:
# 		y = lowess(y,t,is_sorted=True, frac=0.025)[:,1]
#		y = lowess(t,y,f=0.05)
		# y = butter_lowpass_filtfilt(y,10,5000)
		
		ti= np.linspace(min(t),max(t),len(t))
		yi = np.interp(ti,t,y)
	
		yi=np.diff(yi)/np.diff(ti);
		yi = butter_lowpass_filtfilt(yi,10,5000)
		yi[np.nonzero(yi<0)]=0
		y = np.cumsum(yi)*np.diff(ti)
		t=ti[0:-1];	
			



	if deriv: 
		y=np.diff(y)/np.diff(t);
		t=t[0:-1];
		Units[p]=Units[p]+"/ns"
		
	if integ: 
		y=np.cumsum((0.5*(y + np.roll(y,-1)))[0:-1]*np.diff(t));
		t=t[0:-1];
		Units[p]=Units[p]+"*ns"		

	fig.append([]);
	ax.append([]);

	fig[ip], ax[ip] = plt.subplots(figsize=(9,6))
	ax[ip].plot(t,y,linewidth=2)
	ttl =  Vars[p];
	ax[ip].set(title=ttl, xlabel='time ('+ Units[0].strip()+')', ylabel=Units[p])
	
	if logscale: ax[ip].set(yscale='log')

	ax[ip].set_xlim([min(t), max(t)]);
	
	if cmin: 
		ymin = cmin[ip]
	else:
		ymin = min(y)
		ymin = ymin*(1 - 0.05*np.sign(ymin))
	
	if cmax: 
		ymax = cmax[ip]
	else:
		ymax = max(y)
		ymax = ymax*(1 + 0.05*np.sign(ymax))
		
	ax[ip].set_ylim([ymin, ymax]);	
	
	#ax[ip].text(0.95*ax[ip].get_xlim()[0],1.1*ax[ip].get_ylim()[0],fname)

	if not os.path.exists(dirname+"plots"):  os.makedirs(dirname+"plots")
	fig[ip].savefig(dirname + "plots/" + "history_" + ttl.replace(" ", "_") + '.png',dpi=200, bbox_inches="tight")


	fout = open(dirname+ "plots/" + "probe"+"%d" %(p)+'_hist.out', 'w')
	fout.write("#\t" + Vars[p] + "\ttime\tns\tunits\t"+Units[p]+"\n");
	for i in range(len(t)):
		fout.write("%.8e %.8e" %(t[i], y[i]));
		fout.write("\n");
	fout.close()
	
	# irange=np.flatnonzero((y>=0.1*(np.max(y)-np.min(y)))&(y<=0.9*(np.max(y)-np.min(y))))
	# print("pulse lenght %.2e" %(t[irange[-1]]-t[irange[0]]))
	



if show_flag: plt.show();

# temporary stuff
#curr4=dat[indx[:,0],Vars.index('circuit 1 element 4 current')].flatten();
#volt4=dat[indx[:,0],Vars.index('circuit 1 element 4 voltage')].flatten();
#ne=dat[indx[:,0],Vars.index('net energy')].flatten();
#te=dat[indx[:,0],Vars.index('total energy')].flatten();

#print "integrated power (J)", np.sum(np.array([1.0e-9*1.0e3*(dat[i+1,0]-dat[i,0])*volt4[i]*curr4[i] for i in range(len(curr4)-1)]))
##print "FEG+PEG-PEL+DEDXL-FE-PE=", dat[-1,44]+dat[-1,45]-dat[-1,46]+dat[-1,47]-dat[-1,48]-dat[-1,49]
#print "NE=", ne[-1]
#print "TE=", te[-1]
#print "TE-NE=", te[-1]-ne[-1]


