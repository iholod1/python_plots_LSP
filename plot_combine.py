#!/usr/bin/env python
# script to plot from several files
# input form pots.in
# I. Holod, 04/05/17

import os.path
import argparse
import numpy as np


import matplotlib.pyplot as plt
if 'classic' in plt.style.available: plt.style.use('classic')
import matplotlib.colors as colors
import matplotlib.cm as cmx

dirname=os.getcwd() + '/';

class C(object):
    pass
arg=C()

parser = argparse.ArgumentParser(description=__doc__);
parser.add_argument('-xlog', action='store_true', help="use log x-scale");
parser.add_argument('-ylog', action='store_true', help="use log y-scale");
parser.add_argument('-scatter', action='store_true', help="scatter plot");
parser.add_argument('-legend', type=str, help="legend location");
parser.add_argument('-yfactor', type=float, help='y multiplier');
parser.add_argument('-infile', type=str, help="input file name")

parser.parse_args(namespace=arg)

xlog = arg.xlog
ylog = arg.ylog
scatter = arg.scatter
lloc = arg.legend if arg.legend!=None else 'b'

yfactor = float(arg.yfactor) if arg.yfactor != None else 1.0



rerun = False;
fid = dirname + arg.infile if arg.infile!=None else dirname + "plots.in" 


headr=[];fname=[];x0=[];y0=[];x1=[];xoffset=[];y1=[];xlbl=[];ylbl=[];leg=[];
with open(fid) as infile:
    for line in infile:
        if line[0]=="#": headr.append(line.split("\t"));
        elif len(line)>1: 
            fname.append(line.split(",")[0]);
            x0.append(float(line.split(",")[1]));
            x1.append(float(line.split(",")[2]));
            xoffset.append(float(line.split(",")[3]));
            y0.append(float(line.split(",")[4]));
            y1.append(float(line.split(",")[5]));
            xlbl.append(line.split(",")[6]);
            ylbl.append(line.split(",")[7]);
            leg.append(line.split(",")[8][0:-1]);
        else: 
            continue;

nplt=len(fname);

x=[];y=[];
for i in range(len(fname)):
    fid = fname[i];
    print(fid)
    if rerun: os.system("plot_hist.py -p 51 -i 3200 -s 0 -l '/p/lscratchh/holod1/mj/5torr_Mather/driver5/kin_b'");
    headr=[];tt=[];dat=[];
    with open(fid) as infile:
        for line in infile:
            if line[0]=="#": headr.append(line.split("\t"));
            else: 
                tt.append(line.split()[0]);
                dat.append(line.split()[1:]);
    x.append(np.array(tt).astype(float));
    y.append(np.array(dat).astype(float)*yfactor);


from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

f = plt.figure(figsize=(9,6))

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

if nplt>1:
    par=[]
    for i in range(nplt-1):
        par.append(host.twinx());
    offset = 10
    new_fixed_axis = par[-1].get_grid_helper().new_fixed_axis

#  par[-1].axis["right"] = new_fixed_axis(loc="right", axes=par[-1],offset=(offset, 0))
#  par[-1].axis["right"].toggle(all=True)

host.set_xlim(x0[0],x1[0]);
host.set_ylim(y0[0],y1[0]);
host.set_xlabel(xlbl[0])
host.set_ylabel(ylbl[0])
if xlog: host.set_xscale("log")
if ylog: host.set_yscale("log")

values=range(nplt)
cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

for i in range(nplt-1):
#  par[i].set_ylabel(ylbl[i+1]); # set labels at the RHS
    par[i].set_ylim(y0[i+1],y1[i+1]);
    if xlog: par[i].set_xscale("log")
    if ylog: par[i].set_yscale("log")



colorVal = scalarMap.to_rgba(values[0])
if scatter:
    linestyle = "None"
    marker = "."
else:
    linestyle="-"
    marker = "None"
    
p=[];
p.append(host.plot(x[0]+xoffset[0],y[0],linewidth=2,label=leg[0],color=colorVal,linestyle=linestyle,marker=marker));
for i in range(nplt-1):
    colorVal = scalarMap.to_rgba(values[i+1])
    p.append(par[i].plot(x[i+1]+xoffset[i+1],y[i+1],linewidth=2,label=leg[i+1],color=colorVal,linestyle=linestyle,marker=marker))

host.axis["left"].label.set_color(p[0][0].get_color())

for i in range(nplt-1):
    par[i].axis["right"].label.set_color(p[i+1][0].get_color())

loc={'r':'right','cr':'center right','ur':'upper right','lr':'lower right',
     'l':'left','cl':'center left','ul':'upper left','ll':'lower left',
     'c':'center','b':'best'}

if lloc in loc.keys():
    host.legend(loc=loc[lloc], shadow=True)
else:
    host.legend(loc='best', shadow=True)

if False: #calculate the average
    Nx=100
    xmin=np.max([np.min(x[i][:]) for i in range(nplt)]);
    xmax=np.min([np.max(x[i][:]) for i in range(nplt)]);
    newx=np.logspace(np.log10(xmin),np.log10(xmax),Nx).reshape((Nx,1));
    newy=np.zeros((Nx,1));
    erry=np.zeros((Nx,1));
    for i in range(Nx):
        dum=[np.interp(newx[i],x[j][:],y[j][:,0]) for j in range(nplt)];
        newy[i,0]=np.mean(dum);
        erry[i,0]=np.std(dum);
    f3, ax3 = plt.subplots()
    ax3.plot(newx.flatten(),newy,linewidth=2,color = 'k')
    ax3.fill_between(newx.flatten(), (newy-erry).flatten(), (newy+erry).flatten())
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    ax3.set_xlim(x0[0],x1[0]);
    ax3.set_ylim(y0[0],y1[0]);
    ax3.set_xlabel(xlbl[0])
    ax3.set_ylabel(ylbl[0])
    f3.savefig(dirname + "aevraged_plot.png",dpi=200)

    fout = open(dirname+'averaged.out', 'w')
    fout.write("#\n");
    for i in range(len(newx)):
        fout.write("%.8e %.8e" %(newx[i], newy[i]));
        fout.write("\n");
    fout.close()
    fout = open(dirname+'std.out', 'w')
    fout.write("#\n");
    for i in range(len(newx)):
        fout.write("%.8e %.8e" %(newx[i], erry[i]));
        fout.write("\n");
    fout.close()

f.savefig(dirname + "combined_plot.png",dpi=200, bbox_inches="tight")
plt.show()
exit()
