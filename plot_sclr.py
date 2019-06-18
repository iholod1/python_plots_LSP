#!/usr/bin/env python
"""Script to plot from sclr*.p4"""
__author__ = "Ihor Holod"
__credits__ = ["I. Holod", "D. Higginson", "A. Link"]
__email__ = "holod1@llnl.gov"
__version__ = "190529"

import sys

def plt_sclr(**kwargs):
    """Plot from sclr.p4 file"""
    import argparse
    import os.path
    import numpy as np
    import matplotlib
    from read_xdr import readXDRsclr, readXDRstruct
    import gc
    
    dirname=os.getcwd() + '/'
    if not os.path.exists(dirname+"plots"):
        os.makedirs(dirname+"plots")
    
    class C(object):
        pass
    arg=C()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', type=int, help='time index')
    parser.add_argument('-p', type=int, help='probe')
    parser.add_argument('-minx', type=float, help='minimum x')
    parser.add_argument('-maxx', type=float, help='maximum x')
    parser.add_argument('-minz', type=float, help='minimum z')
    parser.add_argument('-maxz', type=float, help='maximum z')
    parser.add_argument('-theta', type=float, action='append', help='azimuthal angle (range)')
    parser.add_argument('-polarz', type=float, action='append', help='polar projection at z (range)')
    
    parser.add_argument('-cmap', type=str, help='colormap')
    parser.add_argument('-cmin', type=float, help='min value')
    parser.add_argument('-cmax', type=float, help='max value')
    parser.add_argument('-slicex', type=float, action='append', help='z-slice along given x')
    parser.add_argument('-slicez', type=float, action='append', help='x-slice along given z')
    
    parser.add_argument('-f', action='store_true', help="add horizontal mirror image")
    parser.add_argument('-a', action='store_true', help="keep aspect ratio")
    parser.add_argument('-log', action='store_true', help="log scale")
    parser.add_argument('-smooth', action='store_true', help="smooth contour")
    parser.add_argument('-save', action='store_true', help="save only")
    parser.add_argument('-noaxis', action='store_true', help="show axis")
    parser.add_argument('-crit', type=float, help="use density criterion")
    
    
    if "arglist" in kwargs:
        print(kwargs["arglist"])
        parser.parse_args(kwargs["arglist"],namespace=arg)
    else:
        parser.parse_args(namespace=arg)    
    
    num = int(arg.i) if arg.i != None else 1
    ivar = int(arg.p) if arg.p != None else 0
    
    x0 = float(arg.minx) if arg.minx != None else None 
    x1 = float(arg.maxx) if arg.maxx != None else None 
    z0 = float(arg.minz) if arg.minz != None else None 
    z1 = float(arg.maxz) if arg.maxz != None else None
    
    if arg.theta != None:
        theta = arg.theta 
        if type(theta)!=list: 
            theta=[theta]
        theta = np.sort(theta)
    else:
        theta = None
    
    if arg.polarz != None:
        polarZ = arg.polarz 
        if type(polarZ)!=list: 
            polarZ=[polarZ]
        polarZ = np.sort(polarZ)
    else:
        polarZ = None
    
    cmap = arg.cmap  if arg.cmap != None else None 
    cmin = float(arg.cmin) if arg.cmin != None else None 
    cmax = float(arg.cmax) if arg.cmax != None else None 
    
    sliceX = arg.slicex if arg.slicex != None else None
    sliceZ = arg.slicez if arg.slicez != None else None  
    
    flip = arg.f
    aspect= arg.a
    logScale = arg.log
    smooth = arg.smooth
    noShow = arg.save
    noAxis = arg.noaxis
    
    crit = float(arg.crit) if arg.crit != None else None  
    
    if noShow: matplotlib.use('Agg') # to run in pdebug (not using $DISPLAY)
    import matplotlib.pyplot as plt
    if 'classic' in plt.style.available: plt.style.use('classic')
    import matplotlib.collections as mc
    from matplotlib.colors import LogNorm
    from matplotlib import ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pylab as pl
    
    def line2d(x0,x1,z0,z1,X,Z,dat,val):
        from scipy import interpolate
       
        if (z1==z0): return()
        
        indx=np.argwhere((X>=np.min([x0,x1]))&(X<=np.max([x0,x1])))
        indx=indx.reshape(indx.shape[0],1)
        indz=np.argwhere((Z>=z0)&(Z<=z1))
        indz=indz.reshape(1,indz.shape[0])
        
        tanA=(x1-x0)/(z1-z0)
        # cosA = np.cos(np.arctan(tanA))
        npt=100
        z = np.linspace(z0,z1,npt)
        x = x0 + (z-z0)*tanA
        r = np.sqrt((x-x0)**2 + (z-z0)**2).reshape((npt,1))
        f=interpolate.interp2d(X[indx].flatten(), Z[indz].flatten(), dat[indx,indz].transpose(), kind='cubic')
        lnout = np.zeros((npt,1))
        for i in range(npt):
            lnout[i]=f(x[i], z[i])
            
        f, ax = plt.subplots()
        # ax.plot(z[1:],np.diff(lnout[:,0])/np.diff(r[:,0]),linewidth=2)
        ax.plot(z,lnout,linewidth=2)
        i = np.nonzero(lnout[::-1]>val)[0][0]        
        print(x[::-1][i])
        print(z[::-1][i])
        plt.show()
        return(lnout)
        
    
    def plotSliceZ():
        data=np.zeros((len(X),1));
        if len(sliceZ)>1:
            zrange = np.linspace(sliceZ[0],sliceZ[1],100)
        else:
            zrange = sliceZ
        
        for i in range(len(X)):
            data[i,0]=np.mean(np.interp(zrange,Z[:],dat[i,:]));
        f, ax = plt.subplots()
        ax.plot(X[:],data[:,0],linewidth=2)
        ax.plot(X[:],np.ones(len(data[:,0]))*np.mean(data[:,0]),linewidth=2)
        print("SliceZ mean %.2e" %(np.mean(data[:,0])))
        ax.set(title=VarNames[ivar]+" (" + \
            VarUnits[ivar] + ")" +" X-slice at Z = %.2f" %(sliceZ[0]) + " at "+tstamp+"ns")
        ax.set_xlim([xmin,xmax]);
        ax.set_ylim([cmin,cmax]);
        ax.set(xlabel='X (cm)', ylabel=VarUnits[ivar])
        if logScale: ax.set(yscale='log')
        f.savefig(dirname +"plots/" + VarNames[ivar] + "_%06.1f" %(time) + '_sliceZ.png',dpi=200, bbox_inches="tight")
    #     fout = open(dirname +"plots/slicez.out", "a+")
    #     fout.write("%.8e %.8e\n" %(time, np.mean(data[0,:])));
    #     fout.close()      
        fout = open(dirname +"plots/" + VarNames[ivar] + "_%06.1f" %(time) + '_sliceZ.out', "w")
        for i in range(len(X)):
            fout.write("%.8e %.8e\n" %(X[i],data[i,0]))
        fout.close()
        if not noShow: 
            plt.show()
    # # sheath front locator
    #     zdum=Z[0,:]
    #     ydum=np.diff(data[0,:]);
    #     zfront=zdum[np.argwhere(ydum==np.min(ydum))][0]
    #     print("sheath front z-location %.2f" %(zfront))
    #     fout = open(dirname+'sheath_front.out', "a")
    #     fout.write("%.8e %.8e" %(time, zfront));
    #     fout.write("\n");
    #     fout.close()         
        return()
    
    def plotSliceX():
        data=np.zeros((1,len(Z)));
        if len(sliceX)>1:
            xrange = np.linspace(sliceX[0],sliceX[1],100)
        else:
            xrange = sliceX
                
        for j in range(len(Z)):
            data[0,j]=np.mean(np.interp(xrange,X[:],dat[:,j]));
        f, ax = plt.subplots()
        ax.plot(Z[:],data[0,:],linewidth=2)
        ax.plot(Z[:],np.ones(len(data[0,:]))*np.mean(data[0,:]),linewidth=2)
        print("SliceX integral %.2e" %(np.trapz(data[0,:],Z)))
        ax.set(title=VarNames[ivar]+" (" + \
            VarUnits[ivar] + ")" +" Z-slice at X = %.4f" %(sliceX[0]) + " at "+tstamp+"ns")
        ax.set_xlim([zmin,zmax]);
        ax.set_ylim([cmin,cmax]);
        ax.set(xlabel='Z (cm)', ylabel=VarUnits[ivar])
        if logScale: ax.set(yscale='log')
        f.savefig(dirname +"plots/" + VarNames[ivar] + "_%06.1f" %(time) + '_sliceX.png',dpi=200, bbox_inches="tight")
        fout = open(dirname +"plots/slicex.out", "a+")
        fout.write("%.8e %.8e\n" %(time, np.mean(data[0,:])));
        fout.close()
        fout = open(dirname +"plots/" + VarNames[ivar] + "_%06.1f" %(time) + '_sliceX.out', "w")
        for i in range(len(Z)):
            fout.write("%.8e %.8e\n" %(Z[i],data[0,i]))
        fout.close()

        ind = np.nonzero(data[0,::-1]>2e18)[0][0]
        fout = open(dirname +"plots/front.out", "a+")
        fout.write("%.8e %.8e\n" %(time, Z[::-1][ind]));
        fout.close()        
        
        if not noShow: 
            plt.show()
        return()
    
    def addStructure(ax):
        (xa,ya,za,xb,yb,zb)=readXDRstruct(dirname+'struct.p4')
        lines=[[(za[i],xa[i]),(zb[i],xb[i])] for i in range(len(xa))]
        lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
        ax.add_collection(lc)
        if flip:
            lines=[[(za[i],-xa[i]),(zb[i],-xb[i])] for i in range(len(xa))]
            lc = mc.LineCollection(lines, color=(0.9,0.9,0.9),linewidths=1)
            ax.add_collection(lc)
        return()
    
    def plotPolar():
        r, th = np.meshgrid(X, Y)
        f, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(8,8))
        cm = matplotlib.cm.get_cmap(cmap)
        val = dat.transpose(1,0)
    #     contour_levels = arange(cmin, 3, 0.05)
        if logScale:
    #         im = ax.pcolor(th, r, dat.transpose(1,0),norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)
            im = ax.contourf(th, r, val,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)
        else:
    #         im = ax.pcolor(th, r, dat.transpose(1,0),vmin=cmin,vmax=cmax,cmap=cm)
            im = ax.contourf(th, r, val,100,vmin=cmin,vmax=cmax,cmap=cm)
        ax.xaxis.grid(True, zorder=0)
        ax.yaxis.grid(True, zorder=0)
               
            
        if noAxis:
            ax.set(title=VarNames[ivar])     
            ax.get_xaxis().set_visible(True)
            ax.get_yaxis().set_visible(False)
        else:
            title = VarNames[ivar] +  " (" + VarUnits[ivar] + ")" +"/Time=%.1fns/Z=%.1fcm" %(time,polarZ[0])
            ax.set(title=title, xlabel="R (cm)")
            plt.colorbar(im)
            
            
        f.savefig(dirname + "plots/" + VarNames[ivar] + "_T%06.1f_Z%04.1f.png" %(time,polarZ[0]),dpi=200, bbox_inches="tight")
        if not noShow: 
            plt.show()
        return()
    
    
    
    #######################################################
    
    fname=dirname+'sclr'+str(num)+'.p4'
    print(fname)
    
    if "sdata" in kwargs:
        (X,Y,Z,Var,VarNames,VarUnits,time)=kwargs["sdata"]
    else:
        (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRsclr(fname,silent=noShow)
        
    # if not noShow:
    #     for i in range(len(Var)):
    #         print(str(i) + " " + VarNames[i] + " " + VarUnits[i])
    tstamp =  "%.2f" % time
    
    dat=Var[ivar]
    if crit!=None:
        cvar = VarNames.index("RhoT" + VarNames[ivar][4:])
        dat[np.nonzero(Var[cvar]<crit)]=0.0
    
    if flip: x0=min(X)
    xmin=x0 if (x0!=None) and (x0>min(X)) and (x0<=max(X)) else min(X)
    xmax=x1 if (x1!=None) and (x1>xmin) and (x1<=max(X)) else max(X)
    zmin=z0 if (z0!=None) and (z0>min(Z)) and (z0<max(Z)) else min(Z)
    zmax=z1 if (z1!=None) and (z1>zmin) and (z1<=max(Z)) else max(Z)
    
    indx=np.argwhere((X>=xmin)&(X<=xmax))
    indx=indx.reshape(indx.shape[0],1,1)
    indz=np.argwhere((Z>=zmin)&(Z<=zmax))
    indz=indz.reshape(1,1,indz.shape[0])
    indy=np.arange(len(Y)).reshape(1,len(Y),1)
    
    #indz=np.transpose(indz)
    X=X[indx].flatten()
    Y=Y[indy].flatten()
    Z=Z[indz].flatten()
    
    dat=dat[indx,indy,indz]
    
    if type(polarZ)!=type(None):
        # calculate weights for nonuniform grid spacing
        dz = np.diff(Z)
        wZ = np.ones(len(Z))
        wZ[0] = 0.5*dz[0]
        wZ[1:-1]=0.5*(dz + np.roll(dz,-1))[0:-1]
        wZ[-1]=0.5*dz[-1]
    
        zdum = polarZ[0]
        k10 = np.max([1,int(np.argwhere(Z>=zdum)[0][0])])
        k00 = int(k10-1)
        w00 = (Z[k10] - zdum)/(Z[k10] - Z[k00])
        wZ[k00] *= w00
        
        zdum = polarZ[-1]
        k11 = np.max([1,int(np.argwhere(Z>=zdum)[0][0])])
        k01 = int(k11-1)
        w01 = (Z[k11] - zdum)/(Z[k11] - Z[k01])
        wZ[k11] *= (1-w01)
        dat = np.average(dat[:,:,k00:k11+1], axis=2, weights=wZ[k00:k11+1])
    
        plotPolar()
        exit() 
    
    if type(theta)!=type(None):
        wY = np.ones(len(Y))
        # calculate weights for nonuniform grid spacing
        if len(Y)>1:
            dy = np.diff(Y)
            wY[0] = 0.5*dy[0]
            wY[1:-1]=0.5*(dy + np.roll(dy,-1))[0:-1]
            wY[-1]=0.5*dy[-1]    
        
        thdum = np.radians(theta[0])
        thdum = divmod(thdum,np.pi*2)[1] if thdum>np.pi*2 else thdum
        j10 = int(np.argwhere(Y>=thdum)[0][0])
        if j10==0: 
            j10+=1
        j00 = int(j10-1)
        w00 = (Y[j10] - thdum)/(Y[j10] - Y[j00])
        wY[j00] = w00*wY[j00] # weight of outermost left grid point
        
        thdum = np.radians(theta[-1])
        thdum = divmod(thdum,np.pi*2)[1] if thdum>np.pi*2 else thdum
        j11 = int(np.argwhere(Y>=thdum)[0][0])
        if j11==0: 
            j11+=1    
        j01 = int(j11-1)
        w01 = (Y[j11] - thdum)/(Y[j11] - Y[j01])
        wY[j11] = (1 - w01)*wY[j11] # # weight of outermost right grid point
        
        dat = np.average(dat[:,j00:j11+1,:], axis=1, weights=wY[j00:j11+1])
    else:
        dat=np.squeeze(dat[:,0,:])
    
    if cmin is None:
        cmin = dat.min()
    if cmax is None:
        cmax = dat.max()
    
    fig, ax = plt.subplots(figsize=(8,8))
    if flip:
        dat=np.concatenate((np.flipud(dat),dat),axis=0);
        X = np.concatenate([-X[::-1],X])
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    if logScale:
        locator = ticker.LogLocator(base=10)
        if smooth:
            dat[np.nonzero(dat>cmax)] = cmax
            dat[np.nonzero(dat<cmin)] = cmin
            levels = np.power(10,np.linspace(np.log10(cmin),np.log10(cmax),num=100))
            im = ax.contourf(Z,X,dat,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm,locator=locator,levels=levels)
        else:
            im = ax.pcolor(Z,X,dat,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)
    else:
        if smooth:
            dat[np.nonzero(dat>cmax)] = cmax
            dat[np.nonzero(dat<cmin)] = cmin        
            levels = ticker.MaxNLocator(nbins=100).tick_values(cmin, cmax)
            im = ax.contourf(Z,X,dat,vmin=cmin,vmax=cmax,cmap=cm,levels=levels)
        else:
            im = ax.pcolor(Z,X,dat,vmin=cmin,vmax=cmax,cmap=cm)
        
    ax.set_xlim([min(Z[:]),max(Z[:])])
    ax.set_ylim([min(X[:]),max(X[:])])
    
    title = VarNames[ivar] + " (" + VarUnits[ivar] + ")" + "/Time %.1fns" %(time)
    if type(theta)!=type(None):
        title = title + "/Angle %.1f-%.1fdeg" %(theta[0],theta[-1])
    ax.set(title=title,xlabel="Z (cm)",ylabel="R (cm)")
    
    # add structure
    if os.path.isfile(dirname+'struct.p4'):
        addStructure(ax)
        
    if (aspect or flip):
        ax.set_aspect('equal');
    else:
        ax.set_aspect('auto')    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if not logScale: #set exponent base for colorbar formatter
        CB = plt.colorbar(im, cax=cax)
        CB.formatter.set_powerlimits((0, 2))
        CB.update_ticks()
    else:
        CB = plt.colorbar(im, cax=cax, ticks=locator)
    
    if noAxis:
        ax.set(title=VarNames[ivar],
           xlabel="Z (cm)", ylabel="R (cm)")     
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    ftitle = VarNames[ivar] + "_T%07.1f" %(time)
    if type(theta)!=type(None):
        ftitle = ftitle + "_A%5.1f-%5.1f" %(theta[0],theta[-1])
    fig.savefig(dirname + "plots/" + ftitle + '.png',dpi=200, bbox_inches="tight")
    #cax.yaxis.major.locator.set_params(nbins=8) 
    
    if sliceX!= None and sliceX[-1]<=xmax and sliceX[0]>=xmin: # plot slice along Z
        plotSliceX()
    
    if sliceZ!= None and sliceZ[-1]<=zmax and sliceZ[0]>=zmin: # plot slice along X
        plotSliceZ()
        
    # line2d(0,2,0,8,X,Z,dat,1.5e18)
            
    if not noShow: 
        plt.show()
    
    return()

if __name__=="__main__":
    plt_sclr(arglist=sys.argv[1:])
    exit()

# if ivar==1 and False:
#     Phi=0.0;
#     for i in xrange(X.size-1):
#         dX=0.01*(X[i+1]-X[i]); # m
#         Xc=0.5*0.01*(X[i+1]+X[i]); # m
#         for j in xrange(Z.size-1):
#             dZ=0.01*(Z[0,j+1]-Z[0,j]); # m
#             rBy=np.mean([dat[i,j],dat[i+1,j],dat[i,j+1],dat[i+1,j+1]]); # amp
#             By=2.0*1e-7*rBy/Xc; # T
#             Phi+=By*dX*dZ; # Wb
#     print("Magnetic flux (Wb) = %.2e" %(Phi))