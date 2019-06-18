#!/usr/bin/env python
# script to plot from flds*.p4
# I. Holod, 190529

import sys

def plt_flds(**kwargs):
    """Plot from flds.p4 file"""
    import argparse
    import os.path
    import numpy as np
    
    import matplotlib
    
    from read_xdr import readXDRflds
    from read_xdr import readXDRstruct
    
    dirname=os.getcwd() + '/'
    if not os.path.exists(dirname+"plots"):
        os.makedirs(dirname+"plots")
    
    class C(object):
        pass
    arg=C()
    parser = argparse.ArgumentParser(description='Process arguments');
    parser.add_argument('-i', type=int, help='time index');
    parser.add_argument('-p', type=int, help='probe');
    parser.add_argument('-x', action='store_true',help='x-component');
    parser.add_argument('-y', action='store_true',help='y-component');
    parser.add_argument('-z', action='store_true',help='z-component');
    
    parser.add_argument('-f', action='store_true',help='add mirrored image');
    parser.add_argument('-a', action='store_true', help="keep aspect ratio");
    
    parser.add_argument('-log', action='store_true', help="log scale");
    parser.add_argument('-minx', type=float, help='minimum x');
    parser.add_argument('-maxx', type=float, help='maximum x');
    parser.add_argument('-minz', type=float, help='minimum z');
    parser.add_argument('-maxz', type=float, help='maximum z');
    parser.add_argument('-theta', type=float, action='append', help='azimuthal angle')
    parser.add_argument('-polarz', type=float, help='polar projection at z')
    parser.add_argument('-cmap', type=str, help='colormap');
    parser.add_argument('-cmin', type=float, help='min value');
    parser.add_argument('-cmax', type=float, help='max value');
    parser.add_argument('-slicex', type=float, action='append', help='z-slice along given x');
    parser.add_argument('-slicez', type=float, action='append', help='x-slice along given z');
    parser.add_argument('-save', action='store_true', help="save only");
    parser.add_argument('-noaxis', action='store_true', help="show axis");
    
    if "arglist" in kwargs:
        print(kwargs["arglist"])
        parser.parse_args(kwargs["arglist"],namespace=arg)
    else:
        parser.parse_args(namespace=arg)   
    
    num = int(arg.i) if arg.i != None else 1;
    ivar = int(arg.p) if arg.p != None else 0;
    
    xc = 1 if arg.x else 0
    yc = 1 if arg.y else 0
    zc = 1 if arg.z else 0
    
    x0 = float(arg.minx)  if arg.minx != None else None 
    x1 = float(arg.maxx)  if arg.maxx != None else None 
    z0 = float(arg.minz)  if arg.minz != None else None 
    z1 = float(arg.maxz)  if arg.maxz != None else None 
    
    if arg.theta != None:
        theta = arg.theta 
        if type(theta)!=list: 
            theta=[theta]
        theta = np.sort(theta)
    else:
        theta = None
        
    polarZ = float(arg.polarz) if arg.polarz != None else None
    
    cmap = arg.cmap  if arg.cmap != None else None 
    cmin = float(arg.cmin)  if arg.cmin != None else None 
    cmax = float(arg.cmax)  if arg.cmax != None else None
    
    sliceX = arg.slicex  if arg.slicex != None else None
    sliceZ = arg.slicez  if arg.slicez != None else None    
    
    flip = arg.f;
    aspect= arg.a;
    logScale = arg.log;
    noShow = arg.save;
    noAxis = arg.noaxis
    
    if noShow: matplotlib.use('Agg') # to run in pdebug (not using $DISPLAY)
    import matplotlib.pyplot as plt
    if 'classic' in plt.style.available: plt.style.use('classic')
    import matplotlib.collections as mc
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pylab as pl
    
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
        ax.set(title=VarNames[ivar] + "("+str(xc)+","+str(yc)+","+str(zc)+")" +" (" + \
            VarUnits[ivar] + ")" +" Z-slice at X = %.4f" %(sliceX[0]) + " at "+tstamp+"ns")
        ax.set_xlim([zmin,zmax]);
        ax.set_ylim([cmin,cmax]);
        ax.set(xlabel='Z (cm)', ylabel=VarUnits[ivar])
        if logScale: ax.set(yscale='log')
        f.savefig(dirname +"plots/" + VarNames[ivar] + "x"*xc + "y"*yc + "z"*zc + \
                  "_%06.1f" %(time) + '_sliceX.png',dpi=200, bbox_inches="tight")
        fout = open(dirname +"plots/slicex.out", "a+")
        fout.write("%.8e %.8e\n" %(time, np.mean(data[0,:])));
        fout.close()        
        if not noShow: 
            plt.show()
        return()
    
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
    #     ax3.plot(X[:,0],np.ones(len(data3[:,0]))*np.mean(data3[:,0]),linewidth=2)
        ax.set(title=VarNames[ivar] + "("+str(xc)+","+str(yc)+","+str(zc)+")"  +" (" + \
            VarUnits[ivar] + ")" +" X-slice at Z = %.2f" %(sliceZ[0]) + " at "+tstamp+"ns")
        ax.set_xlim([xmin,xmax]);
        ax.set_ylim([cmin,cmax]);
        ax.set(xlabel='X (cm)', ylabel=VarUnits[ivar])
        if logScale: ax.set(yscale='log')
        f.savefig(dirname +"plots/" + VarNames[ivar] + "x"*xc + "y"*yc + "z"*zc + \
                  "_%06.1f" %(time) + '_sliceZ.png',dpi=200, bbox_inches="tight")
        fout = open(dirname +"plots/slicez.out", "a+")
        fout.write("%.8e %.8e\n" %(time, np.mean(data[0,:])));
        fout.close()      
        if not noShow: 
            plt.show()
    
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
        if logScale:
    #         im = ax.pcolor(th, r, dat.transpose(1,0),norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)
            im = ax.contourf(th, r, dat.transpose(1,0),norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)        
        else:
    #         im = ax.pcolor(th, r, dat.transpose(1,0),vmin=cmin,vmax=cmax,cmap=cm)
            im = ax.contourf(th, r, dat.transpose(1,0),100,vmin=cmin,vmax=cmax,cmap=cm)        
        ax.xaxis.grid(True, zorder=0)
        ax.yaxis.grid(True, zorder=0)
    
        if noAxis:
            ax.set(title=VarNames[ivar] + "("+str(xc)+","+str(yc)+","+str(zc)+")")     
            ax.get_xaxis().set_visible(True)
            ax.get_yaxis().set_visible(False)
        else:
            title = VarNames[ivar] + "("+str(xc)+","+str(yc)+","+str(zc)+")" + " (" + VarUnits[ivar] + ")" +"/Time=%.1fns/Z=%.1fcm" %(time,polarZ)       
            ax.set(title=title, xlabel="R (cm)")
            plt.colorbar(im)
        
        f.savefig(dirname + "plots/" + VarNames[ivar] + "x"*xc + "y"*yc + "z"*zc + "_T%06.1f_Z%04.1f.png" %(time,polarZ),dpi=200, bbox_inches="tight")
        if not noShow: 
            plt.show()
        return()
    
    def flowPlot():
        plt.figure()
        ax = plt.gca()
        scale = 1*max([max(datx.flatten()),max(datz.flatten())])
        ax.quiver(Z[0::10], X[0::10], datz[0::10,0::10], datx[0::10,0::10],scale=scale)
        return(ax)
    #     im = ax.pcolor(Z,X,dat,vmin=cmin,vmax=cmax,cmap=cm)
    #     
    #     ax.quiver(Z, X, U, V, angles='xy', scale_units='xy', scale=1)
    #     ax.set_xlim([-1, 10])
    #     ax.set_ylim([-1, 10])
    #     plt.draw()
    #     plt.show()
    
    
    fname=dirname+'flds'+str(num)+'.p4'
    print(fname)
    
    if "fdata" in kwargs:
        (X,Y,Z,Var,VarNames,VarUnits,time)=kwargs["fdata"]
    else:
        (X,Y,Z,Var,VarNames,VarUnits,time)=readXDRflds(fname,silent=noShow)
    
    # if not noShow:
    #     for i in range(len(Var)):
    #         print(str(i) + " " + VarNames[i] + " " + VarUnits[i]);
    
    tstamp =  "%.2f" % time
    
    if xc+yc+zc>1:
        dat=np.sqrt(xc*np.square(Var[ivar,0,:,:,:])+yc*np.square(Var[ivar,1,:,:,:])+zc*np.square(Var[ivar,2,:,:,:]))
    else:
        dat=xc*Var[ivar,0,:,:,:]+yc*Var[ivar,1,:,:,:]+zc*Var[ivar,2,:,:,:]
        
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
    
    if False:
        # plot quiver
        datx = np.squeeze(Var[ivar,0,indx,indy,indz])
        datz = np.squeeze(Var[ivar,2,indx,indy,indz])
        ax = flowPlot()
        addStructure(ax)
        plt.show()
        exit()
    
    
    if polarZ!=None:
        k1 = int(np.argwhere(Z>=polarZ)[0][0])
        k0 = int(k1-1)
        w0 = (Z[k1] - polarZ)/(Z[k1] - Z[k0])
        w1 = 1 - w0
        dat = w0*dat[:,:,k0] + w1*dat[:,:,k1]
        plotPolar()
        exit() 
    
    if type(theta)!=type(None):
        wY = np.ones(len(Y)) # initial weights of grid points (needs to take into account nonuniform spacing
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
    
    
    if flip:
        dat=np.concatenate((np.flipud(dat),dat),axis=0);
        X = np.concatenate([-X[::-1],X])
    #  xmin=-xmax
    
    fig, ax = pl.subplots(figsize=(8,8))
    #im = ax.imshow(dat, interpolation='none',origin='lower', extent=(zmin, zmax, xmin, xmax))
    #CB = plt.colorbar(im) 
    #im = ax.pcolormesh(Z,X,dat)
    cm = matplotlib.cm.get_cmap(cmap)
    
    if logScale:
        im = ax.pcolor(Z,X,dat,norm=LogNorm(vmin=cmin,vmax=cmax),cmap=cm)
    else:
        im = ax.pcolor(Z,X,dat,vmin=cmin,vmax=cmax,cmap=cm)
    
    ax.set_xlim([min(Z[:]),max(Z[:])])
    ax.set_ylim([min(X[:]),max(X[:])])
    
    title = VarNames[ivar] + "(" + str(xc) + "," + str(yc) + "," + str(zc) + ")" + " (" + VarUnits[ivar] + ")" + "/Time %.1fns" %(time)
    if type(theta)!=type(None):
        title = title + "/Angle %.1f-%.1fdeg" %(theta[0],theta[1])       
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
    CB =plt.colorbar(im, cax=cax)
    
    if noAxis:
        ax.set(title=VarNames[ivar],
           xlabel="Z (cm)", ylabel="R (cm)")     
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
    
    ftitle = VarNames[ivar] + "x"*xc + "y"*yc + "z"*zc + "_T%06.1f" %(time)
    if type(theta)!=type(None):
        ftitle = ftitle + "_A%05.1f-%05.1f" %(theta[0],theta[1]) 
    fig.savefig(dirname + "plots/" + ftitle + '.png',dpi=200, bbox_inches="tight")
    
    if sliceX!= None and sliceX[-1]<=xmax and sliceX[0]>=xmin: # plot slice along Z
        plotSliceX()
    
    if sliceZ!= None and sliceZ[-1]<=zmax and sliceZ[0]>=zmin: # plot slice along X
        plotSliceZ()
    
    # file name containing time step number
    #fig.savefig(dirname + "plots/" + VarNames[ivar] + '_'+ str(num) + '.png',dpi=200)
    # file name containing time step number with fixed format
    #fig.savefig(dirname + "plots/" + VarNames[ivar] + '_'+ "%07d" %(num) + '.png',dpi=200)
    # file name containing time step with fixed format
    
    
    if not noShow: 
        plt.show();
    
    
    ###
if __name__=="__main__":
    plt_flds(arglist=sys.argv[1:])
    exit()    

