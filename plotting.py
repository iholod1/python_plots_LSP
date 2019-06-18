def plot1d(x,y,xlbl,ylbl,**kwargs):
    """ 1D plot
        takes x,y,xlbl,ylbl as input
        extra arguments: xmin,xmax,ymin,ymax,title,fname
        extra arguments x1,y1
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    
    f, ax = plt.subplots(figsize=(4, 3), dpi=120)
    lbl0 = None
    lbl1 = None
   
    legend=[]
    
    if type(x)!=list:
        x=[x]
        y=[y]
    if "legend" in kwargs: 
        legend=kwargs["legend"]
        if type(legend)!=list:
            legend=[legend]
        if len(legend)!=len(x): 
            return()

    for ip in range(len(x)):
        if "legend" in kwargs: 
            label = legend[ip]
        else:
            label=None
            
        ax.plot(x[ip],y[ip],linewidth= 2,label=label)

#     ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 3))
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    # ax1.legend(loc='best', shadow=True)
    if "xmin" in kwargs:
        ax.set_xlim([kwargs["xmin"], ax.get_xlim()[-1]])
    if "xmax" in kwargs:
        ax.set_xlim([ax.get_xlim()[0],kwargs["xmax"]])
    if "ymin" in kwargs:
        ax.set_ylim([kwargs["ymin"], ax.get_ylim()[-1]])
    if "ymax" in kwargs:
        ax.set_ylim([ax.get_ylim()[0],kwargs["ymax"]])
    if "logy" in kwargs:
        ax.set_yscale("log")


    if len(legend)>0: ax.legend(loc='best', shadow=True)
    if "fname" in kwargs:
        fname = kwargs["fname"]+".png"
        f.savefig(fname,dpi=200,bbox_inches="tight")
    # plt.show()
    return(f,ax)