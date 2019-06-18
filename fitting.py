'''
Created on May 4, 2018

@author: holod1
'''
def linReg(x,y,**kwargs):
    import numpy as np
    
    i0 = 0
    i1 = len(x)
    
    if "x0" in kwargs:
        x0 = max(kwargs["x0"],np.min(x))
        i0=int(np.min(np.argwhere(x>=x0)))
    if "x1" in kwargs:
        x1 = min(kwargs["x1"],np.max(x))
        i1=int(np.max(np.argwhere(x<=x1)))+1        

    return(np.polyfit(x[i0:i1],y[i0:i1],1))