# hist1d

import numpy as np

def hist1dlog(dat1,weight,nBin,*args,**kwargs):
    """generate weighted histogram data with log10 binning"""
    minVal=kwargs.get('minVal',None);
    maxVal=kwargs.get('maxVal',None);
    if not minVal: minVal = max(0.0001,min(dat1))
    if not maxVal: maxVal = max(minVal,max(dat1))
    ibin = 10**(np.linspace(np.log10(minVal),np.log10(maxVal),nBin+1))
    val, ibin = np.histogram(dat1, bins=ibin, weights=weight, density=False)

    ibin=map(lambda x, y: 0.5*(x + y), ibin[1:], ibin[:-1])

    return (ibin,val)

def hist1d(dat1,weight,nBin=30,minVal=None,maxVal=None,logScale=False):
    """generate weighted histogram data with linear binning"""
    if not minVal: minVal = min(dat1)
    if not maxVal: maxVal = max(dat1)
    
    if logScale:
        minVal = max(0.0001,minVal)
        maxVal = max(minVal,maxVal)
    
    if logScale:
        ibin = 10**(np.linspace(np.log10(minVal),np.log10(maxVal),nBin+1))
    else:
        ibin = np.linspace(minVal,maxVal,nBin+1)
        
    val, ibin = np.histogram(dat1, bins=ibin, weights=weight, density=False)

    ibin=map(lambda x, y: 0.5*(x + y), ibin[1:], ibin[:-1])

    return (list(ibin),val)
