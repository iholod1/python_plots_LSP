# energy analysis

import numpy as np

def kin_en(px,py,pz,mass):
    """returns kinetic energy in MeV"""
    ke=[];rmasse = 0.5110;
    for i in range(len(px)):
        ptot2 = px[i]**2+py[i]**2+pz[i]**2
        gamma = np.sqrt(1.0+ptot2)
        ke.append((gamma-1.0)*mass*rmasse)
    return(ke)

def energy(px,py,pz,mass,*args):
    """returns kinetic energy in MeV and momentum angle wrt to z-axis"""
#     ke=[];
#     ang=[];
    reslt = {}
    rmasse = 0.5110;
    clight = 2.99792458e10 #cm/s
    ptot2 = px**2 + py**2 + pz**2
    gamma = np.sqrt(1.0+ptot2)
    if 'ke' in args:
        reslt['ke'] = (gamma-1.0)*mass*rmasse
    if "ang" in args:
        reslt['ang'] = np.arccos(pz/np.sqrt(ptot2))
    if "vx" in args:
        reslt['vx'] = px/gamma*clight
    if "vy" in args:
        reslt['vy'] = py/gamma*clight
    if "vz" in args:
        reslt['vz'] = pz/gamma*clight
 
    return(reslt)

def theta_ang(px,py,pz):
    """returns momentum angle wrt to z-axis"""
    ang=[];
    for i in range(len(px)):
        ptot2 = px[i]**2+py[i]**2+pz[i]**2
        ang.append(np.arccos(pz[i]/np.sqrt(ptot2)))
    return(ang)

def velocity(px,py,pz):
    """returns velocity norm to c"""
    vx=[];vy=[];vz=[];
    for i in range(len(px)):
        ptot2 = px[i]**2+py[i]**2+pz[i]**2
        gamma = np.sqrt(1.0+ptot2)
        vx.append((px[i]/gamma))
        vy.append((py[i]/gamma))
        vz.append((pz[i]/gamma))
    return(vx,vy,vz)

def getMaxwellian(v,vt,v0):
    """returns Maxwellian distribution"""
    f = np.exp(-0.5*(v-v0)**2/vt**2)
    f = f/np.max(f)
    return f

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))
