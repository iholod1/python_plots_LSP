#!/usr/bin/env python

from sys import exit
import math
import numpy as np
import scipy as sp

r1=3.0
r2=5.0
eps1=5.6

z=138.*np.log10(r2/r1)/np.sqrt(eps1);

print 'impedance (Ohm) = ', z
print 'phase velocity = ', 1./np.sqrt(eps1)

