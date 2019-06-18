#!/usr/bin/env python
# generates probe list
import os, sys, getopt
import numpy as np


z0=3.6
z1=4.2
r0=0.05
r1=0.1

nZ=int((z1-z0)/0.01) + 1
nR=int((r1-r0)/0.02) + 1

rR=np.linspace(r0,r1,nR);
rZ=np.linspace(z0,z1,nZ);


fout = open('probes.out', 'w')
p=1

fout.write("probe" + "%d" %(p) +";\n");
fout.write('energy field_energy \n');
fout.write('; \n');
p+=1;

fout.write("probe" + "%d" %(p) +";\n");
fout.write('energy particle_energy \n');
fout.write('; \n');
p+=1;

fout.write("probe" + "%d" %(p) +";\n");
fout.write('extraction 1 charge \n');
fout.write('; \n');
p+=1;

for i in range(nR):
  for j in range(nZ):
# ion density
    fout.write('probe' + '%d' %(p) +';\n');
    fout.write('label "nI r = ' + '%.3f' %(rR[i]) + 'cm' + ' z = ' +\
               '%.3f' %(rZ[j]) + 'cm"' +'\n');
    fout.write('field  RHON \n');
    fout.write('species 4 \n');
    fout.write('at '+ '%.3f' %(rR[i])+' 0.0 '+'%.3f' %(rZ[j]) +'\n');
    fout.write('; \n');
    p+=1;
# electron density
    fout.write('probe' + '%d' %(p) +';\n');
    fout.write('label "ne r = ' + '%.3f' %(rR[i]) + 'cm' + ' z = ' +\
               '%.3f' %(rZ[j]) + 'cm"' +'\n');
    fout.write('field  RHON \n');
    fout.write('species 3 \n');
    fout.write('at '+ '%.3f' %(rR[i])+' 0.0 '+'%.3f' %(rZ[j]) +'\n');
    fout.write('; \n');
    p+=1;
# Ez
    fout.write("probe" + "%d" %(p) +";\n");
    fout.write('label "Ez r = ' + '%.3f' %(rR[i]) + 'cm' + ' z = ' +\
               '%.3f' %(rZ[j]) + 'cm"' +'\n');
    fout.write('field E Z \n');
    fout.write('at '+ '%.3f' %(rR[i])+' 0.0 '+'%.3f' %(rZ[j]) +'\n');
    fout.write('; \n');
    p+=1;
# Br
    fout.write("probe" + "%d" %(p) +";\n");
    fout.write('label "Br r = ' + '%.3f' %(rR[i]) + 'cm' + ' z = ' +\
               '%.3f' %(rZ[j]) + 'cm"' +'\n');
    fout.write('field B X \n');
    fout.write('at '+ '%.3f' %(rR[i])+' 0.0 '+'%.3f' %(rZ[j]) +'\n');
    fout.write('; \n');
    p+=1;
# By
    fout.write("probe" + "%d" %(p) +";\n");
    fout.write('label "By r = ' + '%.3f' %(rR[i]) + 'cm' + ' z = ' +\
               '%.3f' %(rZ[j]) + 'cm"' +'\n');
    fout.write('field B Y \n');
    fout.write('at '+ '%.3f' %(rR[i])+' 0.0 '+'%.3f' %(rZ[j]) +'\n');
    fout.write('; \n');
    p+=1;
# Bz
    fout.write("probe" + "%d" %(p) +";\n");
    fout.write('label "Bz r = ' + '%.3f' %(rR[i]) + 'cm' + ' z = ' +\
               '%.3f' %(rZ[j]) + 'cm"' +'\n');
    fout.write('field B Z \n');
    fout.write('at '+ '%.3f' %(rR[i])+' 0.0 '+'%.3f' %(rZ[j]) +'\n');
    fout.write('; \n');
    p+=1;



fout.close()
exit()
