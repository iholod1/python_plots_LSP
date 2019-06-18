#!/usr/bin/env python3
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, getopt, os, os.path
from PIL import Image

def im_combine(fid1,fid2,tstep):
    """combine 2 images together"""
    
    images = []
    images.append(Image.open(fid1))
    images.append(Image.open(fid2))
    s = [i.size for i in images]
    w = [i[0] for i in s]
    h = [i[1] for i in s]
    print(w,h)
    total_width = max(w[0:2])
    total_height = sum(h[0:2])

    new_im = Image.new('RGB', (total_width, total_height),"white")

    y_offset = max(h[0:1])
    new_im.paste(images[0], (0,0))
    x_offset = w[0]
    new_im.paste(images[1], (0,y_offset))
    new_im.save('test'+tstamp+'.png')
    return()

mydir=os.getcwd()
# dirname = os.path.join(dirname,"plots")


f1 = []
f2 = []
t = []


# dirname1 = os.path.join("C:",os.sep, "Users","holod1","Documents","data","verus","P2V1A_mini","30kV12T_DT","kin3","RhoT5")
# dirname1 = os.path.join("C:",os.sep, "Users","holod1","Documents","data","verus","P2V1A_mini","30kV12T_hres","kin","RhoT5")
dirname1 = os.path.join("C:",os.sep, "Users","holod1","Documents","data","verus","P2V1A_mini","30kV12T_DT","fluid3","RhoT2")
print(dirname1)
lst_dir=os.listdir(dirname1)
for f in lst_dir:
    if f.startswith('RhoT2_T') and f.endswith('.png'):
        f1.append(f)
        t.append(f[7:-4])
print(t)        
# dirname2 = os.path.join("C:",os.sep, "Users","holod1","Documents","data","verus","P2V1A_mini","30kV12T_DT","kin3","Temp5")
# dirname2 = os.path.join("C:",os.sep, "Users","holod1","Documents","data","verus","P2V1A_mini","30kV12T_hres","kin","Temp5")
dirname2 = os.path.join("C:",os.sep, "Users","holod1","Documents","data","verus","P2V1A_mini","30kV12T_DT","fluid3","Temp2")
print(dirname2)
lst_dir=os.listdir(dirname2)
for f in lst_dir:
    if f.startswith('Temp2_T') and f.endswith('.png'):
        if f[7:-4] in t:
            f2.append(f)

for i in range(len(f2)):
    fid1 = os.path.join(dirname1,f1[i])
    fid2 = os.path.join(dirname2,f2[i])
    tstamp = t[i]
    print(fid1)
    print(fid2)
    im_combine(fid1,fid2,tstamp)

exit()


