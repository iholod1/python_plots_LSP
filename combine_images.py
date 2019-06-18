#!/usr/bin/env python
# script to execute multiple commands using MPI
# I. Holod, 06/09/17

import sys, getopt, os, os.path
from mpi4py import MPI
from PIL import Image

def im_combine(fid1,fid2,fid3,fid4,tstep):
    """combine 4 images together"""
    images = map(Image.open, [fid1,fid2,fid3,fid4])
    widths=[i.size[0] for i in images]
    heights=[i.size[1] for i in images]
    total_width = sum(widths[0:2])
    total_height = max(heights[0:2])+max(heights[2:4])

    new_im = Image.new('RGB', (total_width, total_height),"white")

    y_offset = max(heights[0:1])
    new_im.paste(images[0], (0,0))
    x_offset = widths[0]
    new_im.paste(images[1], (x_offset,0))
    new_im.paste(images[2], (0,y_offset))
    new_im.paste(images[3], (x_offset,y_offset))

    new_im.save('test'+tstamp+'.png')
    return()


comm=MPI.COMM_WORLD
myRank=comm.Get_rank()
nProc=int(comm.Get_size());

dirname=os.getcwd() + '/';

mask1=["Ez_",".png"]
mask2=["rBtheta_",".png"]
mask3=["pdf_extract3_",".png"]
mask4=["RhoT4_",".png"]


if myRank==0:
  fnames=[];
  lst_dir=os.listdir(dirname)
  for fid in lst_dir:
    if mask1[0] in fid and mask1[1] in fid:
      l0=len(mask1[0]); 
      l1=len(mask1[1]);
      tstamp = str(fid[l0:-l1])
      if (mask2[0]+tstamp+mask2[1] in lst_dir) and \
         (mask3[0]+tstamp+mask3[1] in lst_dir) and \
         (mask4[0]+tstamp+mask4[1] in lst_dir):
	      fnames.append(tstamp)

  print "Number of PEs = ", nProc
  print "Total number of files = ", len(fnames)
else:
  fnames=None;

comm.Barrier()
fnames=comm.bcast(fnames, root=0);

nFiles=int(len(fnames));

# distribute tasks
j=myRank
myList=[];
while j<nFiles:
  myList.append(fnames[j])
  j+=nProc

print "my list:", myList

for tstamp in myList:
  fid1 = mask1[0]+tstamp+mask1[1];
  fid2 = mask2[0]+tstamp+mask2[1];
  fid3 = mask3[0]+tstamp+mask3[1];
  fid4 = mask4[0]+tstamp+mask4[1];
  im_combine(fid1,fid2,fid3,fid4,tstamp)
  exit()

comm.Barrier()
exit()
