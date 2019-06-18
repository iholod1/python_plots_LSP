import sys, getopt, os, os.path
def readDeck(dirname,str1,str2):
  """Reads LSP deck
     Input: directory, word to trigger search, value of interest, word to stop search
     Output: value"""


#str1= 'species3' # trigger word
#str2= 'selection_ratio' # value of interest
  str3= 'species' # first stop word after trigger


#  dirname=os.getcwd() + '/';
  lst_dir=os.listdir(dirname);
  fnames=[];
  for fid in lst_dir:
    if '.lsp' in fid: fnames.append(fid)
  if len(fnames)==0: 
    print("No input files");
    return();
  elif len(fnames)>1: # in case there are several decks in the directory
    while True:
      try:
        print(fnames);
        n = input("Which file? ")-1;
      except SyntaxError:
        print("Incorrect choice");
        continue;
      else:
        if type(n) is not int:
           print("Incorrect choice: must be integer number")
           continue;
        if n>=0 and n<len(fnames):
          fname=fnames[n];
          break
        else:
          print("Incorrect choice: must be between 1 and" + str(len(fnames)));
          continue;
  else: fname=fnames[0];
 
  print(fname);

  f0=False;
  f1=False;
  with open(fname) as infile:
    for line in infile:
       if len(line)<2: continue
       ln = line.replace(" ","").lower().split()[0];
       lnsplit = line.lower().split();
       if "[particlespecies]" in ln: 
         f1=True;
       if f1 and (str1 in ln[0:min([len(str1),len(ln)])]): 
         f0=True;
         continue;
       if f0 and (str2 in ln[0:min([len(str2),len(ln)])]): 
         val = float(lnsplit[1]);
         if (str2=="mass") and (str(lnsplit[min([2,len(lnsplit)-1])])=="amu"): val*=1822.3;
         break;
       if f0 and (str3 in ln[0:min([len(str3),len(ln)])]): 
         break;
  
  if 'val' in locals(): return val;

