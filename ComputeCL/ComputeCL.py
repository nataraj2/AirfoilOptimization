import numpy as np
from math import *
#from pycse import bvp
import matplotlib.pyplot as plt
from scipy import *


with open('filenames.txt', 'r') as f:
    filenames = f.read().splitlines()

#print filenames

numfiles=size(filenames)
print numfiles
cl=zeros(numfiles)
cd=zeros(numfiles)
clbycd=zeros(numfiles)

filename = "CoefficientOfLift.txt"
file=open(filename,"w")

for filenum in arange(0,numfiles,1):
	print filenum
	data = loadtxt(filenames[filenum])

	#Need to avoid double (or more) counting
	npts = size(data[:,0])
	count=0
	nonredundantlist=zeros(1000)
	cdvec = zeros(1000)

	for i in arange(0,npts,1):
		shouldinclude=1
		for j in arange(0,i-1,1):
			if(data[i,0]==data[j,0] and data[i,1]==data[j,1]):
				shouldinclude=0
		
		if(shouldinclude==1):
			nonredundantlist[count]=data[i,4]
			cdvec[count] = data[i,5]
			count=count+1
	cl[filenum] = np.sum(nonredundantlist)*(10.0/512.0)/(0.5*1.226*256.9**2)
	cd[filenum] = np.sum(cdvec)*(10.0/512.0)/(0.5*1.226*256.9**2)
	clbycd[filenum] = cl[filenum]/cd[filenum]
	print filenum, cl[filenum], cd[filenum], clbycd[filenum]	
	file.write("%f\n"%cl[filenum])

file.close()
	

#plt.legend(loc=0)
#plt.ylabel('$C_L$')
#plt.xlabel(r'$\alpha$ (deg)')
#plt.savefig('./Images/clvsdeg_ansys.jpeg')
#plt.show()
