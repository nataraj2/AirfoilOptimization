import numpy as np
from math import *
#from pycse import bvp
import matplotlib.pyplot as plt
from scipy import *

airfoil_no = 7
filename = '../AirfoilData%03d.txt'%airfoil_no

actualdata = loadtxt(filename)
start = 8
end = 156

filename = 'CorrectedAirfoil%03d.txt'%airfoil_no
file = open(filename,'w')
for i in arange(start,end,1):
	file.write('%f %f\n'%(actualdata[i,0],actualdata[i,1]))	
file.write('%f %f'%(actualdata[start,0],actualdata[start,1]))
file.close()

print filename
data = loadtxt(filename)
plt.plot(data[:,0],data[:,1],'-b',linewidth=2)
imgfilename='OptimalAirfoil.png'
plt.axis('equal')
plt.savefig(imgfilename)

	

plt.show()



