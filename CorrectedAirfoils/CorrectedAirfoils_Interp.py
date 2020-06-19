import numpy as np
from math import *
#from pycse import bvp
import matplotlib.pyplot as plt
from scipy import *

airfoil_no = 46
filename = '../AirfoilData%03d.txt'%airfoil_no

actualdata = loadtxt(filename)
start = 15
end = 145

filename = 'CorrectedAirfoil%03d.txt'%airfoil_no
file = open(filename,'w')
for i in arange(start,end,1):
	file.write('%f %f\n'%(actualdata[i,0],actualdata[i,1]))	
file.write('%f %f'%(actualdata[start,0],actualdata[start,1]))
file.close()

print filename
data = loadtxt(filename)
#plt.plot(data[:,0],data[:,1],'-r')


npoints = size(data[:,0]) + size(data[:,0]) - 1
nactual = size(data[:,0])


print npoints, nactual
new_airfoil = zeros([npoints,2])

for i in arange(0,nactual-1,1):
	new_airfoil[2*i,:] = data[i,:]
	new_airfoil[2*i+1,:] = (data[i,:]+data[i+1,:])/2.0
new_airfoil[npoints-1,0] = data[nactual-1,0]
new_airfoil[npoints-1,1] = data[nactual-1,1]

filename = 'CorrectedAirfoil%03d.txt'%airfoil_no
file = open(filename,'w')
for i in arange(0,npoints,1):
        file.write('%f %f\n'%(new_airfoil[i,0],new_airfoil[i,1]))
file.write('%f %f'%(new_airfoil[0,0],new_airfoil[0,1]))
file.close()

finaldata = loadtxt(filename)

plt.plot(finaldata[:,0],finaldata[:,1],'-k')
plt.axis('equal')
	

plt.show()



