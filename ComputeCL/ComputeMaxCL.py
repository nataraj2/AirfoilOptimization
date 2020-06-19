import numpy as np
from math import *
#from pycse import bvp
import matplotlib.pyplot as plt
from scipy import *

CL = loadtxt('CoefficientOfLift.txt')
nairfoils = size(CL)

print max(CL[:])

optimal_CL = zeros(nairfoils)
optimal_CL[:] = 1.778
index_vec = zeros(nairfoils)

for i in arange(0,nairfoils,1):
	plt.plot(i,CL[i],'ob')
	plt.xlabel('Airfoil index',fontsize=20)
	plt.ylabel('$C_L$',fontsize=20)
	index_vec[i] = i
	
plt.plot(index_vec,optimal_CL,'-r',linewidth=2)	
plt.xlim([0,nairfoils])
plt.show()

	




