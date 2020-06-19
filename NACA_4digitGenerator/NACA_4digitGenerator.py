import numpy as np
from math import *
#from pycse import bvp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy import *

nx = 60
ny = 60

xgrid = zeros([nx,ny])
ygrid = zeros([nx,ny])

for ii in arange(0,nx,1):
        for jj in arange(0,ny,1):
                xgrid[ii,jj] = -0.00781 +ii*20.0/1024

for ii in arange(0,nx,1):
        for jj in arange(0,ny,1):
                ygrid[ii,jj] = -0.01951*20 + jj*20.0/1024


Mmin = 0.0
Mmax = 0.06
nM  = 10
delM = (Mmax-Mmin)/(nM-1)
MM = zeros(nM)

for i in arange(0,nM,1):
	MM[i] = Mmin + i*delM

Pmin = 0.35
Pmax = 0.55
nP = 10
delP = (Pmax-Pmin)/(nP-1)
PP = zeros(nP)

for i in arange(0,nP,1):
	PP[i] = Pmin + i*delP


Tmin = 0.09
Tmax = 0.14
nT = 10
delT = (Tmax-Tmin)/(nT-1)
TT = zeros(nT)

for i in arange(0,nT,1):
	TT[i] = Tmin + i*delT


n = 81
xc = zeros(n)
yc = zeros(n)
theta = zeros(n)
xu = zeros(n)
yu = zeros(n)
xl = zeros(n)
yl = zeros(n)
yt = zeros(n)

for i in arange(0,n,1):
	beta = i*pi/(n-1)
	xc[i] = (1.0 - cos(beta))/2.0
	print xc[i]
	#xc[i] = i*1.0/n



a0 = 0.2969
a1 = -0.126
a2 = -0.3516
a3 = 0.2843
a4 = -0.1015

counter = 0
lap1 = 0
lap2 = 0
switch1 = 0
switch2 = 0

Pstart = 0
Pend = nP
for Pcount in arange(Pstart,Pend,1):
	P = PP[Pcount]
	
	if(switch1==0):
		Mstart = nM-1
		Mend = 0
		Mskip = -1
		switch1 = 1
	else:
		Mstart = 1
		Mend = nM
		Mskip = 1
		switch1 = 0		
	
	if(lap1==0):
		Mstart = nM-1
		Mend = -1
		lap1=1

	for Mcount in arange(Mstart,Mend,Mskip):
		M = MM[Mcount]

		if(switch2==0):
                	Tstart = nT-1
                	Tend = -1
			Tskip = -1
                	switch2 = 1
		else:
        	        Tstart = 0
                	Tend = nT
			Tskip = 1
                	switch2 = 0

		for Tcount in arange(Tstart,Tend,Tskip):
			T = TT[Tcount]
			print M, P, T
			for i in arange(0,n,1):
				x = xc[i]
				if(x < P):
					yc[i] = M/P**2*(2*P*x-x**2) 
					dycdx = 2*M/P**2*(P-x)
					theta[i] = atan(dycdx)
				else:
					yc[i] = M/(1-P)**2*(1 - 2*P +2*P*x - x**2)
					dycdx = 2*M/(1-P)**2*(P-x)
					theta[i] = atan(dycdx)

				yt[i] = T/0.2*(a0*x**0.5+a1*x+a2*x**2+a3*x**3+a4*x**4)

			for i in arange(0,n,1):
			
				xl[i] = xc[i] + yt[i]*sin(theta[i]) 
				yl[i] = yc[i] - yt[i]*cos(theta[i]) 

				xu[i] = xc[n-i-1] - yt[n-i-1]*sin(theta[n-i-1]) 
				yu[i] = yc[n-i-1] + yt[n-i-1]*cos(theta[n-i-1]) 

			print "hello"
			print yt[n-1]*sin(theta[n-1])
					  
     

			filename = 'geomshift%04d.txt'%counter
			print filename
			file = open(filename,"w")
		
			cutoff = 1

			for i in arange(cutoff,n,1):
                		file.write("%f %f \n" % (xu[i],yu[i]))
			for i in arange(1,n-cutoff,1):
                		file.write("%f %f \n" % (xl[i],yl[i]))
                	file.write("%f %f \n" % (xu[cutoff],yu[cutoff]))
					

			file.close()				

			plt.plot(xu,yu,'ok')
			plt.plot(xl,yl,'ok')
			plt.axis('equal')
			for ii in arange(0,nx,1):
                		plt.plot(xgrid[ii,:],ygrid[0,:],'-k')

		        for jj in arange(0,ny,1):
                		plt.plot(xgrid[:,0],ygrid[:,jj],'-k')
			#imgfilename = './Images/NACA_M=%d_P=%d_T=%d.png'%(M*100,P*100,T*100)
			plt.title('NACA AIrfoil no: %d, M=%03f, P=%03f, T=%03f'%(counter, M*100,P*100,T*100),fontsize=20)
			imgfilename = './Images/NACA_%03d.png'%(counter)
			plt.savefig(imgfilename)
			#plt.draw()
			#plt.pause(0.001)
			#plt.show()
			#exit()
			plt.clf()
			counter = counter+1	
			print counter


#plt.show()

		
