"""
Plot how grain concentration varies with grain size in torus output fits files.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata

data = np.genfromtxt('grainfrac_new.txt')
zs = data[:,2]

#take slice from centre
midplane = np.unique(abs(zs[np.argsort(abs(zs))]))[0]
mpdata = data[abs(zs)<=midplane,:]

xs = mpdata[:,0]
ys = mpdata[:,1]
#zs = mpdata[:,20]

grainsizes = np.logspace(-1,np.log10(200),num=50)

labels = ['0.1um','1um','10um','100um',
		  '1mm','1cm','10cm','100cm']
latexlabels = [r'0.1${\rm\mu}$m',r'1${\rm\mu}$m',r'10${\rm\mu}$m',r'100${\rm\mu}$m',
		  	   '1mm','1cm','10cm','100cm']

for i, col in enumerate([0,7,14,21,28,35,42,49]):
	zs = mpdata[:,col+3]
	Ncontours=15.
	levels = [i*zs.max()/Ncontours for i in range(0,int(Ncontours)+1)]
	plt.figure()
	# define grid.
	xi = np.linspace(min(xs),max(xs),1000)
	yi = np.linspace(min(ys),max(ys),1000)
	# grid the data.
	zi = griddata((xs, ys), zs, (xi[None,:], yi[:,None]), method='cubic')
	zi[zi<0] = 0
	zi[np.isnan(zi)] = 0

	# contour the gridded data, plotting dots at the randomly spaced data points.
	CS = plt.contourf(xi,yi,zi,cmap=plt.cm.magma,levels=levels)	
	cbar = plt.colorbar(ticks=[0,zi.max()/2.,290.*zi.max()/300.],orientation="horizontal",
						fraction=0.046,pad=0.005,label='log dust-to-gas ratio') # draw colorbar
	cbar.ax.set_xticklabels(['0', '%.2f' %np.log10(zi.max()/2.), '%.2f' %np.log10(zi.max())])
	plt.axis('square')
	plt.xlim(min(xs),max(xs))
	plt.ylim(min(ys),max(ys))
	plt.xticks([])
	plt.yticks([])
	plt.text(-58000,54000,'a=%s' %latexlabels[i], fontsize=15, c='white')
	plt.tight_layout()
	plt.savefig('grainfrac%s.png' %labels[i], bbox_inches='tight')

plt.show()
