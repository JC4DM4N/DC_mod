import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

udist = 1.5e13

data = np.genfromtxt('../GRAINDIST/1E-6_amax100cm/grainfrac.txt')
zs = data[:,2]

#take slice from centre
midplane = np.unique(abs(zs[np.argsort(abs(zs))]))[0]
mpdata = data[abs(zs)<=midplane,:]

xs = mpdata[:,0]
ys = mpdata[:,1]
rads = np.sqrt(xs*xs + ys*ys)

grainsizes = np.logspace(-1,np.log10(200),num=50)

labels = ['0.1um','1um','10um','100um',
		  '1mm','1cm','10cm','100cm']
latexlabels = [r'0.1${\rm\mu}$m',r'1${\rm\mu}$m',r'10${\rm\mu}$m',r'100${\rm\mu}$m',
		  	   '1mm','1cm','10cm','100cm']

for i, col in enumerate([0,7,14,21,28,35,42,49]):
	eps = mpdata[:,col+3]
        xs = mpdata[:,0]
        ys = mpdata[:,1]

        slice = np.isclose(np.arctan2(ys,xs),1.56499941,0.03)

        rads = np.sqrt(mpdata[:,0]*mpdata[:,0] + mpdata[:,1]*mpdata[:,1])
       
        f = interp1d(rads[slice],eps[slice])
        import pdb; pdb.set_trace()
        radsnew = np.arange(rads.min()+1,rads.max()-1,100)
        epsnew = f(radsnew)

        plt.plot(radsnew, epsnew)

plt.yscale('log')
plt.show()
