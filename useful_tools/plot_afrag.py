"""
Plot the grain fragmentation threshold from torus runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fitGaussian(x, N, mu, sig):
	return (N/np.sqrt(2*np.pi*sig*sig))*np.exp(-1*(x-mu)**2/(2*sig*sig))

def fitLogNorm(x, N, mu, sig):
	return (N/(x*sig*np.sqrt(2*np.pi)))*np.exp(-1*(np.log(x)-mu)**2/(2*sig*sig)) + 0.01

data1e6 = np.genfromtxt('../SA_runs/1E-6_amaxfrag10/afrag.txt')
data5e7 = np.genfromtxt('../SA_runs/5E-7_amaxfrag10/afrag.txt')
data1e7 = np.genfromtxt('../SA_runs/1E-7_amaxfrag10/afrag.txt')
data5e8 = np.genfromtxt('../SA_runs/5E-8_amaxfrag10/afrag.txt')
data1e8 = np.genfromtxt('../SA_runs/1E-8_amaxfrag10/afrag.txt')

data1e6 = data1e6[np.argsort(data1e6[:,0])]
data5e7 = data5e7[np.argsort(data5e7[:,0])]
data1e7 = data1e7[np.argsort(data1e7[:,0])]
data5e8 = data5e8[np.argsort(data5e8[:,0])]
data1e8 = data1e8[np.argsort(data1e8[:,0])]

udist = 1.5e13

data1e6[:,0] = data1e6[:,0]/udist
data5e7[:,0] = data5e7[:,0]/udist
data1e7[:,0] = data1e7[:,0]/udist
data5e8[:,0] = data5e8[:,0]/udist
data1e8[:,0] = data1e8[:,0]/udist

data1e6 = data1e6[data1e6[:,0]>20]
data5e7 = data5e7[data5e7[:,0]>20]
data1e7 = data1e7[data1e7[:,0]>20]
data5e8 = data5e8[data5e8[:,0]>20]
data1e8 = data1e8[data1e8[:,0]>20]

rads = np.unique(data1e6[:,0])

#fit smooth curves to data
guesses = [np.max(data1e6[:,7]), 20, np.mean(data1e6[:,0])]
popt, pcov = curve_fit(fitGaussian, data1e6[:,0], data1e6[:,7], p0=guesses)
plt.plot(rads, fitGaussian(rads,popt[0],popt[1],popt[2]), '^', markersize=2.5, label=r'$\dot{M}$=1e-6, q=0.31')

guesses = [np.max(data5e7[:,7]), 20, np.mean(data5e7[:,0])]
popt, pcov = curve_fit(fitGaussian, data5e7[:,0], data5e7[:,7], p0=guesses)
plt.plot(rads, fitGaussian(rads,popt[0],popt[1],popt[2]), '-.', label=r'$\dot{M}$=5e-7, q=0.25')

guesses = [np.max(data1e7[:,7]), 20, np.mean(data1e7[:,0])]
popt, pcov = curve_fit(fitLogNorm, data1e7[:,0], data1e7[:,7], p0=guesses)
plt.plot(rads, fitLogNorm(rads,popt[0],popt[1],popt[2]), ':', label=r'$\dot{M}$=1e-7, q=0.16')

guesses = [np.max(data5e8[:,7]), 20, np.mean(data5e8[:,0])]
popt, pcov = curve_fit(fitLogNorm, data5e8[:,0], data5e8[:,7], p0=guesses)
plt.plot(rads, fitLogNorm(rads,popt[0],popt[1],popt[2]), '--', label=r'$\dot{M}$=5e-8, q=0.14')

guesses = [np.max(data1e8[:,7]), 20, np.mean(data1e8[:,0])]
popt, pcov = curve_fit(fitLogNorm, data1e8[:,0], data1e8[:,7], p0=guesses)
plt.plot(rads, fitLogNorm(rads,popt[0],popt[1],popt[2]), '-', label=r'$\dot{M}$=1e-8, q=0.10')

plt.xlim([20,100])
plt.ylim([0,6000])
plt.xlabel(r'$R$ (AU)', fontsize=12)
plt.ylabel(r'$a_{\rm frag}$ ($\mu$m)', fontsize=12)
plt.legend()
plt.savefig('/disk2/cadman/plots/dustconc20/png/afrag10ms.png')
plt.show()
