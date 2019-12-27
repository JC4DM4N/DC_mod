import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import spline
from scipy.optimize import curve_fit
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument('-freq','--freq')
args = parser.parse_args()
freq = args.freq


def fitLogNorm(x, N, mu, sig):
	return (N/(x*sig*np.sqrt(2*np.pi)))*np.exp(-1*(np.log(x)-mu)**2/(2*sig*sig)) + 0.01

amax_list = ['10um','1mm','10cm','100cm','frag10','frag30']
mdot_list = ['1E-8','5E-8','1E-7','5E-7','1E-6']
peaks = []

for amax in amax_list:
    for mdot in mdot_list:
        #get peak pixel flux from FITS file
        im = fits.open("%s_amax%s/continuum%sGHz.fits" %(mdot,amax,freq))
        peak1 = im[0].data[0:1200,0:1200].max()
        peak2 = im[0].data[0:1200,2000:3200].max()
        peak3 = im[0].data[2000:3200,0:1200].max()
        peak4 = im[0].data[2000:3200,2000:3200].max()
	peak = np.max([peak1,peak2,peak3,peak4])
        im.close()
	peaks.append(peak)

peaks = np.reshape(np.array(peaks),[6,5])

x = np.log10(np.array([1e-8,5e-8,1e-7,5e-7,1e-6]))
xnew = np.log10(np.linspace(1e-8,1e-6,300))

#for i, row in enumerate(peaks):
#    smooth = spline(x, row, xnew, 2)
#    #plt.plot(x,row,label=amax_list[i])
#    #plt.plot(xnew,smooth,label=amax_list[i])
#plt.ylim([0,1e-5])
#plt.legend()
#plt.show()

x = np.array([1e-8,5e-8,1e-7,5e-7,1e-6])
for i, row in enumerate(peaks):
    plt.scatter(np.log10(x),row,label=amax_list[i])

plt.ylim([0,1e-5])
#plt.xscale('log')
plt.legend()
plt.show()
