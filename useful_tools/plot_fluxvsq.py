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

def getImPeakOld(im):
    #mask centre of image
    pix2au = im.header[13]
    imcentre = np.array([im.header[3],im.header[4]])/2                            #centre pixel
    imsize = im.header[3]*pix2au                                                  #width of image in AU    
    masksize = 0.25*imsize/2                                                      #mask radius in AU
    xs,ys = np.meshgrid(np.arange(im.header[3]),np.arange(im.header[4]))
    rads = np.sqrt(np.abs(xs-imcentre[0])**2 + np.abs(ys-imcentre[1])**2)*pix2au  #dist from centre pixel in AU
    mask = rads>masksize                                                          #mask pixels in inner region
    peak = np.max(im.data[mask])
    return peak

def getImPeak(im):
    #mask centre of image
    pix2au = 220./3200.
    imcentre = np.array([1600,1600])                                              #centre pixel
    imsize = 3200.*pix2au                                                          #width of image in AU    
    masksize = 0.25*imsize/2                                                      #mask radius in AU
    xs,ys = np.meshgrid(np.arange(3200),np.arange(3200))
    rads = np.sqrt(np.abs(xs-imcentre[0])**2 + np.abs(ys-imcentre[1])**2)*pix2au  #dist from centre pixel in AU
    mask = rads>masksize                                                          #mask pixels in inner region
    peak = np.max(im.data[mask])
    return peak

amax_list = ['10um','1mm','10cm','100cm','frag10','frag30']
amax_labels = ['10um','1mm','10cm','100cm',r'v$_{\rm frag}=10$ms$^{-1}$',r'v$_{\rm frag}=30$ms$^{-1}$']
mdot_list = ['1E-8','1.58E-8','2.81E-8','5E-8','1E-7','1.58E-7','2.81E-7','5E-7','1E-6']
peaks = []

for amax in amax_list:
    for mdot in mdot_list:
        #get peak pixel flux from FITS file
        try:
            im = fits.open("%s_amax%s/continuum%sGHz.fits" %(mdot,amax,freq))
            peak = getImPeak(im[0])
            im.close()
        except:
            peak = 0
	peaks.append(peak)

peaks = np.reshape(np.array(peaks),[6,9])

x = np.log10(np.array([1e-8,5e-8,1e-7,5e-7,1e-6]))
xnew = np.log10(np.linspace(1e-8,1e-6,300))

#for i, row in enumerate(peaks):
#    smooth = spline(x, row, xnew, 2)
#    #plt.plot(x,row,label=amax_list[i])
#    #plt.plot(xnew,smooth,label=amax_list[i])
#plt.ylim([0,1e-5])
#plt.legend()
#plt.show()

x = np.array([1e-8,1.58e-8,2.81e-8,5e-8,1e-7,1.58e-7,2.81e-7,5e-7,1e-6])
for i, row in enumerate(peaks):
    #plt.scatter(np.log10(x),row,label=amax_list[i])
    plt.plot(np.log10(x),row,label=amax_labels[i])

plt.xlabel(r'log$_{10}$($\dot{M}$)')
plt.ylabel('Peak spiral flux / mJy')
plt.ylim([0,1.1*np.max(peaks)])
#plt.xscale('log')
plt.legend()
plt.show()
