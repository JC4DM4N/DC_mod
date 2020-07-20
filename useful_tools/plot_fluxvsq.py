"""
Plot how flux ratio varies with disk mass in torus & CASA synthetic observations.
"""

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

wav = 3e8/(int(freq)*10**6)

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
    peak = np.max(im.data[0][0][mask])
    #now find minimum interarm flux at ~same radial annulus to peak
    peakloc = rads[im.data[0][0]==peak]
    annulus = (rads<peakloc+1.) & (rads>peakloc-1.)                               #create annulus +- 1au at this radius
    ratio = min(im.data[0][0][annulus])
    ratio = peak/ratio
    return peak, ratio

def spiralFlux(im,m):
    #I know that the spirals are located at theta = 1/0.38 * log(r/13.5)
    #Take RMS flux in spiral regions / RMS flux in anti-spiral regions
    #defined by spiral peak theta + - 360./m/4. degrees

    pix2au = 220./3200.
    imcentre = np.array([1600,1600])                                              #centre pixel
    imsize = 3200.*pix2au                                                          #width of image in AU
    masksize = [70.,100.]                                                     #mask radius in AU
    xs,ys = np.meshgrid(np.arange(3200),np.arange(3200))
    rads = np.sqrt(np.abs(xs-imcentre[0])**2 + np.abs(ys-imcentre[1])**2)*pix2au  #dist from centre pixel in AU
    thetas = np.degrees(np.arctan2((ys-imcentre[0]),(xs-imcentre[0])))
    thetas[thetas<0] += 360.

    iarmtheta = np.degrees(1./0.38*np.log(rads/13.5))                            #theta of first arm
    armtheta = iarmtheta + 360./m/2.                                           #theta of interarm region
    armtheta[armtheta<0]+=360
    iarmtheta[iarmtheta<0]+=360

    mask = (rads>masksize[0]) & (rads<masksize[1])
    #arm = (thetas>armtheta-360./m/4.) & (thetas<armtheta+360/m/4.)

    arm = (thetas>armtheta-(360./m/8.)) & (thetas<armtheta+(360./m/8.))
    iarm = (thetas>iarmtheta-(360./m/8.)) & (thetas<iarmtheta+(360./m/8.))

    armrms = np.sqrt(np.mean(im.data[0,0][arm & mask]**2))
    iarmrms = np.sqrt(np.mean(im.data[0,0][iarm & mask]**2))
    #armrms = np.sqrt(np.mean(im.data[arm & mask]**2)) #use for continuum torus images
    #iarmrms = np.sqrt(np.mean(im.data[iarm & mask]**2)) #use for continuum torus images
    rms = np.sqrt(np.mean(im.data[0,0][2750:3200,2750:3200]**2))

    armmax = np.max(im.data[0,0][arm&mask])
    iarmmin = np.min(im.data[0,0][iarm&mask])

    plot=False
    if plot:
        plt.imshow(im.data[0,0])
        plt.scatter(xs[arm&mask],ys[arm&mask],s=0.1,c='r')
        plt.scatter(xs[iarm&mask],ys[iarm&mask],s=0.1,c='b')
        plt.xlim([0,3200])
        plt.ylim([0,3200])
        plt.show()


    #import pdb; pdb.set_trace()

    #return armmax/iarmmin
    return armrms/iarmrms

amax_list = ['10um','1mm','10cm','100cm','frag10','frag30']
amax_labels = [r'10${\rm \mu}$m','1mm','10cm','100cm',r'v$_{\rm frag}=10$ms$^{-1}$',r'v$_{\rm frag}=30$ms$^{-1}$']
mdot_list = ['1E-8','1.58E-8','2.81E-8','5E-8','1E-7','1.58E-7','2.81E-7','5E-7','1E-6']
m_list = [8,8,8,8,8,4,4,4,4]

cols = ['blue','orange','green','red','purple','brown']
markers = ['x','o','v','s','*','p']

#acc_max = 0
for d in ['SA_runs','SA_runs_nodustconc']: #'SA_runs_nodustconc'
    peaks = []
    for amax in amax_list:
        for i, mdot in enumerate(mdot_list):
            #get peak pixel flux from FITS file
            try:
                #im = fits.open("../SA_runs/%s_amax%s/continuum%sGHz.fits" %(mdot,amax,freq))
                im = fits.open("../%s/%s_amax%s/mjyCASA_%sGHz_ant8_noisyimage.fits" %(d,mdot,amax,freq))
                #peak, ratio = getImPeak(im[0])
                ratio = spiralFlux(im[0], m_list[i])
                #print(mdot, amax, ratio)
                im.close()
            except:
                peak = 0
                ratio = 0
            peaks.append(ratio)
            #mins.append(iarmrms)

    peaks = np.reshape(np.array(peaks),[6,9])
    plt.figure()

    x = np.array([1e-8,1.58e-8,2.81e-8,5e-8,1e-7,1.58e-7,2.81e-7,5e-7,1e-6])
    for i, row in enumerate(peaks):
        #plt.scatter(np.log10(x),row,label=amax_list[i])
        plt.scatter(np.log10(x),row,s=25.,label=amax_labels[i],marker=markers[i])
        plt.plot(np.log10(x),row)
        #plt.plot(np.log10(x),mins[i],label=amax_labels[i])

    plt.xlabel(r'log$_{10}$($\dot{M}$)', fontsize=15)
    #plt.ylabel('Arm RMS Flux / Interarm RMS Flux', fontsize=15)
    plt.ylabel(r'$F_{\rm arm,RMS} / F_{\rm interarm, RMS}$', fontsize=15)
    plt.title(r'$f_{\rm obs} = $%sGHz, $\lambda$=%.1fmm' %(freq,wav), fontsize=15)
    #acc_max = max(acc_max, np.max(peaks))
    #plt.ylim([0,1.1*np.max(peaks)])
    plt.ylim([0,5.])
    plt.xlim([-8,-6])
    #plt.xscale('log')
    plt.legend()
    plt.plot([-8,-6],[1,1],'--',c='black')
    plt.tight_layout()
    plt.savefig('../%s/fluxvsq_%sGHz_%s.png' %(d, freq, d))
plt.show()
