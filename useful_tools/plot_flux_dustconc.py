import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument('--freq', '-freq')
parser.add_argument('--mdot', '-mdot')
parser.add_argument('--amax', '-amax')
args = parser.parse_args()

def fluxRatio(im,m):
    #I know that the spirals are located at theta = 1/0.38 * log(r/13.5)
    #Take RMS flux in spiral regions / RMS flux in anti-spiral regions
    #defined by spiral peak theta + - 360./m/4. degrees

    pix2au = 220./3200.
    imcentre = np.array([1600,1600])                                              #centre pixel
    imsize = 3200.*pix2au                                                          #width of image in AU
    masksize = [50.,100.]                                                     #mask radius in AU
    xs,ys = np.meshgrid(np.arange(3200),np.arange(3200))
    rads = np.sqrt(np.abs(xs-imcentre[0])**2 + np.abs(ys-imcentre[1])**2)*pix2au  #dist from centre pixel in AU
    thetas = np.degrees(np.arctan2((ys-imcentre[0]),(xs-imcentre[0])))
    thetas[thetas<0] += 360.

    iarmtheta = np.degrees(1./0.38*np.log(rads/13.5))                         #theta of first interarm
    armtheta = iarmtheta + 360./m/2.                                           #theta of arm region

    armtheta[armtheta<0]+=360
    iarmtheta[iarmtheta<0]+=360

    mask = (rads>masksize[0]) & (rads<masksize[1])
    #arm = (thetas>armtheta-360./m/4.) & (thetas<armtheta+360/m/4.)

    arm = (thetas>armtheta-(360./m/8.)) & (thetas<armtheta+(360./m/8.))
    iarm = (thetas>iarmtheta-(360./m/8.)) & (thetas<iarmtheta+(360./m/8.))

    armrms = np.sqrt(np.mean(im.data[0][0][arm & mask]**2))
    iarmrms = np.sqrt(np.mean(im.data[0][0][iarm & mask]**2))

    armpeak = max(im.data[0,0][arm&mask])
    iarmpeak = max(im.data[0,0][iarm&mask])
    peakratio = armpeak/iarmpeak

    plot=False
    if plot:
        plt.imshow(im.data[0,0])
        plt.scatter(xs[arm&mask],ys[arm&mask],s=0.1,c='r')
        plt.scatter(xs[iarm&mask],ys[iarm&mask],s=0.1,c='b')
        plt.xlim([0,3200])
        plt.ylim([0,3200])
        plt.show()

    return peakratio

#im1 = fits.open('../SA_runs/%s_amax%s/mjyCASA_%sGHz_ant8_noisyimage_residuals.fits' %(args.mdot, args.amax, args.freq))
#ratio1 = fluxRatio(im1[0], 4)
#im2 = fits.open('../SA_runs2/%s_amax%s/mjyCASA_%sGHz_ant8_noisyimage_residuals.fits' %(args.mdot, args.amax, args.freq))
#ratio2 = fluxRatio(im2[0], 4)

SA = []
SA2 = []

mdot_list = ['1E-8','1.58E-8','2.81E-8','5E-8','1E-7','1.58E-7','2.81E-7','5E-7','1E-6']
m_list = [8,8,8,8,8,4,4,4,4]
for i, mdot in enumerate(mdot_list):
    im1 = fits.open('../SA_runs/%s_amaxfrag10/mjyCASA_%sGHz_ant8_noisyimage_residuals.fits' %(mdot, args.freq))
    SA.append(fluxRatio(im1[0], m_list[i]))
    im2 = fits.open('../SA_runs2/%s_amaxfrag10/mjyCASA_%sGHz_ant8_noisyimage_residuals.fits' %(mdot, args.freq))
    SA2.append(fluxRatio(im2[0], m_list[i]))

x = np.array([1e-8,1.58e-8,2.81e-8,5e-8,1e-7,1.58e-7,2.81e-7,5e-7,1e-6])
plt.plot(x, SA, label='with dust conc')
plt.plot(x, SA2, label='w/o dust conc')
plt.legend()
plt.show()