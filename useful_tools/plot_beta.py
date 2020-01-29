import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mdot', '-mdot')
parser.add_argument('--amax', '-amax')
args = parser.parse_args()

mdot = args.mdot
amax = args.amax

f1 = 680
f2 = 127

# Going to calc beta parameter between 680GHz and 127GHz
im1 = fits.open("%s_amax%s/mjyCASA_%iGHz_ant8_noisyimage_residuals.fits" %(mdot,amax,f1))
im2 = fits.open("%s_amax%s/mjyCASA_%iGHz_ant8_noisyimage_residuals.fits" %(mdot,amax,f2))

#im1 = fits.open("continuum%iGHz.fits" %f1)
#im2 = fits.open("continuum%iGHz.fits" %f2)

im1[0].data[im1[0].data<=0] = 0.001
im2[0].data[im2[0].data<=0] = 0.001

#calculate beta parameter from DiPierro 15, equation 15.
beta = np.log(im2[0].data) - np.log(im1[0].data)
beta = beta/(np.log(f2)-np.log(f1)) 
beta -= 2

#set regions not in the disc to beta=beta_ism=1.7

xs = np.arange(-1600,1600)
ys = np.arange(-1600,1600)
XX, YY = np.meshgrid(xs,ys)
dist = np.sqrt(XX*XX + YY*YY)
lim = 3200./220.*100.
betaplot = beta[0,0]
betaplot[dist>lim] = 1.7

betaplot[betaplot<0]=0
betaplot[betaplot>2]=2

#import pdb; pdb.set_trace()

plt.imshow(betaplot)
plt.colorbar()
plt.show()
