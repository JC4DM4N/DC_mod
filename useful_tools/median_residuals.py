import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

im = fits.open('imlup/1.3e-8/continuum240_4GHz.fits')

'''
#elias27
dist = 116 #pc
inc = np.radians(28.8)
pa = np.radians(56.2)

#waoph
dist = 123 #pc
inc = np.radians(47.3)
pa = np.radians(6.8)
'''
#imlup
dist = 158
inc = np.radians(47.5)
pa = np.radians(24.5)

#pix2au = abs(im[0].header['CDELT1']*60*60*dist)
pix2au = abs(im[0].header['CDELT1'])

#imsize = im[0].data[0,0].shape
imsize = im[0].data.shape

#make array of distance from image centre at each point in array (units: au)
XX, YY = np.meshgrid(np.arange(imsize[0]).astype(float),np.arange(imsize[1]).astype(float))
#ZZ = np.zeros(im[0].data[0,0].shape)
ZZ = np.zeros(im[0].data.shape)

XX -= imsize[0]/2.
YY -= imsize[1]/2.
YY = np.flip(YY)

#transform for inclination
XX_ = XX
YY_ = YY*np.cos(inc)
ZZ_ = -1*YY*np.sin(inc)

#transform for position angle
XX__ = XX_*np.cos(pa) - ZZ_*np.sin(pa)
YY__ = YY_
ZZ__ = XX_*np.sin(pa) + ZZ_*np.cos(pa)

XX += imsize[0]/2.
YY += imsize[1]/2.

#import pdb; pdb.set_trace()

rads = np.sqrt(XX__*XX__ + YY__*YY__)*pix2au

#plt.imshow(im[0].data)
#plt.contour(XX,YY,rads)
#plt.show()

bins = range(0, int(np.max(rads)))

for i in range(1, np.max(bins)):
	#median = np.median(im[0].data[0,0][(rads<=bins[i]) & (rads>bins[i-1])])
	#im[0].data[0,0][(rads<=bins[i]) & (rads>bins[i-1])] -= median
	median = np.median(im[0].data[(rads<=bins[i]) & (rads>bins[i-1])])
	im[0].data[(rads<=bins[i]) & (rads>bins[i-1])] -= median

im.writeto('imlup/1.3e-8_residuals/continuum240GHz_residuals.fits',overwrite=True)
im.close()
#import pdb; pdb.set_trace()
