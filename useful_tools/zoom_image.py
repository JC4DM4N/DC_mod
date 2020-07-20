"""
Zoom in on region of fits image.
"""

from astropy.io import fits

boxsize = [750,2450]

im = fits.open('mjyCASA_240GHz_ant8_noisyimage.fits')

im[0].data = [[im[0].data[0,0][boxsize[0]:boxsize[1],boxsize[0]:boxsize[1]]]]
im.writeto('mjyCASA_240GHz_ant8_noisyimage_zoomed.fits',overwrite=True)
im.close()
