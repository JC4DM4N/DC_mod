"""
Plot the opacity spectral index, beta, from torus & CASA synthetic observations.
"""

import os
from astropy.io import fits
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import argparse
import aplpy
import pyfits
import math
from matplotlib import cm
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-folder', '--folder')
args = parser.parse_args()
folder = args.folder

f1 = 100.
f2 = 460.

CASA=1

if CASA==1:
    im1 = fits.open('%s/mjyCASA_%sGHz_ant8_noisyimage.fits' %(folder,int(f1)))
    im2 = fits.open('%s/mjyCASA_%sGHz_ant8_noisyimage.fits' %(folder,int(f2)))

    im1[0].data = im1[0].data[0,0]
    im2[0].data = im2[0].data[0,0]
else:
    im1 = fits.open('%s/continuum%sGHz.fits' %(folder,int(f1)))
    im2 = fits.open('%s/continuum%sGHz.fits' %(folder,int(f2)))

#get region within disc radius
x = range(3200)
y = range(3200)
XX, YY = np.meshgrid(x,y)
XX -= 1600
YY -= 1600
rads = np.sqrt(XX*XX + YY*YY)
rdisc = 3200./220.*70.
indisc = rads < rdisc
outdisc = rads>rdisc


ncells = 3200

print (im1[0].data).shape


#Some cells have negative flux, which doesn't make sense.
#Set to background RMS at each wavelength, rather than fixed value

total = 0.0
count = 0
for i in range(2750,3199):
    for j in range(2750,3199):
        total = total + im1[0].data[i,j]*im1[0].data[i,j]
        count = count + 1

RMS = total/count
RMS = math.sqrt(RMS)

im1[0].data[im1[0].data<RMS] = RMS

total = 0.0
count = 0
for i in range(2750,3199):
    for j in range(2750,3199):
        total = total + im2[0].data[i,j]*im2[0].data[i,j]
        count = count + 1

RMS = total/count
RMS = math.sqrt(RMS)


im2[0].data[im2[0].data<RMS] = RMS

#calc pixelwise beta
beta = np.log(im1[0].data) - np.log(im2[0].data)
    
beta = beta/(np.log(f1) - np.log(f2))
beta -= 2

#I played around with getting things outside the disc to equal b_ism, but I don't think it matters. Let's just leave it.
mu = 1.7
sigma=0.5
for i in range(0,ncells):
    for j in range(0,ncells):
        if((outdisc[i,j])):
            if(beta[i,j] < 1.0):           
               # beta[i,j]  = 1.7
                beta[i,j] = beta[i,j]
#beta[(outdisc) & (beta<0.0)] =np.random.normal(mu,sigma,1000)
#beta[outdisc and beta<0.0]      = 1.7

# Put beta into a 2d array.
beta2d=beta.reshape(3200,3200)

betax=beta2d[:,0]

xlist=np.arange(0,ncells,1)
ylist = np.arange(0,ncells,1)

#Change coordinate to centre on zero
xlist-=ncells/2
ylist-=ncells/2


#Check and remove existing beta file if necessary
try:
    os.remove('%s/beta_cass.fits' %folder)
except:
    pass

#Write the beta values to fits file
hdu = fits.PrimaryHDU(beta)
hdul = fits.HDUList([hdu])
hdul[0].data = [[hdul[0].data]]
hdul[0].header = im1[0].header
hdul.writeto('%s/beta_cass.fits' %folder)

beta_file=("%s/beta_cass.fits" %folder)
f = aplpy.FITSFigure(beta_file)

ax = plt.axes([0,0,1,1],frameon=False)
hdulist = fits.open(beta_file)
data,header_data=pyfits.getdata(beta_file,0,header=True)

bmaj =  hdulist[0].header['bmaj'] 
bmin =  hdulist[0].header['bmin']  
bpa  =  hdulist[0].header['bpa'] 

scidata = hdulist[0].data

total = 0.0
count = 0
for i in range(2750,3199):
    for j in range(2750,3199):
        total = total + scidata[0,0,i,j]*scidata[0,0,i,j]
        count = count + 1

RMS = total/count
RMS = math.sqrt(RMS)

print "----------------------"
print "RMS is",RMS,"mJy/beam"
print "RMS is",RMS*1000,"micro Jy/beam"
print "beam is:", bmaj*3600,"asec by",bmin*3600,"asec"

rms=RMS
contours=[3.*rms,4.*rms,5.*rms,6.0*rms,9.*rms,12.*rms,15.*rms,18.*rms,24.*rms,36.*rms]

printcontours = [x / rms for x in contours]
print "contours are",printcontours
print " "
print " "

#imgage=f.show_contour(str,levels=contours,colors="white")
f.show_colorscale(cmap=cm.inferno,vmin=0.0,vmax=2.0,stretch='linear')
#f.show_colorscale(cmap=cm.inferno,stretch='linear')
f.add_colorbar()
f.colorbar.show()
f.colorbar.set_location('right')
f.colorbar.set_axis_label_text(r'$\beta_{\rm 100GHz/460GHz}$')
f.colorbar.set_axis_label_font(size=16), #weight='bold')
f.colorbar.set_pad(0.0) #padding between bar and axis
f.axis_labels.hide()
f.tick_labels.hide_y()
f.tick_labels.hide_x()
ax.text(0.15,0.85,r'$a_{\rm max}=10{\rm cm}$',color='white',fontsize=20)

f.show_beam(major=bmaj,minor=bmin,angle=90,fill=True,color='white')
f.ticks.set_minor_frequency(1)

plt.tight_layout()
plt.savefig('%s/beta_%s.png' %(folder,folder))
plt.show()
