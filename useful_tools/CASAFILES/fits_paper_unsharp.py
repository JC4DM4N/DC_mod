import pyfits
import matplotlib
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt,numpy as np
import copy
import math
import aplpy
import matplotlib as M
#import mynormalize
from astropy import units as u
from astropy.io import fits
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--freq","-freq")

args = parser.parse_args()

freq = str(args.freq)

#str="millijytest.fits"
str=("mjyCASA_%sGHz_ant8_noisyimage_residuals.fits" %freq)
#str="millijyalmaout22.fits"

f = aplpy.FITSFigure(str)
ax = plt.axes([0,0,1,1],frameon=False)
str1=str
hdulist = fits.open(str1)
data,header_data=pyfits.getdata(str1,0,header=True)

bmaj =  hdulist[0].header['bmaj'] 
bmin =  hdulist[0].header['bmin']  
bpa  =  hdulist[0].header['bpa'] 


scidata = hdulist[0].data


##print scidata[0,0,1300,1300]

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
#rms=72e-3
#rms= 0.0009


rms=RMS
contours=[3.*rms,4.*rms,5.*rms,6.0*rms,9.*rms,12.*rms,15.*rms,18.*rms,24.*rms,36.*rms]

#contours=[3.*rms,6.*rms,9.*rms,12.*rms,15.*rms,18.*rms]
#contours=[3.*rms,4.*rms,5.*rms,6.*rms,9.*rms,12.*rms,15.*rms,18.*rms]
#newList = [x / myInt for x in myList]
printcontours = [x / rms for x in contours]
print "contours are",printcontours
print " "
print " "

imgage=f.show_contour(str,levels=contours,colors="white")
f.show_colorscale(cmap="gist_heat",vmin=0.0,vmax=0.1)

#f.add_colorbar()
#f.colorbar.show()
#f.colorbar.set_location('right')
#f.colorbar.set_axis_label_text(r'Flux')
#f.colorbar.set_axis_label_font(size=16), #weight='bold')
#f.colorbar.set_pad(0.0) #padding between bar and axis
f.axis_labels.hide()
f.tick_labels.hide_y()
f.tick_labels.hide_x()
#f.colorbar.set_axis_label_font(size=26), #weight='bold')
#f.colorbar.set_font(size=16, weight='medium', \
#                      stretch='normal', family='sans-serif', \
#                      style='normal', variant='normal')
#f.colorbar.set_pad(0.0) #padding between bar and axis


#f.show_beam(major=0.0275/3600,minor=0.0233/3600,angle=90,fill=True,color='white')
f.show_beam(major=bmaj,minor=bmin,angle=90,fill=True,color='white')
#f.beam.show()
f.ticks.set_minor_frequency(1)


plt.subplots_adjust(left=0.12, bottom=0.00001, right=0.94, top=0.99999, wspace=0.00001, hspace=0.00001)

plt.subplots_adjust(left=0.12, bottom=0.00001, right=0.92, top=0.98, wspace=0.00001, hspace=0.00001)
#f.colorbar.set_axis_label_font(size=16), #weight='bold')
#f.colorbar.set_pad(0.0) #padding between bar and axis
#f.axis_labels.hide()
#f.tick_labels.hide_y()
#f.tick_labels.hide_x()
#f.colorbar.set_axis_label_font(size=40), #weight='bold')
#f.colorbar.set_font(size=30, weight='medium', \
#                      stretch='normal', family='sans-serif', \
#                      style='normal', variant='normal')
#f.colorbar.set_pad(0.0) #padding between bar and axis
#f.colorbar.set_axis_label_font(size=12, weight='bold')
#M.rcParams['text.usetex'] = True #Let TeX do the typsetting
#M.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
#M.rcParams['font.family'] = 'sans-serif' # ... for regular text
#M.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
#M.rcParams['font.sans-serif'] = 'Helvetica' # Choose a nice font here


plt.tight_layout()
###plt.savefig('m24e-8_220GHz_stokes_8armsnew.png')
"""
##plt.savefig('q3_1e-9_almaC40-8_1h.pdf')
##plt.savefig('q3_1e-9_almaC40-8_1h.png')
##plt.savefig('q3_1e-9_almaC40-8_1h.eps')
##plt.savefig('q3_1e-9_almaC40-8_1h.jpg')
##plt.savefig('q3_1e-9_almaC40-8_1h.ps')
"""
plt.savefig("FINALIMAGE_%sGHz.eps" %freq)
plt.clf()
