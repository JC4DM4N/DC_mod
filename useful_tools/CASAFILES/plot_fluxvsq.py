import numpy as np
import argparse
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument('-freq','--freq')
args = parser.parse_args()
freq = args.freq


amax_list = ['10um','1mm','10cm','100cm','frag10','frag30']
mdot_list = ['1E-6','5E-7','1E-7','5E-8','1E-8']
peaks = []

for amax in amax_list:
    for mdot in mdot_list:
        #get peak pixel flux from FITS file
        im = fits.open("%s_amax%s/continuum%sGHz.fits" %(mdot,amax,freq))
        peak = str(im[0].data.max())
        im.close()
	peaks.append(peak)


peaks = np.reshape(np.array(peaks),[5,6])
import pdb; pdb.set_trace()
