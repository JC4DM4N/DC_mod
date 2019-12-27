from astropy.io import fits
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--freq","-freq")
args = parser.parse_args()
freq = str(args.freq)

#get peak pixel flux from FITS file
im = fits.open("continuum%sGHz.fits" %freq)
peak = im[0].data.max()
im.close()
print(peak)
