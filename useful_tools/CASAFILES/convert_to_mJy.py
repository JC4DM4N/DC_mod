import pyfits
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--freq", "-freq")
args = parser.parse_args()
freq = str(args.freq)

for image in ['noisyimage', 'noisyimage_residuals']:
    str1=("CASA_%sGHz_ant8_%s.fits" %(freq,image))
    corstr="mjy"+str1
    data,header_data=pyfits.getdata(str1,0,header=True)
    data=data*1.0e3
    pyfits.writeto(corstr,data,header_data)
