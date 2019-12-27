
import shutil
import numpy as np
import os

# In CASA:
# execfile("thisfilename.py")

freq=680

numGaussians = 2 # 1 or 2

# Find peak intensity - get max value of image
modelname = ('sim_%sGHz_ant8/sim_%sGHz_ant8.alma.cycle6.6.noisy.image' %(freq,freq))
xstat = imstat(imagename=modelname)

#Look up the maxval and xstat to find where the highest flux pixel is.
maxval= float(xstat["max"])
max_position=[]

max_position=(xstat["maxpos"])

#==============================================
# Writing the estimates.txt file and setting box size
# peak intensity, peak x-pixel value, peak y-pixel value, major axis, minor axis, position angle, fixed (optional)
# 1 - small gaussian, 2 - large gaussian
pi1 = 0.9*maxval
pi2 = 0.1*maxval
#highest flux pixel
pxv = max_position[0]
pyv = max_position[1]

mia1 = 0.05 # major axis guess
mia2 = 0.05 # minor axis guess (arcsec)
#If inclined, may want to change this to ~90 or simething
pa = '0deg'
box = '0,0,3000,3000' # region we care about in pixels (x1,y1,x2,y2)
fopt = ',p' # must include ','

f = open("estimates.txt", "w") 
f.write(str(pi1) + "," + str(pxv) + "," + str(pyv) + "," + 
        str(mia1) + "arcsec," + str(mia1) + "arcsec," + pa + fopt) 
if numGaussians > 1:
        f.write("\n")
        f.write(str(pi2) + "," + str(pxv) + "," + str(pyv) + "," + 
                str(mia2) + "arcsec," + str(mia2) + "arcsec," + pa + fopt) 
f.close() 

#==============================================
# Fit Gaussians
default(imfit)
imagename = modelname
model = modelname+".model"
box = box
residual = modelname+".residuals"
estimates = "estimates.txt"
overwrite = True
append = False
rms = -1
excludepix = []
logfile = modelname+'.log'
#imfit()
go()

#==============================================
# Get stats on new image
#xstat=imstat(imagename=modelname+".residuals")
#maxval=float(xstat["max"])

# Normalize new image
#string="IM0/%f" % (maxval,)
#immath(imagename=modelname+".residuals", expr=string, outfile=modelname+".residuals.normalized")

#==============================================
# Export new image to fits
exportfits(imagename=modelname+".residuals", 
        fitsimage=modelname+".residuals.fits", 
        dropstokes=True, dropdeg=True, stokeslast=True, overwrite=True)

# Delete files
#shutil.rmtree(modelname+".model")
#shutil.rmtree(modelname+".residuals")
#shutil.rmtree(modelname+".residuals.normalized")






