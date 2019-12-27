#import shutil
import numpy as np
import os
from astropy.io import fits

freq=680
pwv=5.186
#pwv=2.748
pwv=0.472
t_obs='3h'

#get peak pixel flux from FITS file
im = fits.open("continuum%sGHz.fits" %freq)
peak = str(im[0].data.max())
im.close()

#FIRST RUN SIMOBSERVE
#old inbright="2.0e-7Jy/pixel"
#old incell="0.000491arcsec",
#old cell="0.000491arcsec"
#old totaltime="3h"
#old antennalist="alma.cycle5.8.cfg",

simobserve(project="sim_%sGHz_ant8" %freq,skymodel="continuum%sGHz.fits" %freq,
           inbright="%sJy/pixel"%peak,indirection="J2000 05h12m00 -38d00m00",
           incell="0.000491arcsec",incenter="%sGHz" %freq,inwidth="7.5GHz",
           complist="",compwidth="8GHz",setpointings=True,
           ptgfile="$project.ptg.txt",integration="10s",
           direction="J2000 05h12m00 -38d00m00",mapsize=['', ''],
           maptype="ALMA",pointingspacing="",caldirection="",calflux="1Jy",
           obsmode="int",refdate="2018/05/03",hourangle="transit",
           totaltime="%s"%t_obs,antennalist="alma.cycle6.6.cfg",
           sdantlist="aca.tp.cfg",sdant=0,
           thermalnoise="tsys-atm",user_pwv=pwv,t_ground=270.0,t_sky=260.0,
           tau0=0.1,seed=11111,leakage=0.0,graphics="none",verbose=True,
           overwrite=True)

#THEN RUN SIMANALYZE

simanalyze(project="sim_%sGHz_ant8" %freq,image=True,imagename="default",
           skymodel="",vis="sim_%sGHz_ant8.alma.cycle6.6.noisy.ms" %freq,
           modelimage="sim_%sGHz_ant8.alma.cycle6.6.skymodel" %freq,
           imsize=[3200, 3200],
           imdirection="J2000 -13.0h 20.0m 2.13162820728e-12s -200.0d 0.0m 0.0s ",
           cell="0.000491arcsec",interactive=False,niter=10000,
           threshold="0.01mJy",weighting="briggs",mask=[],outertaper=[],
           pbcor=True,stokes="I",featherimage="",analyze=False,showuv=True,
           showpsf=True,showmodel=True,showconvolved=False,showclean=True,
           showresidual=False,showdifference=True,showfidelity=True,
           graphics="none",verbose=False,overwrite=True,dryrun=False,logfile="")

#print('Performing unsharp masking...')
execfile('unsharp_casa.py')

print('Exporting fits...')
exportfits('sim_%sGHz_ant8/sim_%sGHz_ant8.alma.cycle6.6.noisy.image' %(freq,freq),
           'CASA_%sGHz_ant8_noisyimage.fits' %freq)
exportfits('sim_%sGHz_ant8/sim_%sGHz_ant8.alma.cycle6.6.noisy.image.residuals' %(freq,freq),
           'CASA_%sGHz_ant8_noisyimage_residuals.fits' %freq)
print('...done')

