"""
Script to measure dust concentration factor from Phantom SPH runs, by fitting a
    gaussian profile to a radial slice of the disc.
"""

import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import nct
from scipy.optimize import curve_fit

def fitGaussian(x, N, mu, sig):
        #return (N/np.sqrt(2*np.pi*sig*sig))*np.exp(-1*(x-mu)**2/(2*sig*sig)) + 0.01
        return N*np.exp(-1*(x-mu)**2/(2*sig*sig))

parser = argparse.ArgumentParser()
parser.add_argument('-disc')
args = parser.parse_args()
disc = args.disc

files = glob.glob('%s/asciifiles/*.ascii' %disc)
sizes = [file.split('dustysgdisc_')[1].split('.ascii')[0] for file in files]    #sort files
nums = []
for size in sizes:
    if size[-2:]=='um':
        num = int(size[:-2])
    elif size[-2:]=='mm':
        num = int(size[:-2])*1e3 
    nums.append(num)
files = np.asarray(files)[np.argsort(nums)]
nums = np.asarray(nums)[np.argsort(nums)]

if disc=='disc2':
    range = [78,85]
if disc=='disc3':
    range = [63,70]
elif disc=='disc4':
    range = [60,65]

G = 6.672041e-8
umass = 1.99e33
udist = 1.50e13
utime = np.sqrt((udist**3)/(G*umass))

#first find theta for slice i want to use
#for this i will use the dustysgdisc_200mm.ascii file
data = np.genfromtxt('%s/asciifiles/dustysgdisc_350mm.ascii' %disc)
rads = np.sqrt(data[:,0]*data[:,0] + data[:,1]*data[:,1])                 #radial dist from particles to star
clean = (rads>=20.) & (rads<=100.) & (data[:,10]!=0) & (data[:,-1]==1)    #remove unwanted data columns

data = data[clean]
thetapeak = np.arctan2(data[data[:,10]==np.max(data[:,10]),1][0],
                       data[data[:,10]==np.max(data[:,10]),0][0])         #find location of peak dust-to-gas in disc
thetaslice = [thetapeak-np.radians(2.5), thetapeak+np.radians(2.5)]       #window 2.5 degrees either side of thetapeak 

fstat = open('%s/fitstats.txt' %disc, 'w')                   #file for storing Gaussian fit statistics

multiplot=True
first=True

peak_df = []
sizes = []

for i, file in enumerate(files):
    sizes.append(file.split('dustysgdisc_')[1].split('.ascii')[0])

    if nums[i]<3.5e5:
        continue

    data = np.genfromtxt(file)

    rads = np.sqrt(data[:,0]*data[:,0] + data[:,1]*data[:,1])                 #radial dist from particles to star
    #calc sound speed [(gamma-1)*ui] and omega and scale height
    cs = np.sqrt((5./3.-1)*data[:,9])
    omega = np.sqrt(G*1*umass/(np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)*udist)**3)
    omega = omega*utime
    H = cs/omega

    clean = (abs(data[:,2]<=H)) & (rads>=15.) & (rads<=105.) & (data[:,10]!=0) & (data[:,-1]==1)    #remove unwanted data columns
    data = data[clean]
    thetas = np.arctan2(data[:,1],data[:,0])                              #angle of each data point
    inslice = (thetas>=thetaslice[0]) & (thetas<=thetaslice[1])           #datapoints inside slice of interest
    data = data[inslice]
    rads = np.sqrt(data[:,0]*data[:,0]+data[:,1]*data[:,1])

    data[:,10] = data[:,10]*1e8                                           #correct for dust-to-gas ratio=1e-10 in sph

    plt.figure()
    spiral = (rads>range[0]) & (rads<range[1])
    try:
        popt,pcov = curve_fit(fitGaussian,rads[spiral],data[spiral,10],
                              p0=[data[spiral,10].max(),0.5*np.sum(range),2.5]) #np.diff(range)[0]])
        xfit = np.linspace(range[0]-10,range[1]+10,100)
        yfit = fitGaussian(xfit,popt[0],popt[1],popt[2])
        plt.plot(xfit,yfit)
        fitfailed=False
    except:
        fitfailed=True

    spiral = (rads>range[0]-10) & (rads<range[1]+10)
    plt.scatter(rads[spiral],data[spiral,10],s=0.05,label=file.split('dustysgdisc_')[1].split('.ascii')[0])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.xlabel(r'$R, $ AU')
    plt.ylabel('Dust-to-gas mass ratio')
    plt.ylim([0,0.1])
    plt.xlim([range[0]-10, range[1]+10])
    plt.savefig('%s/figures/%s.pdf' %(disc, file.split('dustysgdisc_')[1].split('.ascii')[0]))

    if ((file.split('dustysgdisc_')[1].split('.ascii')[0] in 
         ['100mm','200mm','300mm','350mm','400mm','450mm','500mm','1000mm','2000mm']) & 
         (not fitfailed)):
        plt.figure('combo')
        plt.plot(xfit,yfit,label=file.split('dustysgdisc_')[1].split('.ascii')[0])

    if multiplot and not fitfailed:
        if sizes[i] in ['20mm','50mm','100mm','200mm','400mm','2000mm']:
            if first:
                multifig,multiax = plt.subplots(3,2,figsize=(5,7))
                first=False
            if sizes[i]=='20mm':
                sp=(0,0)
                multiax[sp].get_xaxis().set_visible(False)
            elif sizes[i]=='50mm':
                sp=(0,1)
                multiax[sp].get_yaxis().set_visible(False)
                multiax[sp].get_xaxis().set_visible(False)
            elif sizes[i]=='100mm':
                sp=(1,0)
                multiax[sp].get_xaxis().set_visible(False)
                multiax[sp].set_ylabel('Dust-to-gas mass ratio')
            elif sizes[i]=='200mm':
                sp=(1,1)
                multiax[sp].get_yaxis().set_visible(False)
                multiax[sp].get_xaxis().set_visible(False)
            elif sizes[i]=='400mm':
                sp=(2,0)
                multiax[sp].set_xlabel(r'$R, $ AU')
            elif sizes[i]=='2000mm':
                sp=(2,1)
                multiax[sp].get_yaxis().set_visible(False)
                multiax[sp].set_xlabel(r'$R, $ AU')
            multiax[sp].scatter(rads[spiral],data[spiral,10],s=0.025)
            multiax[sp].plot(xfit,yfit,label=r'$a=$ %s' %file.split('dustysgdisc_')[1].split('.ascii')[0])
            multiax[sp].legend(loc='upper right')
            multiax[sp].set_ylim([0,0.1])
            multiax[sp].set_xlim([range[0]-10, range[1]+10])

    plt.figure()
    plt.scatter(rads,data[:,10],s=0.05,label=file.split('dustysgdisc_')[1].split('.ascii')[0])
    if not fitfailed:
        xfit = np.linspace(20,100,500)
        yfit = fitGaussian(xfit,popt[0],popt[1],popt[2])
        plt.plot(xfit,yfit)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.xlabel(r'$R, $ AU')
    plt.ylabel('Dust-to-gas mass ratio')
    plt.ylim([0,0.1])
    plt.xlim([20,100])
    plt.savefig('%s/figures/%s_full.pdf' %(disc, file.split('dustysgdisc_')[1].split('.ascii')[0]))

    print('Done ', file.split('dustysgdisc_')[1].split('.ascii')[0])
    if fitfailed:
        print('Gaussian fit failed')
        fstat.write('%i %i %i %i %i %i %i \n' %(nums[i], 0,0,0,0,0,0))
    else:
        print('fit values: ', popt, [pcov[0,0]**0.5,pcov[1,1]**0.5,pcov[2,2]**0.5])
        fstat.write('%i %.3f %.3f %.3f %.3f %.3f %.3f \n' %(nums[i],popt[0],pcov[0,0]**0.5,popt[1],pcov[1,1]**0.5,popt[2],pcov[2,2]**0.5))
    print('')


#multifig.savefig('%s/figures/multi.pdf' %disc)
fstat.close()

plt.figure('combo')
plt.grid(alpha=0.25)
plt.legend()
plt.xlabel(r'$R, $ AU')
plt.ylabel('Dust-to-gas mass ratio')
plt.savefig('%s/figures/combo.pdf' %disc)

plt.show()
