import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

class DustConc():

    def __init__(self,ascii_file,grainsize):
        data = np.genfromtxt(ascii_file)
        self.grainsize = grainsize

        xpos = data[:,0]
        ypos = data[:,1]
        zpos = data[:,2]
        mass = data[:,3]
        dens = data[:,5]
        energy = data[:,9]

        gasx = xpos[data[:,-1]==1]
        gasy = ypos[data[:,-1]==1]
        gasz = zpos[data[:,-1]==1]
        dustx = xpos[data[:,-1]==2]
        dusty = ypos[data[:,-1]==2]
        dustz = zpos[data[:,-1]==2]
        gasdens = dens[data[:,-1]==1]
        dustdens = dens[data[:,-1]==2]
        gasenergy = energy[data[:,-1]==1]

        self.G = 6.672041e-8
        self.umass = 1.99e33
        self.udist = 1.50e13
        self.utime = np.sqrt((self.udist**3)/(self.G*self.umass))
        self.uergg = self.udist**2/(self.utime**2)

        self.gas = np.array(zip(gasx,gasy,gasz,gasdens,gasenergy))
        self.dust = np.array(zip(dustx,dusty,dustz,dustdens))

        self.rin = 3
        self.rout = 50
        self.thetaslice = [np.radians(0), np.radians(2)]

        self.gasmass = mass[data[:,-1]==1][0]*self.umass
        self.gamma = 5./3.
        self.graindens = 3.0 #g/cm^3
        self.mstar = 1.0
        print('NOTE: The stellar mass used is %.2f' %self.mstar)

        self.title = ascii_file

    def runAnalysis(self):
        self.getThetaSlice()
        self.dusttogasRadial(nbins=50,plot=True)
        self.stokesRadial(nbins=50,plot=True)

    def getThetaSlice(self):
        gastheta = np.arctan2(self.gas[:,1], self.gas[:,0])
        dusttheta = np.arctan2(self.dust[:,1], self.dust[:,0])

        thetamin = self.thetaslice[0]
        thetamax = self.thetaslice[1]

        self.gasinslice = self.gas[(gastheta[:]>=thetamin) &
                                   (gastheta[:]<thetamax)]
        self.gasrinslice = np.sqrt(self.gasinslice[:,0]*self.gasinslice[:,0] +
                                   self.gasinslice[:,1]*self.gasinslice[:,1])
        
        self.dustinslice = self.dust[(dusttheta[:]>=thetamin) & 
                                     (dusttheta[:]<thetamax)]
        self.dustrinslice = np.sqrt(self.dustinslice[:,0]*self.dustinslice[:,0] + 
                                    self.dustinslice[:,1]*self.dustinslice[:,1])     

    def dusttogasRadial(self,nbins,plot=False):
        bins = np.linspace(self.rin,self.rout,nbins)

        #normalise dust to gas ratio to 0.01
        normcoeff = 0.01/np.true_divide(np.sum(self.dust[:,3]),
                                        np.sum(self.gas[:,3]))

        dusttogas=[]

        for i in range(1,len(bins)):
            gasdens = self.gasinslice[(self.gasrinslice>bins[i-1]) &
                                      (self.gasrinslice<=bins[i]), 3]
            gasdens = np.sum(gasdens)
            dustdens = self.dustinslice[(self.dustrinslice>bins[i-1]) &
                                        (self.dustrinslice<=bins[i]), 3]
            dustdens = np.sum(dustdens)

            dusttogas.append(normcoeff*np.true_divide(dustdens, gasdens))

        if plot:
            plt.plot(bins[1:],dusttogas)
            plt.title(self.title)
            plt.xlabel('Radius (AU)')
            plt.ylabel('Dust to Gas Ratio')
            plt.show()

    def stokesRadial(self, nbins, plot=False):
        bins = np.linspace(self.rin,self.rout,nbins)

        mstar = self.mstar*self.umass

        stokesarr=[]
        epsilonarr=[]

        for i in range(1,len(bins)):
            gasenergy = self.gasinslice[(self.gasrinslice>bins[i-1]) &
                                        (self.gasrinslice<=bins[i]), 4]
            gasenergy = gasenergy*self.uergg

            spsound = np.sqrt(self.gamma*(self.gamma-1)*gasenergy)
            spsound = np.max(spsound)

            radius = self.gasrinslice[(self.gasrinslice>bins[i-1]) &
                                      (self.gasrinslice<=bins[i])]
            radius = np.mean(radius)
            radius = radius*self.udist

            omega = np.sqrt(self.G*mstar/radius**3)

            gasdens = self.gasinslice[(self.gasrinslice>bins[i-1]) &
                                      (self.gasrinslice<=bins[i]), 3]
            gasdens = np.max(gasdens)
            gasdens = gasdens*self.umass/self.udist**3

            stokesnum = self.grainsize*self.graindens*omega/gasdens/spsound
            stokesarr.append(stokesnum)

            epsilon = 1. + 2.*10./(stokesnum + 1./stokesnum) - stokesnum/200.
            epsilonarr.append(epsilon)

        if plot:
            plt.plot(bins[1:],stokesarr)
            plt.title(self.title)
            plt.xlabel('Radius (AU)')
            plt.ylabel('stokes')
            plt.show()

            plt.plot(bins[1:],epsilonarr)
            plt.title(self.title)
            plt.xlabel('Radius (AU)')
            plt.ylabel('epsilon')
            plt.show()

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dustfile', type=str, nargs='+',
                   help='The name of the dust file to be used')
    dustin = parser.parse_args().dustfile[0]
    dustfile = 'dustysgdisc_' + str(dustin) + '.ascii'

    if str(dustin)[-2:] == 'mm':
        grainsize = int(str(dustin)[:-2])
        grainsize = grainsize*0.1 #convert to cm
    elif str(dustin)[-2:] == 'um':
        grainsize = int(str(dustin)[:-2])
        grainsize = grainsize*0.0001 #convert to cm
    else:
        print('Could not recognise grainsize')
        quit()

    dustyanalysis = DustConc(dustfile, grainsize)
    dustyanalysis.runAnalysis()

main()
