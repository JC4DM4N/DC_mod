import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kde

class DustConc():

    def __init__(self, ascii_file, grainsize, folder):
        filepath = str(folder)+'/'+str(ascii_file)
        data = np.genfromtxt(filepath)
        self.grainsize = grainsize

        xpos = data[:, 0]
        ypos = data[:, 1]
        zpos = data[:, 2]
        dens = data[:, 5]
        energy = data[:, 9]
        dustfrac = data[:, 10]

        gasx = xpos[data[:, -1] == 1]
        gasy = ypos[data[:, -1] == 1]
        gasz = zpos[data[:, -1] == 1]
        gasdens = dens[data[:, -1] == 1]
        gasenergy = energy[data[:, -1] == 1]
        dustfrac = dustfrac[data[:, -1] == 1]
        
        self.rin = 30.
        self.rout = 100.
        self.nbins = 60.

        self.normcoeff = 0.01/np.mean(dustfrac)

        dustfrac = dustfrac*self.normcoeff

        #import pdb; pdb.set_trace()

        gasr = (gasx**2 + gasy**2)**0.5
        self.normcoeff = 0.01/np.mean(dustfrac[(gasr >= self.rin) &
        									   (gasr <= self.rout)])
        dustfrac = dustfrac*self.normcoeff

        self.G = 6.672041e-8
        self.umass = 1.99e33
        self.udist = 1.50e13
        self.utime = np.sqrt((self.udist**3)/(self.G*self.umass))
        self.uergg = self.udist**2/(self.utime**2)

        self.gas = np.array(zip(gasx, gasy, gasz, gasdens, gasenergy, dustfrac))

        self.thetawidth = 5. #degrees

        self.gamma = 5./3.
        self.graindens = 3.0 #g/cm^3
        self.mstar = 1.0

        self.titleog = filepath
        self.title = filepath

    def rice04plot(self, plot=True):
    	dustfrac = self.gas[:,5]

        gasx = self.gas[((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 <= self.rout) &
                        ((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 >= self.rin) &
                        (dustfrac != 0.0), 0]
        gasy = self.gas[((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 <= self.rout) &
                        ((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 >= self.rin) &
                        (dustfrac != 0.0), 1]
        dustfrac = dustfrac[((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 <= self.rout) &
       			    		((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 >= self.rin) &
                            (dustfrac != 0.0)]

        thetapeak = np.arctan2(gasy[dustfrac[:]==np.max(dustfrac)][0], 
                               gasx[dustfrac[:]==np.max(dustfrac)][0])

        print('max dustfrac: ' + str(np.max(dustfrac))[:7] + ' at ' + 
        	  str(np.degrees(thetapeak))[:7] + ' degrees')

        rnge = [-3, 1]

        dustfrac = dustfrac[(np.log10(dustfrac[:]) >= rnge[0]) &
                            (np.log10(dustfrac[:]) <= rnge[1])]

        if plot:

            hist, bins = np.histogram(np.log10(dustfrac), bins=20, range=rnge)
            hist = np.true_divide(hist, sum(hist))

            plt.bar(bins[:-1], hist, align='edge', width=np.diff(bins),
                    color='white', edgecolor='black')
            plt.title(self.titleog)
            plt.ylabel('N / N_tot')
            plt.xlim(rnge)
            plt.xlabel(r'log( $ \rho_P$ / $\rho$ )')
            plt.show()
        return thetapeak

    def runAnalysis(self, thetapeak, plot=False):
        thetaslice = [thetapeak - np.radians(self.thetawidth/2.),
                      thetapeak + np.radians(self.thetawidth/2.)]
        self.getThetaSlice(thetaslice)
        peak, peakbin = self.dusttogasRadial(plot=plot)
        return peak

    def getThetaSlice(self, thetaslice):
        gastheta = np.arctan2(self.gas[:, 1], self.gas[:, 0])

        self.title = self.titleog + ' theta = ', np.degrees(thetaslice[0])

        thetamin = thetaslice[0]
        thetamax = thetaslice[1]

        self.gasinslice = self.gas[(gastheta[:] >= thetamin) &
                                   (gastheta[:] < thetamax)]
        self.gasrinslice = np.sqrt(self.gasinslice[:, 0]*self.gasinslice[:, 0] +
                                   self.gasinslice[:, 1]*self.gasinslice[:, 1])

    def dusttogasRadial(self, plot=False):
        bins = np.linspace(self.rin, self.rout, self.nbins)

        dustfracs = []

        peak = 0
        peakbin = 0

        for i in range(1, len(bins)):

            dustfrac = self.gasinslice[(self.gasrinslice > bins[i-1]) &
                                       (self.gasrinslice <= bins[i]), 5]
            dustfrac = np.mean(dustfrac)

            dustfracs.append(dustfrac)

            if (dustfrac > peak) & (bins[i-1] >= 10):
                peak = dustfrac
                peakbin = i
        print('peak dustfracs: ', np.array(dustfracs)[np.argsort(dustfracs)][-3:])

        if plot:
            plt.plot(bins[1:], dustfracs)
            plt.title(self.title)
            plt.xlim([self.rin, self.rout])
            plt.xlabel('Radius (AU)')
            plt.ylabel('Dust to Gas Ratio')
            plt.show()

        return peak, peakbin

    def plotGas(self,thetapeak):
    	# create data
		x = self.gas[:,0][((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 <= self.rout) &
		                  ((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 >= self.rin)]
		y = self.gas[:,1][((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 <= self.rout) &
					      ((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 >= self.rin)]
		weights = self.gas[:,5][((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 <= self.rout) &
					      ((self.gas[:,0]**2 + self.gas[:,1]**2)**0.5 >= self.rin)]
 
		# Big bins
		plt.hist2d(x, y, bins=(100), cmap=plt.cm.jet, weights=weights)
		plt.plot([0,self.rout*np.cos(thetapeak-np.radians(self.thetawidth/2.))],
				 [0,self.rout*np.sin(thetapeak-np.radians(self.thetawidth/2.))])
		plt.plot([0,self.rout*np.cos(thetapeak+np.radians(self.thetawidth/2.))],
			     [0,self.rout*np.sin(thetapeak+np.radians(self.thetawidth/2.))])
		plt.show()

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('folder', type=str, nargs='+',
                        help='The name of the folder the dust files are in')
    parser.add_argument('dustfile', type=str, nargs='+',
                        help='The name of the dust file to be used')
    folder = parser.parse_args().folder[0]
    dustin = parser.parse_args().dustfile[0]
    dustfile = 'dustysgdisc_' + str(dustin) + '.ascii'

    if str(dustin)[-2:] == 'mm':
        grainsize = int(str(dustin)[:-2])
        grainsize = grainsize*0.1 #convert to cm
    elif str(dustin)[-2:] == 'um':
        grainsize = int(str(dustin)[:-2])
        grainsize = grainsize*0.0001 #convert to cm
    else:
        print 'Could not recognise grainsize'
        quit()

    dustyanalysis = DustConc(dustfile, grainsize, folder)
    thetapeak = dustyanalysis.rice04plot(plot=False)

    thetapeak = np.radians(115.)

    dustyanalysis.runAnalysis(plot=True, thetapeak=thetapeak)
    dustyanalysis.plotGas(thetapeak)


main()

