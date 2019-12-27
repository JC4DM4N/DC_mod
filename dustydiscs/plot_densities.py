import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf


def fitGaussian(x, N, mu, sig):
	return (N/np.sqrt(2*np.pi*sig*sig))*np.exp(-1*(x-mu)**2/(2*sig*sig)) + 0.01

def fitGaussianSkew(x, N, mu, sig):
	pdf = (N/np.sqrt(2*np.pi*sig*sig))*np.exp(-1*(x-mu)**2/(2*sig*sig))
	cdf = 0.5*(1+erf((x-mu)/(sig*np.sqrt(2))))
	return 2*pdf*cdf + 0.01


def fitLogNorm(x, N, mu, sig):
	return (N/(x*sig*np.sqrt(2*np.pi)))*np.exp(-1*(np.log(x)-mu)**2/(2*sig*sig)) + 0.01

allsizes = np.logspace(0, 7)
q02 = np.array([0.01, 0.0101, 0.0102, 0.0103, 0.011, 0.0115, 0.0122, 0.0129, 0.0141,
	            0.0181, 0.0191, 0.0237, 0.023, 0.02, 0.0233, 0.0198, 0.027, 0.0263,
	            0.0211, 0.0211, 0.0282])
q02sizes = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
	                 200000, 300000, 350000, 400000, 450000, 500000, 1000000, 2000000])

q03 = np.array([0.01, 0.01, 0.01, 0.01, 0.0101, 0.0103, 0.0116, 0.0144, 0.0249, 
	            0.0294, 0.0331, 0.05, 0.0532, 0.0616, 0.0599, 0.0632, 0.0499, 0.0579, 0.0456])
q03sizes = np.array([1, 2, 5, 10, 20, 50, 100, 200, 20000, 50000, 100000, 200000, 
	                 300000, 350000, 400000, 450000, 500000, 1000000, 2000000])

q04 = np.array([0.01, 0.0101, 0.0101, 0.0103, 0.0105, 0.0107, 0.0118, 0.0358, 0.0327, 0.0535, 0.0575,
                0.068, 0.0678, 0.081, 0.0689, 0.0392, 0.0544, 0.0259])
q04sizes = np.array([10, 20, 50, 100, 200, 500, 1000, 20000, 50000, 100000, 200000, 
                     300000, 350000, 400000, 450000, 500000, 1000000, 2000000])

guesses = [np.max(q02), np.log(q02sizes[q02==np.max(q02)][0]), np.log(2e6/2e4)]
popt, pcov = curve_fit(fitGaussian, np.log(q02sizes), q02, p0=guesses)
plt.plot(allsizes, fitGaussian(np.log(allsizes),popt[0],popt[1],popt[2]), c='g')
plt.scatter(q02sizes, q02, c='g', label='q=0.2', edgecolors='face')
plt.xscale('log')

guesses = [np.max(q03), np.log(q03sizes[q03==np.max(q03)][0]), np.log(2e6/2e4)]
popt, pcov = curve_fit(fitGaussian, np.log(q03sizes), q03, p0=guesses)
plt.plot(allsizes, fitGaussian(np.log(allsizes),popt[0],popt[1],popt[2]), c='r')
plt.scatter(q03sizes, q03, c='r', label='q=0.3', edgecolors='face')
plt.xscale('log')

guesses = [np.max(q04), np.log(q04sizes[q04==np.max(q04)][0]), np.log(2e6/2e4)]
popt, pcov = curve_fit(fitGaussian, np.log(q04sizes), q04, p0=guesses)
plt.plot(allsizes, fitGaussian(np.log(allsizes),popt[0],popt[1],popt[2]), c='b')
plt.scatter(q04sizes, q04, c='b', label='q=0.4', edgecolors='face')
plt.xscale('log')

plt.ylabel('Peak dust to gas ratio')
plt.xlabel('Grain size / microns')
plt.title('Peak dust to gas ratio in R=100au disc')
plt.xlim([1,1e7])
plt.legend(loc=2)
#plt.show()

guesses = [np.max(q04), np.log(q04sizes[q04==np.max(q04)][0]), np.log(2e6/2e4)]
popt, pcov = curve_fit(fitLogNorm, np.log(q04sizes), q04, p0=guesses)
plt.plot(allsizes, fitLogNorm(np.log(allsizes),popt[0],popt[1],popt[2]))
plt.xscale('log')

guesses = [np.max(q03), np.log(q03sizes[q03==np.max(q03)][0]), np.log(2e6/2e4)]
popt, pcov = curve_fit(fitLogNorm, np.log(q03sizes), q03, p0=guesses)
plt.plot(allsizes, fitLogNorm(np.log(allsizes),popt[0],popt[1],popt[2]))
plt.xscale('log')

guesses = [np.max(q02), np.log(q02sizes[q02==np.max(q02)][0]), np.log(2e6/2e4)]
popt, pcov = curve_fit(fitLogNorm, np.log(q02sizes), q02, p0=guesses)
plt.plot(allsizes, fitLogNorm(np.log(allsizes),popt[0],popt[1],popt[2]))
plt.xscale('log')
plt.show()

