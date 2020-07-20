"""
Plot how the dust concentration factor, epsilon, varies with stokes
    number in our semi-analytic prescription.
"""

import numpy as np
import matplotlib.pyplot as plt

St = np.arange(0.01,200)
d = 5
eps = 1 + 2*d/(St + 1/St) - St/200.

plt.plot(St,eps)
plt.xlabel('St')
plt.ylabel(r'$\epsilon$')
plt.ylim([0,6])
#plt.xscale('log')
#plt.savefig('epsvsSt.png')
#plt.show()

rads = np.linspace(1,100)
grainsize = np.logspace(-1,6,num=10)
St=[]
for a in grainsize:
    St.append(a*rads**1.5)

