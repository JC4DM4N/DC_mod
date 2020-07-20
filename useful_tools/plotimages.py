"""
Plot galleries for the appedix
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as img
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-freq','--freq')
args = parser.parse_args()
freq = str(args.freq)

nfolders = 54
matrix = [6,9]

folders = np.array(os.listdir('../SA_runs_empty/.'))
folders = folders[folders.argsort()]
#dpn't plot amax=high images
folders = np.array([x for x in folders if 'high' not in x])
folders = np.array([x for x in folders if '.log' not in x])
#remove anything which isn't a folder
folders = folders[:nfolders]

#reorder folders
folders = folders[[26,8,38,50,20,2,32,44,14,
                   27,9,39,51,21,3,33,45,15,
                   25,7,37,49,19,1,31,43,13,
                   24,6,36,48,18,0,30,42,12,
                   28,10,40,52,22,4,34,46,16,
                   29,11,41,53,23,5,35,47,17]]

images=[]
for i, folder in enumerate(folders):
    try:
       im = img.imread('../SA_runs_empty/%s/FINALIMAGE_%sGHz.eps' %(folder,freq))
       images.append(im)
    except:
       images.append('noimage')

f, axarr = plt.subplots(matrix[0],matrix[1],figsize=(15,15),squeeze=True)
row=0
column=0
for i, image in enumerate(images):
    if not image=='noimage':
	    axarr[row,column].imshow(image)
    axarr[row,column].set_title(folders[i])
    axarr[row,column].axis('off')
    column+=1
    if column==matrix[1]:
        row+=1
        column=0

plt.tight_layout()
plt.show()

#f, axarr = plt.subplots(matrix[1],matrix[0],figsize=(10,10),squeeze=True,sharex=True,sharey=True)
plt.figure(figsize=(15,10))
gs1 = gridspec.GridSpec(6,9)
gs1.update(wspace=0, hspace=0)
for i, image in enumerate(images):
    ax1 = plt.subplot(gs1[i])
    ax1.set_frame_on(False)
    if not image=='noimage':
        ax1.imshow(image)
    ax1.xaxis.set_ticks_position('none') 
    ax1.yaxis.set_ticks_position('none') 
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')

plt.subplot(gs1[0]).set_title(r'$\dot{\rm M} = 1\times10^{-8}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[1]).set_title(r'$\dot{\rm M} = 1.58\times10^{-8}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[2]).set_title(r'$\dot{\rm M} = 2.81\times10^{-8}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[3]).set_title(r'$\dot{\rm M} = 5\times10^{-8}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[4]).set_title(r'$\dot{\rm M} = 1\times10^{-7}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[5]).set_title(r'$\dot{\rm M} = 1.58\times10^{-7}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[6]).set_title(r'$\dot{\rm M} = 2.81\times10^{-7}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[7]).set_title(r'$\dot{\rm M} = 5\times10^{-7}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[8]).set_title(r'$\dot{\rm M} = 1\times10^{-6}$ M$_{\odot}$yr$^{-1}$', fontsize=9)

plt.subplot(gs1[0]).set_ylabel(r'$a_{\rm max}=10 {\rm \mu}$m', rotation=90, fontsize=11)
plt.subplot(gs1[9]).set_ylabel(r'$a_{\rm max}=1$mm', rotation=90, fontsize=11)
plt.subplot(gs1[18]).set_ylabel(r'$a_{\rm max}=10$cm', rotation=90, fontsize=11)
plt.subplot(gs1[27]).set_ylabel(r'$a_{\rm max}=100$cm', rotation=90, fontsize=11)
plt.subplot(gs1[36]).set_ylabel(r'$a_{\rm max} \approx 1$mm' '\n' 
                                r'$(v_{\rm frag}={\rm 10ms^{-1}})$', rotation=90, fontsize=11)
plt.subplot(gs1[45]).set_ylabel(r'$a_{\rm max} \approx 1$cm' '\n' 
                                r'$(v_{\rm frag}={\rm 30ms^{-1}})$', rotation=90, fontsize=11)

plt.tight_layout()
#plt.savefig('../SA_runs2/gallery%s.eps'%str(freq))
plt.show()
