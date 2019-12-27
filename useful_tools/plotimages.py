import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as img
import os
import numpy as np

freq = 127
nfolders = 30
matrix = [6,5]

folders = np.array(os.listdir('.'))
folders = folders[folders.argsort()]
#remove anything which isn't a folder
folders = folders[:nfolders]

#reorder folders
folders = folders[[14, 26, 8, 20, 2,
                   15, 27, 9, 21, 3,
                   13, 25, 7, 19, 1,
                   12, 24, 6, 18, 0,
                   16, 28, 10, 22, 4,
                   17, 29, 11, 23, 5]]

images=[]
for i, folder in enumerate(folders):
    try:
       images.append(img.imread('%s/IMAGE_%sGHz.eps' %(folder,freq)))
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
plt.figure(figsize=(8,10.5))
gs1 = gridspec.GridSpec(6,5)
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
plt.subplot(gs1[1]).set_title(r'$\dot{\rm M} = 5\times10^{-8}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[2]).set_title(r'$\dot{\rm M} = 1\times10^{-7}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[3]).set_title(r'$\dot{\rm M} = 5\times10^{-7}$ M$_{\odot}$yr$^{-1}$', fontsize=9)
plt.subplot(gs1[4]).set_title(r'$\dot{\rm M} = 1\times10^{-6}$ M$_{\odot}$yr$^{-1}$', fontsize=9)

plt.subplot(gs1[0]).set_ylabel(r'$a_{\rm max}=10 {\rm \mu}$m', rotation=90, fontsize=11)
plt.subplot(gs1[5]).set_ylabel(r'$a_{\rm max}=1$mm', rotation=90, fontsize=11)
plt.subplot(gs1[10]).set_ylabel(r'$a_{\rm max}=10$cm', rotation=90, fontsize=11)
plt.subplot(gs1[15]).set_ylabel(r'$a_{\rm max}=100$cm', rotation=90, fontsize=11)
plt.subplot(gs1[20]).set_ylabel(r'$a_{\rm max}=a_{\rm frag}$' '\n' 
                                r'$(v_{\rm frag}={\rm 10ms^{-1}})$', rotation=90, fontsize=11)
plt.subplot(gs1[25]).set_ylabel(r'$a_{\rm max}=a_{\rm frag}$' '\n' 
                                r'$(v_{\rm frag}={\rm 30ms^{-1}})$', rotation=90, fontsize=11)

plt.tight_layout()
plt.savefig('oggallery%s.eps'%str(freq))
plt.show()
