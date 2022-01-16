''' Copyright (c) 2022 Li Lab, Southern University of Science and Technology, Shenzhen & Ries Lab, European Molecular Biology Laboratory, Heidelberg.
 author: Sheng Liu
 email: sheng.liu@embl.de
 date: 2022.1.10
 Tested with CUDA 11.3 (Express installation) and python 3.8
'''

import numpy as np
import matplotlib.pyplot as plt
from loclib import localizationlib
import h5py as h5

#Load data IAB
fdata = h5.File('test_data_4pi.mat','r')
I_model = np.array(fdata.get('I')).astype(np.float32)
A_model = np.array(fdata.get('A')).astype(np.float32)
B_model = np.array(fdata.get('B')).astype(np.float32)
data = np.array(fdata.get('data'))
zT = np.array(fdata.get('zT'))[0,0]
imgcenter = 0
Nrep = np.array(fdata.get('Nrep'))[0,0].astype(np.int32)
pz = np.array(fdata.get('pixelsize_z'))[0,0]
px = np.array(fdata.get('pixelsize_x'))[0,0]

#%% link
dll = localizationlib(usecuda=1)
shared = np.array([1,1,1,1,1,1])
locres = dll.loc_4pi(data,I_model,A_model,B_model,shared,pz)

#%% unlink
dll = localizationlib(usecuda=1)
shared = np.array([1,1,0,0,1,1])
locres_uk = dll.loc_4pi(data,I_model,A_model,B_model,shared,pz)

# %% link
P = locres[0]
CRLB = locres[1]
Nfit = P.shape[1]
Nz = Nfit//Nrep
PM = P.reshape((P.shape[0],Nrep,Nz))
stdM = np.sqrt(CRLB.reshape((CRLB.shape[0],Nrep,Nz)))
phif = np.unwrap(PM[-2])
pos = np.stack([PM[0],PM[1],PM[-3],phif])
pos = pos-np.mean(pos,axis=2,keepdims=True)
pos = pos-np.mean(pos,axis=1,keepdims=True)
pos_std = np.std(pos,axis=1)
pos_crlb = np.mean(stdM[[0,1,-2,-1]],axis=1)

# %% unlink
P = locres_uk[0]
CRLB = locres_uk[1]
Nfit = P.shape[1]
Nz = Nfit//Nrep
PM = P.reshape((P.shape[0],Nrep,Nz))
stdM = np.sqrt(CRLB.reshape((CRLB.shape[0],Nrep,Nz)))
phif = np.unwrap(PM[-2])
pos_uk = np.stack([PM[0],PM[1],PM[-3],phif])
pos_uk = pos_uk-np.mean(pos_uk,axis=2,keepdims=True)
pos_uk = pos_uk-np.mean(pos_uk,axis=1,keepdims=True)
pos_std_uk = np.std(pos_uk,axis=1)
pos_crlb_uk = np.mean(stdM[[0,1,-2,-1]],axis=1)

# plot result
Nparams = pos_std.shape[0]
zs = np.linspace(-500,500,Nz)
label = ['x','y','zast','z']
pxsz = np.array([px,px,pz,zT/2/np.pi*pz])*1e3
fig = plt.figure(figsize=[20,4])
for i in range(0,Nparams):
    ax = fig.add_subplot(1,4,i+1)
    plt.plot(zs,pos_std[i]*pxsz[i],'r.',ms=5)
    plt.plot(zs,pos_crlb[i]*pxsz[i],color=(0,0,0))
    plt.plot(zs,pos_std_uk[i]*pxsz[i],'bo',ms=4,markerfacecolor='none')
    plt.plot(zs,pos_crlb_uk[i]*pxsz[i],color=(0.5,0.5,0.5))
    ax.set_title(label[i])
    ax.set_ylabel('precision (nm)')
    ax.set_xlabel('z (nm)')
    if i==0:
        ax.legend(['std','CRLB','std unlink','CRLB unlink'])
plt.show()