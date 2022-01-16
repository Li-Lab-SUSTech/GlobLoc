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

#Load and make cspline PSF
fdata = h5.File('test_data_biplane.mat','r')

I_model = np.array(fdata.get('psf_model')).astype(np.float32)
data = np.array(fdata.get('data'))

#Load transformation matrix
T = np.array(fdata.get('T'))
#Load XY coordinates
cor = np.array(fdata.get('cor'))
imgcenter = 0
Nrep = np.array(fdata.get('Nrep'))[0,0].astype(np.int32)
pz = np.array(fdata.get('pixelsize_z'))[0,0] #um
px = np.array(fdata.get('pixelsize_x'))[0,0] #um

# global fit link all parmeters
dll = localizationlib(usecuda=1)
shared = np.array([1,1,1,1,1])
locres = dll.loc_ast_dual(data,cor,shared,imgcenter,T,fittype=2,I_model=I_model,pixelsize_z=pz)

#global fit link xyz
dll = localizationlib(usecuda=1)
shared = np.array([1,1,1,0,0])
locres_uk = dll.loc_ast_dual(data,cor,shared,imgcenter,T,fittype=2,I_model=I_model,pixelsize_z=pz)

#individual fit
dll = localizationlib(usecuda=1)
locres_ch0 = dll.loc_ast(data[0],fittype=5,I_model=I_model[0],pixelsize_z=pz)
locres_ch1 = dll.loc_ast(data[1],fittype=5,I_model=I_model[1],pixelsize_z=pz)

#position std. and CRLB linkALL
P = locres[0]
CRLB = locres[1]
Nfit = P.shape[1]
Nz = Nfit//Nrep
PM = P.reshape((P.shape[0],Nrep,Nz))
stdM = np.sqrt(CRLB.reshape((CRLB.shape[0],Nrep,Nz)))
pos = PM[0:3]
pos = pos-np.mean(pos,axis=2,keepdims=True)
pos = pos-np.mean(pos,axis=1,keepdims=True)
pos_std = np.std(pos,axis=1)
pos_crlb = np.mean(stdM[0:3],axis=1)

# position std. and CRLB linkXYZ
P = locres_uk[0]
CRLB = locres_uk[1]
Nfit = P.shape[1]
Nz = Nfit//Nrep
PM = P.reshape((P.shape[0],Nrep,Nz))
stdM = np.sqrt(CRLB.reshape((CRLB.shape[0],Nrep,Nz)))
pos_uk = PM[0:3]
pos_uk = pos_uk-np.mean(pos_uk,axis=2,keepdims=True)
pos_uk = pos_uk-np.mean(pos_uk,axis=1,keepdims=True)
# create filter for some bad localization data
mask = np.abs(pos_uk)>10
pos_uk[mask] = 0
pos_std_uk = np.std(pos_uk,axis=1)
pos_crlb_uk = np.mean(stdM[0:3],axis=1)

# %% individual
P0 = locres_ch0[0]
CRLB0 = locres_ch0[1]
P1 = locres_ch1[0]
CRLB1 = locres_ch1[1]
Nfit = P0.shape[1]
Nz = Nfit//Nrep
# transform results back to channel 1 (reference coordiante system)
xyback = np.matmul(np.stack([P1[0],P1[1],np.ones((Nfit,))]).transpose(),np.linalg.inv(T)).transpose()
P1[0:2] = xyback[0:2]

P = (P0[0:5]/CRLB0 + P1[0:5]/CRLB1)/(1/CRLB0+1/CRLB1)
PM = P.reshape((P.shape[0],Nrep,Nz))
pos_i = PM[[0,1,4]]
pos_i = pos_i-np.mean(pos_i,axis=2,keepdims=True)
pos_i = pos_i-np.mean(pos_i,axis=1,keepdims=True)
# create filter for some bad localization data
mask = (np.abs(pos_i[0])>1) | (np.abs(pos_i[1])>1) | (np.abs(pos_i[2])>10)
pos_i[:,mask] = 0
pos_std_i = np.sqrt(np.sum(np.square(pos_i),axis=1)/np.sum(np.int32(~mask),axis=0))

# plot results
Nparams = pos_std_i.shape[0]
zs = np.linspace(-600,600,Nz)
label = ['x','y','z']
pxsz = np.array([px,px,pz])*1e3
fig = plt.figure(figsize=[20,4])
for i in range(0,Nparams):
    ax = fig.add_subplot(1,Nparams,i+1)
    plt.plot(zs,pos_std[i]*pxsz[i],'r.',ms=5)
    plt.plot(zs,pos_crlb[i]*pxsz[i],color=(0,0,0))
    plt.plot(zs,pos_std_uk[i]*pxsz[i],'bo',ms=4,markerfacecolor='none')
    plt.plot(zs,pos_crlb_uk[i]*pxsz[i],color=(0.5,0.5,0.5))
    plt.plot(zs,pos_std_i[i]*pxsz[i],'m^',ms=4,markerfacecolor='none')
    ax.set_title(label[i])
    ax.set_ylabel('precision (nm)')
    ax.set_xlabel('z (nm)')
    if i==0:
        ax.legend(['std','CRLB','std unlink','CRLB unlink','std individual'])

plt.show()
