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

fdata = h5.File('test_data_gauss.mat','r')
data = np.array(fdata.get('data'))
T = np.array(fdata.get('T'))
cor = np.array(fdata.get('cor'))
imgcenter = 0
Nchannel = data.shape[0]
#%% link
shared = np.array([1,1,1,1,1])
Nparam = shared.shape[0]
paramshift = np.zeros((Nchannel,Nparam))
paramshift[1,-1] = 0.3
dll = localizationlib(usecuda=1)
locres = dll.loc_ast_dual(data,cor,shared,imgcenter,T,fittype=1,param_shift=paramshift)
P = locres[0]
CRLB = locres[1]
LL = locres[2]
plt.hist(P[4],bins=50,edgecolor='k')
plt.xlabel('PSF sigma')
plt.ylabel('count')
plt.show()
#%% individual
dll = localizationlib(usecuda=1)
locres_ch0 = dll.loc_ast(data[0],fittype=2)
locres_ch1 = dll.loc_ast(data[1],fittype=2)
P0 = locres_ch0[0]
P1 = locres_ch1[0]

plt.hist(P0[4],bins=50,edgecolor='k')
plt.hist(P1[4],bins=50,edgecolor='k')
plt.xlabel('PSF sigma')
plt.ylabel('count')
plt.legend(['channel 1','channel 2'])
plt.show()
# %%
