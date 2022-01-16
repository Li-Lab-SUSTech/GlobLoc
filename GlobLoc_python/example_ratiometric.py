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

fdata = h5.File('test_data_ratiometric.mat','r')

I_model = np.array(fdata.get('psf_model')).astype(np.float32)
imgcenter = 0
data = np.array(fdata.get('data'))
T = np.array(fdata.get('T'))
cor = np.array(fdata.get('cor'))

photon_ratio = np.array(fdata.get('photon_ratio')).flatten()
pz = np.array(fdata.get('pixelsize_z'))[0,0]
px = np.array(fdata.get('pixelsize_x'))[0,0]
Nchannel = I_model.shape[0]
#%% link
dll = localizationlib(usecuda=1)
shared = np.array([1,1,1,1,0])
Nparam = shared.shape[0]
ratioThreshold = 0.999
crosstalk_All = []
P_All = []
CRLB_All = []
LL_All = []
for dat in data:    
    Ph = []
    CRLBh = []
    LLh = []
    for phr in photon_ratio:
        param_ratio = np.ones((Nchannel,Nparam))
        param_ratio[1,3] = phr
        res = dll.loc_ast_dual(dat,cor,shared,imgcenter,T,fittype=2,I_model=I_model,pixelsize_z=pz,param_ratio=param_ratio)
        Ph.append(res[0])
        CRLBh.append(res[1])
        LLh.append(res[2])
    Ph = np.stack(Ph)
    CRLBh = np.stack(CRLBh)
    LLh = np.stack(LLh)
    LLmax_index = np.argmax(LLh,axis=0)
    LLmax = np.max(LLh,axis=0)
    P = np.zeros((Ph.shape[1:]))
    CRLB = np.zeros((CRLBh.shape[1:]))
    LL = np.zeros((LLh.shape[1:]))
    for i,id in enumerate(LLmax_index):
        P[:,i] = Ph[id,:,i]
        CRLB[:,i] = CRLBh[id,:,i]
        LL[i] = LLh[id,i]
    ratioLL = np.expand_dims(LLmax,axis=0)/LLh
    ratioLL = np.sort(ratioLL,axis=0)
    mask = ratioLL[2]<ratioThreshold
    idF = LLmax_index[mask]
    crosstalk = np.zeros(photon_ratio.shape)
    for i in range(0,len(photon_ratio)):
        crosstalk[i]=np.sum(np.int32(idF==i))/len(idF)    
    P = P[:,idF]
    CRLB = CRLB[:,idF]
    LL = LL[idF]
    crosstalk_All.append(crosstalk)
    P_All.append(P)
    CRLB_All.append(CRLB)
    LL_All.append(LL)
crosstalk_All = np.stack(crosstalk_All)

# %%
np.set_printoptions(precision=4,suppress=True)
print(crosstalk_All)
# %%
