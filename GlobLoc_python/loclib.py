
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
from psf2cspline import psf2cspline_np
from tqdm import tqdm
import os

class localizationlib:
    def __init__(self,usecuda=0):
        thispath = os.path.dirname(os.path.abspath(__file__))
        dllpath_cpu_astM = thispath+'/source/CPUmleFit_LM_MultiChannel.dll'
        dllpath_gpu_astM = thispath+'/source/GPUmleFit_LM_MultiChannel.dll'
        dllpath_cpu_4pi = thispath+'/source/CPUmleFit_LM_4Pi.dll'
        dllpath_gpu_4pi = thispath+'/source/GPUmleFit_LM_4Pi.dll'
        dllpath_cpu_ast = thispath+'/source/CPUmleFit_LM.dll'
        dllpath_gpu_ast = thispath+'/source/GPUmleFit_LM.dll'

        #Load the DLL file
        lib_cpu_astM = ctypes.CDLL(dllpath_cpu_astM,winmode=0)
        lib_gpu_astM = ctypes.CDLL(dllpath_gpu_astM,winmode=0)
        lib_cpu_4pi = ctypes.CDLL(dllpath_cpu_4pi,winmode=0)
        lib_gpu_4pi = ctypes.CDLL(dllpath_gpu_4pi,winmode=0)
        lib_cpu_ast = ctypes.CDLL(dllpath_cpu_ast,winmode=0)
        lib_gpu_ast = ctypes.CDLL(dllpath_gpu_ast,winmode=0)

        #choose GPU :1 else CPU
        if usecuda==1:
            self._mleFit_MultiChannel = lib_gpu_astM.GPUmleFit_MultiChannel
            self._mleFit_4Pi = lib_gpu_4pi.GPUmleFit_LM_4Pi
            self._mleFit = lib_gpu_ast.GPUmleFit_LM
        else:
            self._mleFit_MultiChannel = lib_cpu_astM.CPUmleFit_MultiChannel
            self._mleFit_4Pi = lib_cpu_4pi.CPUmleFit_LM_4Pi
            self._mleFit = lib_cpu_ast.CPUmleFit_LM
        
        
        self._mleFit_4Pi.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctl.ndpointer(np.int32),   # shared
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff
            ctl.ndpointer(np.float32), # dTAll
            ctl.ndpointer(np.float32), # phiA
            ctl.ndpointer(np.float32), # init_z
            ctl.ndpointer(np.float32), # initphase
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL
        ]

        self._mleFit_MultiChannel.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctypes.c_int32,          # fittype: 1 Gauss; 2 spline
            ctl.ndpointer(np.int32),   # shared
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff
            ctl.ndpointer(np.float32), # dTAll
            ctl.ndpointer(np.float32), # varim
            ctl.ndpointer(np.float32), # init_z
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL
        ]

        self._mleFit.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctypes.c_int32,          # fittype: 1 fixed sigma; 2 free sigma; 4 sigmax and sigmay; 5 spline
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff or PSF sigma
            ctl.ndpointer(np.float32), # varim
            ctypes.c_float,             # init_z
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL
        ]

    # Global Fit multichannel
    def loc_ast_dual(self,psf_data,cor,shared,imgcenter,T,fittype,I_model=None,pixelsize_z=0.02,param_ratio=None,param_shift=None):

        Nfit = cor.shape[1]
        Nchannel = cor.shape[0]        
        Nparam = 5
        fittype = np.int32(fittype)
        if fittype==2:
            Iall = []
            Imd = I_model

            #calculate spline coefficients
            pbar = tqdm()
            for i in range(Nchannel):
                pbar.set_description("calculating spline coefficients")
                coeff = psf2cspline_np(Imd[i])
                Iall.append(coeff)
                pbar.update(1)
            pbar.refresh()
            pbar.close()

            Iall = np.stack(Iall).astype(np.float32)
            splinesize = np.array(np.flip(Iall.shape))
            ccz = Iall.shape[-3]//2
            initparam = np.array([-2,-1,0,1,2])*0.15/pixelsize_z+ccz #initial zstart
            paramstart = np.repeat(np.expand_dims(initparam,axis=1),Nfit,axis=1).astype(np.float32)
        else:
            initparam = np.array([1.5]).astype(np.float32)
            Iall = initparam
            paramstart = np.repeat(np.expand_dims(initparam,axis=1),Nfit,axis=1)
            splinesize = np.array([0])

        data = psf_data
        rsz = psf_data.shape[-1]
        bxsz = np.min((rsz,20))
        data = data[:,:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)

        # define shifts between channels
        cor1 = np.concatenate((cor[0],np.ones((Nfit,1))),axis=1)
        T1 = np.stack([np.eye(3),T])
        dx1 = np.zeros((Nfit,Nchannel))
        dy1 = np.zeros((Nfit,Nchannel))
        # parameter shifts between channels
        for i in range(Nchannel):
            cor2 = np.matmul(cor1-imgcenter,T1[i])+imgcenter
            dx1[:,i] = cor2[:,0]-cor[i][:,0]
            dy1[:,i] = cor2[:,1]-cor[i][:,1]
        # transformation between channels
        dTS = np.zeros((Nfit,Nchannel*2,Nparam))
        dTS[:,0:Nchannel,0] = dx1
        dTS[:,0:Nchannel,1] = dy1
        # define scaling between channels
        if param_shift is not None:
            dTS[:,0:Nchannel,:] = dTS[:,0:Nchannel,:] + np.expand_dims(param_shift,axis=0)
        if param_ratio is None:
            dTS[:,Nchannel:]=1
        else:
            dTS[:,Nchannel:]=np.expand_dims(param_ratio,axis=0)
        dTS = dTS.astype(np.float32)

        #Linking parameters
        sharedA = np.repeat(np.expand_dims(shared,axis=0),Nfit,axis = 0).astype(np.int32)

        datasize = np.array(np.flip(data.shape))
        
        varim = np.array((0)).astype(np.float32)

        Pk = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        
        iterations = np.int32(100)
        P = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10

       
        pbar = tqdm()
        for param in paramstart:
            pbar.set_description("localization")
            self._mleFit_MultiChannel(data,fittype,sharedA,iterations,Iall,dTS, varim,param,datasize,
                                      splinesize,Pk,CRLBk,LLk)
            # copy only everything if LLk increases by more than rounding error
            mask = (LLk-LL)>1e-4
            LL[mask] = LLk[mask]
            P[:,mask] = Pk[:,mask]
            CRLB[:,mask] = CRLBk[:,mask]
            pbar.update(1)
        pbar.refresh()
        pbar.close()

        return P, CRLB, LL, Iall

    # Global Fit 4pi
    def loc_4pi(self,psf_data,I_model,A_model,B_model,shared,pixelsize_z):
        Nchannel = psf_data.shape[0]
        Nfit = psf_data.shape[-3]
        Nparam = 6

        # calculate IAB model
        pbar = tqdm()
        pbar.set_description("calculating spline coefficients")    
        Ii = I_model/2
        Ai = A_model/2
        Bi = B_model/2
        IAB = [psf2cspline_np(Ai),psf2cspline_np(Bi),psf2cspline_np(Ii)]  
        IAB = np.stack(IAB)
        pbar.update(1)
        pbar.close()

        IABall = np.repeat(np.expand_dims(IAB,axis=0),Nchannel,axis=0).astype(np.float32)

        rsz = psf_data.shape[-1]
        bxsz = np.min((rsz,20))
        data = psf_data[:,:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)

        dTS = np.zeros((Nfit,Nchannel,Nparam),dtype = np.float32)

        sharedA = np.repeat(np.expand_dims(shared,axis=0),Nfit,axis = 0).astype(np.int32)

        phic = np.array([0,np.pi/2,np.pi,3*np.pi/2])
        phiA = np.repeat(np.expand_dims(phic,axis=0),Nfit,axis = 0).astype(np.float32)
        
        ccz = IABall.shape[-3]//2
        initz = np.array([-1,1])*0.15/pixelsize_z+ccz
        zstart = np.repeat(np.expand_dims(initz,axis=1),Nfit,axis=1).astype(np.float32)


        initphi = np.array([0,np.pi])
        phi_start = np.repeat(np.expand_dims(initphi,axis=1),Nfit,axis=1).astype(np.float32)

        datasize = np.array(np.flip(data.shape))
        splinesize = np.array(np.flip(IABall.shape))
  
        Pk = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        iterations = np.int32(100)
        P = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10

        maxN = 3000
        Nf = np.ceil(Nfit/maxN).astype(np.int32)

        vec = np.linspace(0,Nf*maxN,Nf+1).astype(np.int32)
        vec[-1] = Nfit
        pbar = tqdm()
        for z0 in zstart:
            for phi0 in phi_start:
                pbar.set_description("localization")
                for i in range(Nf):
                    nfit = vec[i+1]-vec[i]
                    ph = np.zeros((Nparam+1+(Nchannel-1)*(Nparam-np.sum(shared)),nfit)).astype(np.float32)
                    ch = np.zeros((Nparam+(Nchannel-1)*(Nparam-np.sum(shared)),nfit)).astype(np.float32)
                    Lh = np.zeros((nfit)).astype(np.float32)
                    datai = np.copy(data[:,vec[i]:vec[i+1]])
                    sharedi = np.copy(sharedA[vec[i]:vec[i+1]])
                    dts = np.copy(dTS[vec[i]:vec[i+1]])
                    phiAi = np.copy(phiA[vec[i]:vec[i+1]])
                    z0i = np.copy(z0[vec[i]:vec[i+1]])
                    phi0i = np.copy(phi0[vec[i]:vec[i+1]])
                    datsz = np.array(np.flip(datai.shape))
                    self._mleFit_4Pi(datai,sharedi,iterations,IABall,dts,phiAi,z0i,phi0i,datsz,splinesize,ph,ch,Lh)
                    Pk[:,vec[i]:vec[i+1]] = ph
                    CRLBk[:,vec[i]:vec[i+1]] = ch
                    LLk[vec[i]:vec[i+1]] = Lh
                mask = (LLk-LL)>1e-4
                LL[mask] = LLk[mask]
                P[:,mask] = Pk[:,mask]
                CRLB[:,mask] = CRLBk[:,mask]
                pbar.update(1)
        pbar.refresh()
        pbar.close()

        return P, CRLB, LL, IABall

    # Individual fit
    def loc_ast(self,psf_data,fittype,I_model=None,pixelsize_z=None):

        Nfit = psf_data.shape[-3]
        fittype = np.int32(fittype)
        if fittype==5:     
            Imd = I_model

            #calculating spline coefficients
            pbar = tqdm()                    
            pbar.set_description("calculating spline coefficients")       
            coeff = psf2cspline_np(Imd)
            pbar.update(1)            
            pbar.refresh()
            pbar.close()
            coeff = coeff.astype(np.float32)
            splinesize = np.array(np.flip(coeff.shape))

            ccz = coeff.shape[-3]//2
            initparam = np.array([-3,-2,-1,0,1,2,3])*0.3/pixelsize_z+ccz
            paramstart = initparam.astype(np.float32)
        else:
            paramstart = np.array([1.5]).astype(np.float32)
            coeff = paramstart
            splinesize = np.array([0])

        if fittype==1:
            Nparam = 4
        elif fittype==4:
            Nparam = 6
        else:
            Nparam = 5

        data = psf_data
        rsz = psf_data.shape[-1]
        bxsz = np.min((rsz,20))
        data = data[:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)

        datasize = np.array(np.flip(data.shape))
        
        varim = np.array((0)).astype(np.float32)
        Pk = np.zeros((Nparam+1,Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam,Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        
        iterations = np.int32(100)
        P = np.zeros((Nparam+1,Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam,Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10
       
        pbar = tqdm()
        for param in paramstart:
            pbar.set_description("localization")
            self._mleFit(data,fittype,iterations,coeff,varim,param,datasize,splinesize,Pk,CRLBk,LLk)
            # copy only everything if LLk increases by more than rounding error
            mask = (LLk-LL)>1e-4
            LL[mask] = LLk[mask]
            P[:,mask] = Pk[:,mask]
            CRLB[:,mask] = CRLBk[:,mask]
            pbar.update(1)
        pbar.refresh()
        pbar.close()

        return P, CRLB, LL, coeff