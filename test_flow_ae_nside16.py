import re
import shutil
import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
import pickle
import time
from tqdm import tqdm
import sys
from modules import model_flow_ae_transformer as model
import platform
import glob
import os
import healpy as hp
import matplotlib.animation as manimation
sys.path.append('database')
import di_marcs as di
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

def pad_collate(batch):
    """
    Pad all sequences with zeros, that will be recognized and not used during training
    
    """
    velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength = zip(*batch)
    
    angles = torch.nn.utils.rnn.pad_sequence(angles, batch_first=True, padding_value=-999)
    stokesi_residual = torch.nn.utils.rnn.pad_sequence(stokesi_residual, batch_first=True, padding_value=0.0)
    T = torch.nn.utils.rnn.pad_sequence(T, batch_first=True, padding_value=0.0)
    stokesi_mn = torch.nn.utils.rnn.pad_sequence(stokesi_mn, batch_first=True, padding_value=0.0)
    wavelength = torch.nn.utils.rnn.pad_sequence(wavelength, batch_first=True, padding_value=0.0)

    velocity = torch.tensor(velocity)
    sini = torch.tensor(sini)
    nangles = torch.tensor(nangles)
    T_max = torch.tensor(T_max)
        
    return velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength

class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_training=None):
        super(Dataset, self).__init__()
        
        print("Reading all data...")

        filename = '/scratch1/aasensio/doppler_imaging/nside16/validation'
        
        with open(f'{filename}_stokes.pk', 'rb') as filehandle:
            stokesi = pickle.load(filehandle)

        with open(f'{filename}_T.pk', 'rb') as filehandle:
            T = pickle.load(filehandle)

        with open(f'{filename}_angles.pk', 'rb') as filehandle:
            angles = pickle.load(filehandle)

        with open(f'{filename}_velocity.pk', 'rb') as filehandle:
            velocity = pickle.load(filehandle)

        with open(f'{filename}_sini.pk', 'rb') as filehandle:
            sini = pickle.load(filehandle)

        if (n_training is None):
            self.n_training_tmp = len(velocity)
        else:
            self.n_training_tmp = n_training
        
        indices = []
        self.n_training = 0
        for i in range(self.n_training_tmp):
            if (np.min(T[i][0]) >= 3000.0):
                indices.append(i)
                self.n_training += 1

        print(' N. training original : {0}'.format(self.n_training_tmp))
        print(' N. training T>3000 K : {0}'.format(self.n_training))

        self.stokesi_mn = [None] * self.n_training
        self.stokesi_residual = [None] * self.n_training
        self.T = [None] * self.n_training
        self.angles = [None] * self.n_training
        self.velocity = [None] * self.n_training
        self.sini = [None] * self.n_training
        self.nangles = [None] * self.n_training
        self.T_max = [None] * self.n_training
        self.wavelength = [None] * self.n_training

        ampl_max = 0.0

        print("Normalizing data...")

        for i in tqdm(range(self.n_training)):
            self.T[i] = torch.tensor(T[i][0].astype('float32'))
            self.T[i] = (self.T[i] - 4700.0) / 500.0
            self.velocity[i] = torch.tensor(velocity[i][0] / 80.0)
            self.sini[i] = torch.tensor(sini[i][0].astype('float32'))
            self.T_max[i] = torch.max(self.T[i])
            
            tmp = stokesi[i][0]
            normalization = np.max(tmp, axis=-1)
            tmp = tmp / normalization[:, None]

            # Add noise            
            tmp += np.random.normal(loc=0, scale=1e-3, size=tmp.shape)

            mn = np.mean(tmp, axis=0)
            tmp = tmp - mn[None, :]

            self.stokesi_mn[i] = torch.tensor(((mn - 0.75) / 0.25).astype('float32'))
            self.stokesi_residual[i] = torch.tensor((tmp / 0.02).astype('float32'))
            self.wavelength[i] = torch.tensor(np.linspace(0.0, 1.0, len(mn)).astype('float32'))
            
            self.angles[i] = torch.tensor(((angles[i][0] - np.pi) / np.pi).astype('float32'))
            self.nangles[i] = torch.tensor(len(self.angles[i]))
        
        self.nangles_max = max(self.nangles)
        
        
    def __getitem__(self, index):

        # Fixed size
        velocity = self.velocity[index]
        sini = self.sini[index]
        T = self.T[index]        
        stokesi_mn = self.stokesi_mn[index]
        nangles = self.nangles[index]
        T_max = self.T_max[index]
        wavelength = self.wavelength[index]

        # Variable size
        angles = self.angles[index]
        stokesi_residual = self.stokesi_residual[index]
                        
        return velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength

    def __len__(self):
        return self.n_training

class Doppler(object):
    def __init__(self, batch_size=64, gpu=0, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size                           

        if (checkpoint is None):
            files = glob.glob('weights_flow_ae_nside16/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)

        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        
        hyper_encoder = checkpoint['hyperparameters_encoder']
        hyper_flow = checkpoint['hyperparameters_flow']

        print("hyper_encoder")
        print(hyper_encoder)
        print("hyper_flow")
        print(hyper_flow)
        
        self.model = model.Model(hyper_encoder, hyper_flow, old=False).to(self.device)        
                        
        self.model.load_state_dict(checkpoint['state_dict'])          

        NSIDE = 2**4
        npix = hp.nside2npix(NSIDE)
        xyz = np.zeros((npix, 3))
        for i in range(npix):
            xyz[i, :] = hp.pix2vec(NSIDE, i, nest=True)

        self.xyz = torch.tensor(xyz.astype('float32')).to(self.device)

        regions = [[5985.1, 5989.0], [6000.1, 6005.0], [6022.0, 6026.0]]        
        self.star = di.DopplerImaging(NSIDE, regions=regions, root_models='database')

    def test(self, remove_plots=True, savefig=False, showplots=False):

        self.dataset = Dataset(n_training=256)

        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)

        self.model.eval()
        loss_avg = 0
        t = tqdm(self.loader)
        correct = 0
        total = 0
        n = 1

        output_all = []
        target_all = []
        latent_all = []
        nangles_all = []
        velocity_all = []
        sini_all = []
        angles_all = []
        stokes_mn_all = []
        stokes_res_all = []
        T_ae_all = []

        with torch.no_grad():
            for batch_idx, (velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength) in enumerate(t):
                                
                velocity, sini, wavelength = velocity.to(self.device), sini.to(self.device), wavelength.to(self.device)
                T, stokesi_mn, nangles = T.to(self.device), stokesi_mn.to(self.device), nangles.to(self.device)
                angles, stokesi_residual, T_max = angles.to(self.device), stokesi_residual.to(self.device), T_max.to(self.device)
                
                latent, output, logprob = self.model.sample(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength, 100, self.xyz)

                T_ae, _ = self.model.ae(T[:, None, :])
   
                output_all.append(output.cpu().numpy() * 500.0 + 4700.0)
                target_all.append(T.cpu().numpy() * 500.0 + 4700.0)
                T_ae_all.append(T_ae.cpu().numpy() * 500.0 + 4700.0)
                latent_all.append(latent.cpu().numpy())
                nangles_all.append(nangles.cpu().numpy())
                velocity_all.append(velocity.cpu().numpy())
                sini_all.append(sini.cpu().numpy())
                angles_all.append(angles.cpu().numpy())
                stokes_mn_all.append(stokesi_mn.cpu().numpy())
                stokes_res_all.append(stokesi_residual.cpu().numpy())

        output = np.concatenate(output_all, axis=0)
        target = np.concatenate(target_all, axis=0)
        T_ae = np.concatenate(T_ae_all, axis=0)
        nangles = np.concatenate(nangles_all, axis=0)
        velocity = np.concatenate(velocity_all, axis=0)
        sini = np.concatenate(sini_all, axis=0)

        output = np.clip(output, 3000.0, 6000.0)
        
        if (showplots):
            fig, ax = pl.subplots(nrows=3, ncols=6, figsize=(15,8))
            for i in range(6):
                median = np.median(output[i, :, :], axis=0)
                mad = np.median(np.abs(output[i, :, :] - median[None, :]), axis=0)
                residual = np.mean(output[i, :, :] - target[i, :][None, :], axis=0)

                if (i == 0):
                    title = 'Median'
                else:
                    title = ''
                pl.axes(ax[0, i])
                hp.orthview(median, nest=True, hold=True, title=title, cmap=pl.cm.inferno)
                pl.text(-2, 1.0, rf'N$_p$={nangles[i]}', fontsize=7.5)

                if (i == 0):
                    title = fr'MAD'
                else:
                    title = ''
                pl.axes(ax[1, i])
                hp.orthview(mad, nest=True, hold=True, title=title, cmap=pl.cm.inferno)
                pl.text(-2, 1.0, rf'v sini={velocity[i]*80*sini[i]:5.2f}', fontsize=7.5)
                                    
                if (i == 0):
                    title = 'Target'
                else:
                    title = ''
                pl.axes(ax[2, i])
                hp.orthview(target[i, :], nest=True, hold=True, title=title, cmap=pl.cm.inferno)
                pl.text(-2, 1.0, rf'i={np.arcsin(sini[i])*180/np.pi:5.2f}', fontsize=7.5)

            if (savefig):
                pl.savefig('figs/flow_inference.png')
            if (remove_plots):
                pl.close()

            rel_error = np.abs(output - target[:, None, :]) / target[:, None, :]
            fig, ax = pl.subplots()
            pl.axes(ax)
            hp.orthview(100.0*np.mean(rel_error, axis=(0, 1)), nest=True, hold=True, title='Relative error [%]')
            if (savefig):
                pl.savefig('figs/flow_relative_error.png')
            if (remove_plots):
                pl.close()

            fig, ax = pl.subplots(nrows=3, ncols=7, figsize=(15,8))
            for i in range(3):
                for j in range(6):                
                    pl.axes(ax[i, j])
                    hp.orthview(output[i, j, :], nest=True, hold=True, title='', cmap=pl.cm.inferno)
                pl.axes(ax[i, -1])
                hp.orthview(target[i, :], nest=True, hold=True, title='Target', cmap=pl.cm.inferno)
            if (savefig):
                pl.savefig('figs/flow_samples.png')
            if (remove_plots):
                pl.close()
        
        # Compute Stokes parameters for number 113, which is the one shown in Fig. 5        
        n_synth = 100
        inclination = np.arcsin(sini[113]) * 180.0 / np.pi
        v = velocity[113] * 80.0
        phase = angles_all[11][3, 0:9] * np.pi + np.pi
        
        obs_mn = stokes_mn_all[11][3, :] * 0.25 + 0.75
        obs_res = stokes_res_all[11][3, 0:9, :] * 0.02
        obs = obs_mn[None, :] + obs_res
        
        stokesi = []
        for i in tqdm(range(n_synth)):
            wl, tmp = self.synthesize(output[113, i, :], inclination, phase, v)
            stokesi.append(tmp)

        # Compute the spectra in the median solution
        wl, stokes_median = self.synthesize(np.median(output[113, :, :], axis=0), inclination, phase, v)
        
        wl, stokes_ae = self.synthesize(T_ae[113, :], inclination, phase, v)

        outfile = f'validation.pk'
        
        with open(outfile, 'wb') as handle:
            pickle.dump([output, target, sini, velocity, nangles, stokesi, stokes_median, stokes_ae, obs, phase], handle)
        
        return output, target, sini, velocity, nangles

    def to_torch(self, velocity_in, sini_in, angles_in, T_max_in, stokes_in):
        sini = torch.tensor([sini_in.astype('float32')])

        velocity = torch.tensor([velocity_in.astype('float32')]) / 80.0

        angles = torch.tensor(((angles_in - np.pi) / np.pi).astype('float32')).unsqueeze(0)
        nangles = torch.tensor([angles.shape[1]])

        T_max = (T_max_in - 4700.0) / 500.0
        T_max = torch.tensor([np.float32(T_max)])

        normalization = np.max(stokes_in, axis=-1)
        stokes = stokes_in / normalization[:, None]
        observations = stokes
        mn = np.mean(stokes, axis=0)
        stokes = stokes - mn[None, :]

        mn = (mn - 0.75) / 0.25        
        mn = torch.tensor(mn.astype('float32')).unsqueeze(0)        

        stokes = stokes / 0.02
        stokes = torch.tensor(stokes.astype('float32')).unsqueeze(0)

        return velocity, sini, angles, nangles, T_max, mn, stokes, observations

    def synthesize(self, T, inclination, phases, velocity):

        n_phases = len(phases)

        los = np.zeros((n_phases, 2))
        for i in range(n_phases):
            los[i, :] = np.array([inclination * np.pi / 180.0, phases[i]])
                                            
        R_spectro = 65000.0
        v_spectro = 3e5 / R_spectro
        v_macro = 4.0
        v_total = np.sqrt(v_spectro**2 + v_macro**2)
        R = 3e5 / v_total

        out_lambda = np.array([5986.54731, 5986.58131, 5986.61631, 5986.65031, 5986.68531,
            5986.71931, 5986.75431, 5986.78831, 5986.82431, 5986.85731,
            5986.89331, 5986.92631, 5986.96231, 5986.99631, 5987.02931,
            5987.06531, 5987.09831, 5987.13431, 5987.16731, 5987.20331,
            5987.23531, 5987.27231, 5987.30531, 5987.34131, 5987.37431,
            5987.41031, 5987.44431, 5987.47931, 5987.51331, 5987.54831,
            5987.58231, 6001.98222, 6002.01522, 6002.04822, 6002.08422,
            6002.11722, 6002.15122, 6002.18422, 6002.21722, 6002.25222,
            6002.28622, 6002.31822, 6002.35222, 6002.38622, 6002.41922,
            6002.45522, 6002.48822, 6002.52122, 6002.55422, 6002.59022,
            6002.62322, 6002.65622, 6002.69022, 6002.72322, 6002.75822,
            6002.79222, 6002.82522, 6002.85922, 6002.89222, 6002.92522,
            6002.96022, 6002.99422, 6003.02622, 6003.06022, 6003.09422,
            6003.12922, 6003.16222, 6003.19622, 6003.22922, 6003.26222,
            6003.29722, 6003.33122, 6003.36422, 6003.39822, 6003.43122,
            6003.46422, 6003.49922, 6003.53322, 6003.56722, 6003.59922,
            6023.48409, 6023.51409, 6023.54709, 6023.58009, 6023.61309,
            6023.64409, 6023.67709, 6023.71009, 6023.74309, 6023.77409,
            6023.80709, 6023.84009, 6023.87309, 6023.90609, 6023.93709,
            6023.97009, 6024.00309, 6024.03609, 6024.06909, 6024.10009,
            6024.13309, 6024.16609, 6024.19909, 6024.23209, 6024.26309,
            6024.29609, 6024.33009, 6024.36209, 6024.39309, 6024.42609,
            6024.46009, 6024.49209, 6024.52509, 6024.55609, 6024.59009,
            6024.62309, 6024.65509])

        stokesi = self.star.compute_stellar_spectrum(T, los, omega=velocity, resolution=R, reinterpolate_lambda=out_lambda)

        normalization = np.max(stokesi, axis=-1)
        stokesi = stokesi / normalization[:, None]

        wl = [None] * 3
        wl[0] = out_lambda[0:31]
        wl[1] = out_lambda[31:80]
        wl[2] = out_lambda[80:]
        stokes = [None] * 3
        stokes[0] = stokesi[:, 0:31]
        stokes[1] = stokesi[:, 31:80]
        stokes[2] = stokesi[:, 80:]

        return wl, stokes

    def variability(self):
                
        torch.set_default_tensor_type(torch.FloatTensor)
        self.model.eval()

        # Generate an artificial star with spots
        self.NSIDE = 2**4
        self.npix = hp.nside2npix(self.NSIDE)
        
        self.minT = 3200.0
        self.maxT = 5800.0
                        
        nspots = 4
        tstar = 5500.0
        smooth = 0.1

        T = tstar * np.ones(self.npix)

        theta = np.array([20.0, 90.0, 90.0, 150.0]) * np.pi/180.0
        phi = np.array([0.0, -90.0, 90.0, 0.0]) * np.pi/180.0

        for i in range(nspots):
            
            vec = hp.ang2vec(theta[i], phi[i])
            
            radius = 0.2
                        
            px = hp.query_disc(self.NSIDE, vec, radius, nest=False)
            
            T[px] = 3500.0
            
        # The smoothing works only on RINGED
        T = hp.sphtfunc.smoothing(T, sigma=smooth, verbose=False)

        T = hp.pixelfunc.reorder(T, inp='RING', out='NESTED')
        
        inclinations = np.linspace(10, 85, 4)

        t = tqdm(inclinations)

        for i, inclination in enumerate(t):
            nphases = 15
            sini = np.sin(inclination * np.pi / 180.0)
            velocity = np.float64(30.0)
            vsini = velocity * sini            
            T_max = 0.0
            
            phase = np.arange(nphases) / nphases * 2.0 * np.pi

            wl, tmp = self.synthesize(T, inclination, phase, velocity)

            tmp = np.concatenate(tmp, axis=1)
            tmp = tmp + np.random.normal(loc=0, scale=1e-3, size=tmp.shape)
            
            if (i == 0):
                velocity_t, sini_t, angles_t, nangles_t, T_max_t, mn_t, stokes_t, observations_t = self.to_torch(velocity, sini, phase, T_max, tmp) 
                wavelength_t = torch.linspace(0.0, 1.0, 117).unsqueeze(0)
            else:
                velocity_t2, sini_t2, angles_t2, nangles_t2, T_max_t2, mn_t2, stokes_t2, observations_t2 = self.to_torch(velocity, sini, phase, T_max, tmp)
                velocity_t = torch.cat([velocity_t, velocity_t2])
                sini_t = torch.cat([sini_t, sini_t2])
                angles_t = torch.cat([angles_t, angles_t2], dim=0)
                nangles_t = torch.cat([nangles_t, nangles_t2])
                mn_t = torch.cat([mn_t, mn_t2], dim=0)
                stokes_t = torch.cat([stokes_t, stokes_t2], dim=0)

                wavelength_t2 = torch.linspace(0.0, 1.0, 117).unsqueeze(0)
                wavelength_t = torch.cat([wavelength_t, wavelength_t2], dim=0)

            t.set_postfix(v=velocity, inc=inclination)
        
        with torch.no_grad():
                    
            velocity, sini, wavelength = velocity_t.to(self.device), sini_t.to(self.device), wavelength_t.to(self.device)
            mn, nangles = mn_t.to(self.device), nangles_t.to(self.device)
            angles, stokes = angles_t.to(self.device), stokes_t.to(self.device)
                
            start = time.time()
        
            latent, output, logprob = self.model.sample(velocity, sini, nangles, angles, mn, stokes, None, wavelength, 500, self.xyz)
            print(f'Elapsed time : {time.time() - start}')

        output = output.cpu().numpy() * 500.0 + 4700.0    

        outfile = f'variability_sini.pk'
        
        with open(outfile, 'wb') as handle:
            pickle.dump([T, output, inclinations], handle) 
        
    def iipeg(self, savefig=False, step=1, doplots=False):
                
        torch.set_default_tensor_type(torch.FloatTensor)
        self.model.eval()

        f = open('iipeg/IIPeg_StokesI_2013_map.obs', 'r')
        for i in range(31):
            f.readline()

        phase = np.zeros(12)
        stokes = np.zeros((12, 117))

        for i in range(12):
            tmp = f.readline()
            phase[i] = float(tmp.split(',')[0].split('=')[1]) * 2.0 * np.pi
            f.readline()
            out = []
            for j in range(24):                
                tmp = f.readline().split(',')[0:-1]                
                tmp = [float(t) for t in tmp]
                out.extend(tmp)
            stokes[i, :] = np.array(out)
        
        inclination = 60.0
        sini = np.sin(inclination * np.pi / 180.0)
        vsini = 23.0
        velocity = vsini / sini        
        T_max = 0.0
        v = vsini / sini

        # Use only some observations
        phase = phase[::step]
        stokes = stokes[::step, :]

        velocity, sini, angles, nangles, T_max, mn, stokes, observations = self.to_torch(velocity, sini, phase, T_max, stokes)    
                    
        wavelength = torch.linspace(0.0, 1.0, 117).unsqueeze(0)       
        
        with torch.no_grad():
                    
            velocity, sini, wavelength = velocity.to(self.device), sini.to(self.device), wavelength.to(self.device)
            mn, nangles = mn.to(self.device), nangles.to(self.device)
            angles, stokes, T_max = angles.to(self.device), stokes.to(self.device), T_max.to(self.device)
                
            start = time.time()

            # Only save 500 but extract 2500 to better sample the posterior and find a better MAP
            latent, output, logprob = self.model.sample(velocity, sini, nangles, angles, mn, stokes, T_max, wavelength, 500, self.xyz)
            latent2, output2, logprob2 = self.model.sample(velocity, sini, nangles, angles, mn, stokes, T_max, wavelength, 2500, self.xyz)            
            latent, output_hd, logprob_hd = self.model.sample(velocity, sini, nangles, angles, mn, stokes, T_max, wavelength, 500, self.xyz)
            print(f'Elapsed time : {time.time() - start}')

        output = output.squeeze().cpu().numpy() * 500.0 + 4700.0
        output2 = output2.squeeze().cpu().numpy() * 500.0 + 4700.0
        output_hd = output_hd.squeeze().cpu().numpy() * 500.0 + 4700.0
        logprob = logprob.squeeze().cpu().numpy()
        logprob2 = logprob2.squeeze().cpu().numpy()

        median = np.median(output, axis=0)
        mad = np.std(output, axis=0)

        median_hd = np.median(output_hd, axis=0)
        mad_hd = np.std(output_hd, axis=0)

        for param in self.model.parameters():
            param.requires_grad = False

        # _, best, _ = self.model.find_MAP(velocity, sini, nangles, angles, mn, stokes, T_max, wavelength, self.device, self.xyz_hd)
        # best = best.detach().cpu().numpy() * 500.0 + 4700.0

        
        obs = [None] * 3
        obs[0] = observations[:, 0:31]
        obs[1] = observations[:, 31:80]
        obs[2] = observations[:, 80:]

        origin = [5980, 6000, 6020]
        labels = ['5987.1', '6003.0', '6024.1']

        # Compute Stokes parameters
        n_synth = 100
        stokesi = []
        for i in tqdm(range(n_synth)):
            wl, tmp = self.synthesize(output[i, :], inclination, phase, v)
            stokesi.append(tmp)

        # Compute the spectra in the median solution
        wl, stokes_median = self.synthesize(median, inclination, phase, v)

        # MAP
        ind = np.argmin(logprob2)
        best = output2[ind, :]
        wl, stokes_map = self.synthesize(best, inclination, phase, v)
        
        # Compute the median spectra for all temperatures
        median_stokes = [None] * 3
        for i in range(3):
            tmp = np.array([stokesi[j][i] for j in range(n_synth)])
            median_stokes[i] = np.median(tmp, axis=0)
        
        if (doplots):
        
            fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(12,5))
            pl.axes(ax[0])
            hp.orthview(median_hd, nest=True, hold=True, title='Median', cmap=pl.cm.inferno)
            hp.visufunc.graticule()

            pl.axes(ax[1])
            hp.orthview(mad_hd, nest=True, hold=True, title='MAD', cmap=pl.cm.inferno)
            hp.visufunc.graticule()

            pl.axes(ax[2])
            hp.orthview(best, nest=True, hold=True, title='MAP', cmap=pl.cm.inferno)
            hp.visufunc.graticule()

            if (savefig):
                pl.savefig('figs/iipeg_median.png')

            fig, ax = pl.subplots(nrows=2, ncols=5, figsize=(12,8))
            for i in range(5):
                pl.axes(ax[0,i])
                hp.orthview(median_hd, nest=True, hold=True, title=f'Phase {i*0.2:3.1f}', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0))
                hp.visufunc.graticule()
                pl.axes(ax[1,i])
                hp.orthview(mad_hd, nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0))
                hp.visufunc.graticule()
            
            if (savefig):
                pl.savefig('figs/iipeg_median_orth.png')

            fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(12,8))
            for i in range(9):
                pl.axes(ax.flat[i])
                hp.orthview(output_hd[i, :], nest=True, hold=True, cmap=pl.cm.inferno, title='')
                hp.visufunc.graticule()

            if (savefig):
                pl.savefig('figs/iipeg_samples.png')

        

            fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,18))
            for region in range(3):
                for i in range(len(phase)):
                    ind = len(phase) - 1 - i
                    ax[region].errorbar(wl[region] - origin[region], obs[region][ind, :] + i*0.2, yerr=1e-3, color='C0')
                    for j in range(n_synth):
                        ax[region].plot(wl[region] - origin[region], stokesi[j][region][ind, :] + i*0.2, color='C1', alpha=0.05)            
                    ax[region].plot(wl[region] - origin[region], stokes_median[region][ind, :] + i*0.2, color='C2', alpha=1.0)
                    ax[region].plot(wl[region] - origin[region], median_stokes[region][ind, :] + i*0.2, color='C3', alpha=1.0)
                    ax[region].plot(wl[region] - origin[region], stokes_map[region][ind, :] + i*0.2, color='C4', alpha=1.0)
                    if (region == 0):
                        ax[region].text(6.52, 1.03+i*0.2, f'{phase[ind]/(2.0*np.pi):5.3f}')

            for region in range(3):
                ax[region].set_xlabel(fr'$\lambda+${origin[region]} [$\AA$]')
                ax[region].set_title(fr'Fe I {labels[region]} $\AA$')

            if (savefig):
                pl.savefig('figs/iipeg_spectra.png')

            pl.show()

            if (savefig):
                metadata = dict(title='IIPeg', artist='Matplotlib',
                    comment='Movie support!')
                writer = manimation.FFMpegWriter(fps=15, metadata=metadata, extra_args=['-vcodec', 'libx264'])

                fig, ax = pl.subplots()
                pl.axis('off')
                fig.subplots_adjust(left=0, bottom=0.1, right=1, top=1, wspace=None, hspace=None)
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])

                map_min = np.min(output)
                map_max = np.max(output)

                with writer.saving(fig, "figs/iipeg.mp4", 200):
                    for i in tqdm(range(100)):
                        pl.clf()
                        pl.axes(ax)
                        hp.orthview(output[i, :], hold=True, cbar=True, title='', nest=True, min=map_min, max=map_max, cmap=pl.cm.inferno)
                        writer.grab_frame()

                pl.close()

        outfile = f'iipeg_step{step}.pk'
        
        # with open(outfile, 'wb') as handle:
            # pickle.dump([output, output_hd, stokesi, stokes_median, median_stokes, wl, phase, obs], handle)
        

if (__name__ == '__main__'):
    pl.close('all')
    
    # network = Doppler(batch_size=10, gpu=3, checkpoint='weights_flow_ae_nside16/2021-06-01-22:19')  # <- in the paper now
    # network = Doppler(batch_size=10, gpu=3, checkpoint='weights_flow_ae_nside16/2021-06-07-09:00')

    network = Doppler(batch_size=10, gpu=3, checkpoint='weights_flow_ae_nside16/2021-06-08-16:23') # Now in the paper

    #---------------
    # Validation
    #---------------
    output, target, sini, velocity, nangles = network.test(remove_plots=False, savefig=False)

    #---------------
    # II Peg
    #---------------
    # network.iipeg(savefig=False, step=1, doplots=False)
    # network.iipeg(savefig=False, step=2, doplots=False)
    # network.iipeg(savefig=False, step=4, doplots=False)

    #---------------
    # Variability
    #---------------
    # network.variability()
