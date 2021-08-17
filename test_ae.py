import shutil
import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
import pickle
from torch.autograd import Variable
import time
from tqdm import tqdm
import sys
from modules import model_ae as model
import platform
import glob
import os
import healpy as hp
sys.path.append('database')
import di_marcs as di
from sklearn.neighbors import KernelDensity
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, NSIDE, n_stars):
        super(Dataset, self).__init__()
        
        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)
        self.n_stars = n_stars
        self.n_training = n_stars

        self.minT = 3200.0
        self.maxT = 5800.0
        
        self.T = [None] * self.n_stars

        np.random.seed(123)

        nspots = np.random.randint(low=1, high=11, size=self.n_stars)
        tstar = np.random.uniform(low=4000, high=5500, size=self.n_stars)
        smooth = np.random.uniform(low=0.1, high=0.1, size=self.n_stars)

        for i in tqdm(range(self.n_stars)):
            T = self.random_star(nspots[i], tstar[i], smooth[i])
            self.T[i] = torch.tensor(T.astype('float32'))
                                
    def random_star(self, nspots, tstar, smooth):
        T = tstar * np.ones(self.npix)

        for i in range(nspots):
            vec = np.random.randn(3)
            vec /= np.sum(vec**2)
            
            radius = np.random.triangular(left=0.1, mode=0.1, right=1.0)

            T_spot_min = self.minT
            T_spot_max = np.min([1.2  * tstar, self.maxT])
            T_spot = np.random.uniform(low=T_spot_min, high=T_spot_max)                        
        
            px = hp.query_disc(self.NSIDE, vec, radius, nest=False)
            
            T[px] = T_spot
            
        # The smoothing works only on RINGED
        T = hp.sphtfunc.smoothing(T, sigma=smooth, verbose=False)

        T = hp.pixelfunc.reorder(T, inp='RING', out='NESTED')

        return T
                        
    def __getitem__(self, index):

        T = (self.T[index][None, :] - 4700.0) / 500.0
                        
        return T

    def __len__(self):
        return self.n_training

class Autoencoder(object):
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
            files = glob.glob('weights_ae/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)

        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)

        hyperparameters = checkpoint['hyperparameters']
        
        self.model = model.Model(hyperparameters).to(self.device)

        NSIDE = hyperparameters['NSIDE']
        self.npix = hp.nside2npix(NSIDE)

        NSIDE_xyz = hyperparameters['NSIDE']
        self.npix_xyz = hp.nside2npix(NSIDE_xyz)

        xyz = np.zeros((self.npix_xyz, 3))
        for i in range(self.npix_xyz):
            xyz[i, :] = hp.pix2vec(NSIDE_xyz, i, nest=True)

        self.xyz = torch.tensor(xyz.astype('float32')).to(self.device)
                        
        self.model.load_state_dict(checkpoint['state_dict'])  

        self.dataset = Dataset(NSIDE=NSIDE, n_stars=self.batch_size)

        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test(self, saveplots=False):
        self.model.eval()
        loss_avg = 0
        t = self.loader
        correct = 0
        total = 0
        n = 1

        # Plot comparisons of original and reconstructed surfaces
        with torch.no_grad():
            for batch_idx, T in enumerate(t):
                                
                T = T.to(self.device)
                
                out, z = self.model(T)
   
                out = out.cpu().numpy() * 500.0 + 4700.0
                target = T[:, 0, :].cpu().numpy() * 500.0 + 4700.0
                z = z.cpu().numpy()
                                                               
                break

            T_single = T[0:1, :, :].expand(100, 1, self.npix)
            out_single, z_single = self.model(T_single)

        fig, ax = pl.subplots(nrows=4, ncols=8, figsize=(12,8))
        for i in range(8):
            pl.axes(ax[0, i])
            hp.orthview(target[i, :], nest=True, hold=True, title='Original', cmap=pl.cm.inferno)
            pl.axes(ax[1, i])
            hp.orthview(out[i, :], nest=True, hold=True, title='AE', cmap=pl.cm.inferno)
            pl.axes(ax[2, i])
            hp.orthview(target[i+5, :], nest=True, hold=True, title='Original', cmap=pl.cm.inferno)
            pl.axes(ax[3, i])
            hp.orthview(out[i+5, :], nest=True, hold=True, title='AE', cmap=pl.cm.inferno)

        if (saveplots):
            pl.savefig('figs/vae_reconstruction.png')

        
        # Check linear interpolation between two latent codes in the latent space
        z0 = z[0:4, :]
        z1 = z[1:5, :]
        fig, ax = pl.subplots(nrows=4, ncols=7, figsize=(15,8))
        out = [None] * 7
        with torch.no_grad():
            for t in range(7):
                z = z0 + (z1 - z0) * t / 6.0
                z = torch.tensor(z.astype('float32')).to(self.device)
                tmp = self.model.decode(z)
   
                out[t] = tmp.cpu().numpy() * 500.0 + 4700.0
                                            
        for i in range(7):
            title = ''
            if (i == 0):
                title = 'Initial'
            if (i == 6):
                title = 'Target'
            for j in range(4):
                pl.axes(ax[j, i])
                hp.orthview(out[i][j, :], nest=True, hold=True, title=title, cmap=pl.cm.inferno)
        
        if (saveplots):
            pl.savefig('figs/vae_latent_interpolation.png')

        # Plot surfaces obtained by sampling the latent space
        out_single = out_single.cpu().numpy() * 500.0 + 4700.0
        fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(10,8))
        for i in range(15):
            pl.axes(ax.flat[i])
            hp.orthview(out_single[i, :], nest=True, hold=True, title='', cmap=pl.cm.inferno)
        pl.axes(ax.flat[-1])
        hp.orthview(target[0, :], nest=True, hold=True, title='Original', cmap=pl.cm.inferno)
        
        if (saveplots):
            pl.savefig('figs/vae_latent_samples.png')        

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

        self.n_lambda = len(out_lambda)

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

    def approximation(self):
        self.model.eval()
        loss_avg = 0
        t = self.loader
        correct = 0
        total = 0
        n = 1

        regions = [[5985.1, 5989.0], [6000.1, 6005.0], [6022.0, 6026.0]]        
        self.star = di.DopplerImaging(2**4, regions=regions, root_models='database')

        # Plot comparisons of original and reconstructed surfaces
        with torch.no_grad():
            for batch_idx, T in enumerate(t):
                                
                T = T.to(self.device)
                
                out, z = self.model(T)
   
                out = out.cpu().numpy() * 500.0 + 4700.0
                target = T[:, 0, :].cpu().numpy() * 500.0 + 4700.0

                if (batch_idx == 5):
                    break

                print(np.percentile(out - target, [10, 50., 90.]), np.std(out - target))

        
        velocity = 30.0
        phases = np.arange(5) / 8.0 * 2.0 * np.pi

        n = 100
        t = tqdm(range(n))
        
        for i in t:
            wl, stokes = self.synthesize(target[i, :], 60.0, phases, velocity)
            wl, stokes2 = self.synthesize(out[i, :], 60.0, phases, velocity)

            if (i == 0):
                stokes_target = np.zeros((n, 5, self.n_lambda))
                stokes_out = np.zeros((n, 5, self.n_lambda))

            
            stokes_target[i, :, :] = np.concatenate(stokes, axis=-1)
            stokes_out[i, :, :] = np.concatenate(stokes2, axis=-1)

            t.set_postfix(std=np.std(stokes_target[0:i, :, :] - stokes_out[0:i, :, :]))

        fig, ax = pl.subplots()
        ax.hist((stokes_out-stokes_target).flatten(), bins='auto', density=True)
        ax.set_xlabel('$\Delta I$')
        ax.set_ylabel('Frequency')
        ax.set_xlim([-0.005, 0.005])

        breakpoint()

        # pl.hist((out-target).flatten(), bins=100, log=True)

if (__name__ == '__main__'):
    pl.close('all')    
    # network = Autoencoder(batch_size=256, gpu=1, checkpoint='weights_vae/2021-05-07-13:30')   # Latent 64
    network = Autoencoder(batch_size=256, gpu=3, checkpoint='weights_ae_nside16/2021-06-01-14:11')   # Latent 32
    # network = Autoencoder(batch_size=32, gpu=1, checkpoint=None)
    # network.test(saveplots=False)
    network.approximation()
