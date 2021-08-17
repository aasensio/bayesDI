import numpy as np
import healpy as hp
import matplotlib.pyplot as pl
import h5py
from tqdm import tqdm

class Surface(object):
    def __init__(self, NSIDE, n_stars, filename=None):
        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)
        self.n_stars = n_stars

        self.minT = 3200.0
        self.maxT = 5800.0

        self.T = np.zeros((self.n_stars, self.npix))

        nspots = np.random.randint(low=1, high=11, size=self.n_stars)
        tstar = np.random.uniform(low=4000, high=5500, size=self.n_stars)
        smooth = np.random.uniform(low=0.1, high=0.1, size=self.n_stars)

        for i in tqdm(range(self.n_stars)):
            self.T[i] = self.random_star(nspots[i], tstar[i], smooth[i])

        if (filename is not None):
            f = h5py.File(filename, 'w')
            ds = f.create_dataset('T', (self.n_stars, self.npix))
            ds[:] = self.T
            f.close()

    def random_star(self, nspots, tstar, smooth):
        T = tstar * np.ones(self.npix)

        for i in range(nspots):
            vec = np.random.randn(3)
            vec /= np.sum(vec**2)
            # radius = np.random.uniform(low=0.1, high=1.0)
            radius = np.random.triangular(left=0.1, mode=0.1, right=1.0)

            T_spot_min = self.minT
            T_spot_max = np.min([1.2  * tstar, self.maxT])
            T_spot = np.random.uniform(low=T_spot_min, high=T_spot_max)            
            
            px = hp.query_disc(self.NSIDE, vec, radius, nest=False)
            
            T[px] = T_spot
            
        T = hp.sphtfunc.smoothing(T, sigma=smooth, verbose=False)

        T = hp.pixelfunc.reorder(T, inp='RING', out='NESTED')

        return T

if (__name__ == '__main__'):
#    tmp = Surface(NSIDE=16, n_stars=5000, filename='/net/diablos/scratch/aasensio/doppler_imaging/nside16/stars_T_spots_validation.h5')
#    tmp = Surface(NSIDE=16, n_stars=100000, filename='/net/diablos/scratch/aasensio/doppler_imaging/nside16/stars_T_spots.h5')
    tmp = Surface(NSIDE=16, n_stars=1000000, filename='/net/diablos/scratch/aasensio/doppler_imaging/nside16_1e6/stars_T_spots.h5')
    
    # pl.close('all')
    # tstar = 5000.0
    # NSIDE = 8
    # nspots = 3
    
    # T = tstar * np.ones(hp.nside2npix(NSIDE))
    # for i in range(nspots):
    #     vec = np.random.randn(3)
    #     vec /= np.sum(vec**2)
    
    #     radius = np.random.uniform(low=0.1, high=1.0)
    #     T_spot = np.random.uniform(low=0.5, high=1.2) * tstar

    #     print(radius, T_spot)
        
    #     if (T_spot <= 3000):
    #         T_spot = 3200.0
    #     if (T_spot >= 6000.0):
    #         T_spot = 5800.0

    
    #     px = hp.query_disc(NSIDE, vec, radius, nest=False)
    #     T[px] = T_spot

    # T = hp.sphtfunc.smoothing(T, sigma=0.1, verbose=False)
        

    # hp.orthview(T)
