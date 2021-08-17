import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from tqdm import tqdm
import zarr
import scipy.interpolate as interp
from convolres import convolres
import time

def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

class DopplerImaging(object):
    """
    """

    def __init__(self, NSIDE, regions=None, root_models=None):
        """
        This class does Doppler Imaging using several techniques

        Parameters
        ----------
        NSIDE : int
            number of sides in the Healpix pixellization.
        los : array [n_phases, 3]
            Angles defining the LOS for each phase
        omega : float
            Rotation velocity in km/s, by default 0.0
        vmax : float, optional
            Maximum velocity to be used. Only used if using Gaussians as [description], by default None
        clv : bool, optional
            [description], by default False
        device : str, optional
            [description], by default 'cpu'
        mode : str, optional
            [description], by default 'conv'
        synth : str, optional
            [description], by default 'kurucz'
        """
        
        self.NSIDE = int(NSIDE)
        self.hp_npix = hp.nside2npix(NSIDE)
       

        # Generate the indices of all healpix pixels
        self.indices = np.arange(hp.nside2npix(NSIDE), dtype="int")
        self.n_healpix_pxl = len(self.indices)

        self.polar_angle, self.azimuthal_angle = hp.pixelfunc.pix2ang(self.NSIDE, np.arange(self.n_healpix_pxl), nest=True)
        self.polar_angle = 2 * self.polar_angle / np.pi - 1.0
        
        self.pixel_vectors = np.array(hp.pixelfunc.pix2vec(self.NSIDE, self.indices, nest=True))

        # Compute LOS rotation velocity as v=w x r
        self.rotation_velocity = np.cross(np.array([0.0, 0.0, 1.0])[:, None], self.pixel_vectors, axisa=0, axisb=0, axisc=0)
            
        self.vec_boundaries = np.zeros((3, 4, self.n_healpix_pxl))
        for i in range(self.n_healpix_pxl):            
            self.vec_boundaries[:, :, i] = hp.boundaries(self.NSIDE, i, nest=True)
                        
        # Read all Kurucz models from the database. Hardwired temperature and mu angles
        print(" - Reading MARCS spectra...")
        if (root_models is None):
            f = zarr.open('marcs.zarr', 'r')
        else:
            f = zarr.open(f'{root_models}/marcs.zarr', 'r')
        self.T_kurucz = f['T'][:]
        self.mu_kurucz = f['mu'][:]
        self.v_kurucz = f['v'][:]
        self.kurucz_velocity = f['velaxis'][:]#[1300:2500]
        self.kurucz_wl = f['wavelength'][:] * 1e8 #[1300:2500] * 1e8
        self.kurucz_spectrum = f['spec'][:]

        ind = np.argsort(self.kurucz_wl)
        self.kurucz_wl = self.kurucz_wl[ind]
        self.kurucz_spectrum = self.kurucz_spectrum[:, :, :, ind]

        if (regions is not None):            
            n_regions = len(regions)            
            for i in range(n_regions):
                print(f'Extracting region {regions[i]}')
                region = regions[i]
                left = np.argmin(np.abs(self.kurucz_wl - region[0]))
                right = np.argmin(np.abs(self.kurucz_wl - region[1]))                
                if (i == 0):
                    wl = self.kurucz_wl[left:right]
                    spectrum = self.kurucz_spectrum[:, :, :, left:right]
                else:
                    wl = np.append(wl, self.kurucz_wl[left:right])
                    spectrum = np.append(spectrum, self.kurucz_spectrum[:, :, :, left:right], axis=-1)
        
            self.kurucz_wl = wl
            self.kurucz_spectrum = spectrum

        self.n_vel, self.n_T, self.nmus, self.nlambda = self.kurucz_spectrum.shape       
                                
        self.T = np.zeros(self.n_healpix_pxl)
        
    def trilinear_interpolate(self, v, T, mu):        
        ind_v0 = np.searchsorted(self.v_kurucz, v) - 1
        ind_T0 = np.searchsorted(self.T_kurucz, T) - 1
        ind_m0 = np.searchsorted(self.mu_kurucz, mu) - 1
        
        vd = (v - self.v_kurucz[ind_v0]) / (self.v_kurucz[ind_v0+1] - self.v_kurucz[ind_v0])
        Td = (T - self.T_kurucz[ind_T0]) / (self.T_kurucz[ind_T0+1] - self.T_kurucz[ind_T0])
        md = (mu - self.mu_kurucz[ind_m0]) / (self.mu_kurucz[ind_m0+1] - self.mu_kurucz[ind_m0])

        c000 = self.kurucz_spectrum[ind_v0, ind_T0, ind_m0, :]
        c001 = self.kurucz_spectrum[ind_v0, ind_T0, ind_m0 + 1, :]
        c010 = self.kurucz_spectrum[ind_v0, ind_T0 + 1, ind_m0, :]
        c011 = self.kurucz_spectrum[ind_v0, ind_T0 + 1, ind_m0 + 1, :]
        c100 = self.kurucz_spectrum[ind_v0 + 1, ind_T0, ind_m0, :]
        c101 = self.kurucz_spectrum[ind_v0 + 1, ind_T0, ind_m0 + 1, :]
        c110 = self.kurucz_spectrum[ind_v0 + 1, ind_T0 + 1, ind_m0, :]
        c111 = self.kurucz_spectrum[ind_v0 + 1, ind_T0 + 1, ind_m0 + 1, :]

        f1 = (1.0 - vd[:, None])
        f2 = vd[:, None]
        
        c00 = c000 * f1 + c100 * f2
        c01 = c001 * f1 + c101 * f2
        c10 = c010 * f1 + c110 * f2
        c11 = c011 * f1 + c111 * f2

        f1 = (1.0 - Td[:, None])
        f2 = Td[:, None]
        
        c0 = c00 * f1 + c10 * f2
        c1 = c01 * f1 + c11 * f2

        c = c0 * (1.0 - md[:, None]) + c1 * md[:, None]

        return c       
        
    def compute_stellar_spectrum(self, T, los, omega=0.0, vel_axis=None, clv=False, resolution=None, reinterpolate_lambda=None):
        """
        Compute the averaged spectrum on the star for a given temperature map and for a given rotation

        Parameters
        ----------
        obs : array of floats [n_phases, n_lambda]
            Observed Stokes profiles for computing the loss
        firstguess : bool, optional
            If it is a first guess or a full optimization, by default False

        Returns
        -------
        float
            Loss function
        array of floats [n_phases, n_lambda]
            Synthetic Stokes parameters
        """
        
        self.T = T
        self.clv = clv
        self.omega = omega    

        # Define the LOS
        self.los = np.zeros_like(los)
        self.los[:, 0] = np.rad2deg(los[:, 1])
        self.los[:, 1] = 90 - np.rad2deg(los[:, 0])

        self.los_vec = hp.ang2vec(los[:, 0], los[:, 1])

        self.n_phases, _ = los.shape
        
        self.visible_pixels = []
        self.area = []
        self.mu = []
        self.total_area = np.zeros(self.n_phases)
        self.vel = []
        self.kernel = []

        self.stokesi = np.zeros((self.n_phases, self.nlambda))

        self.epsilon = 0.01
        
        for loop in range(self.n_phases):

            # Query which pixels are visible
            visible_pixels = hp.query_disc(self.NSIDE, self.los_vec[loop, :], np.pi / 2.0 - self.epsilon, nest=True)

            # Compute area of each projected pixel
            # Use the fact that all Healpix pixels have the same area of the sphere. If the pixels are small enough, we can
            # just project the area to the plane of the sky computing the scalar product of the normal with the LOS
            area_projected = np.sum(self.pixel_vectors * self.los_vec[loop, :][:, None], axis=0)
            
            mu = area_projected[visible_pixels]
            T = self.T[visible_pixels]
                        
            self.total_area[loop] = np.sum(mu)

            vel = np.sum(self.rotation_velocity * self.los_vec[loop, :][:, None], axis=0)
            vel = self.omega * vel[visible_pixels]

            out = self.trilinear_interpolate(vel, T, mu) 
            
            # Sum over the visible surface
            self.stokesi[loop, :] = np.sum(out * mu[:, None], axis=0) / self.total_area[loop]
        
            if (resolution is not None):                
                self.stokesi[loop, :] = convolres(self.kurucz_wl, self.stokesi[loop, :], np.mean(self.kurucz_wl) / resolution)
                

        if (reinterpolate_lambda is not None):
            for loop in range(self.n_phases):
                f = interp.interp1d(self.kurucz_wl, self.stokesi[loop, :])
                tmp = f(reinterpolate_lambda)
                if (loop == 0):
                    out = np.zeros((self.n_phases, tmp.shape[0]))
                out[loop, :] = tmp
            return out
        else:        
            return self.stokesi


if __name__ == "__main__":
    NSIDE = 2**3

    n_phases = 5
    inclination = 60.0
    los = np.zeros((n_phases, 2))
    for i in range(n_phases):
        los[i, :] = np.array([inclination * np.pi / 180.0, 2.0 * np.pi / n_phases * i])

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

    regions = [[5985.1, 5989.0], [6000.1, 6005.0], [6022.0, 6026.0]]
    star = DopplerImaging(NSIDE, regions=regions)
            
    T = 5990.0 * np.ones(hp.nside2npix(NSIDE))
    # vec = np.array([0.9, -0.3, 0.2])
    # vec /= np.sum(vec**2)
    # radius = 0.3
    # px = hp.query_disc(NSIDE, vec, radius, nest=False)
    # T[px] = 3300.0            
    # T = hp.sphtfunc.smoothing(T, sigma=0.2, verbose=False)
    # T = hp.pixelfunc.reorder(T, inp='RING', out='NESTED')

    R_spectro = 65000.0
    v_spectro = 3e5 / R_spectro
    v_macro = 4.0
    v_total = np.sqrt(v_spectro**2 + v_macro**2)
    R = 3e5 / v_total
    
    start = time.time()
    stokesi = star.compute_stellar_spectrum(T, los, omega=80.0, resolution=R, reinterpolate_lambda=out_lambda)
    print(time.time() - start)

    # hp.mollview(star.T, nest=True)

    f, ax = pl.subplots()
    for i in range(n_phases):
        # ax.plot(star.kurucz_wl, stokesi[i, :] / np.max(stokesi[i, :]), color='C0', alpha=0.2)
        ax.plot(out_lambda, stokesi[i, :] / np.max(stokesi[i, :]), color='C0', alpha=0.2)

    pl.show()