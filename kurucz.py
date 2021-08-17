import numpy as np
import scipy.interpolate
import struct
import astropy.constants as const

def read_kurucz_spec(f):
    """
    Read Kurucz spectra that have been precomputed

    Args:
        f (string) : path to the file to be read
        
    Returns:
        new_vel (real array) : velocity axis in km/s
        spectrum (real array) : spectrum for each velocity bin
    """
    f = open(f, "rb")
    res = f.read()
    
    n_chunk = struct.unpack('i',res[0:4])
    
    freq = []
    stokes = []
    
    left = 4
    
    for i in range(n_chunk[0]):
        
        right = left + 4
        n = struct.unpack('i',res[left:right])

        left = right
        right = left + 4
        nmus = struct.unpack('i',res[left:right])


        left = right
        right = left + 8*n[0]
        t1 = np.asarray(struct.unpack('d'*n[0],res[left:right]))
        freq.append(t1)
                
        left = right
        right = left + 8*n[0]*nmus[0]

        t2 = np.asarray(struct.unpack('d'*n[0]*nmus[0],res[left:right])).reshape((n[0],nmus[0]))
        stokes.append(t2)
        
        left = right
        
    freq = np.concatenate(freq)
    stokes = np.concatenate(stokes)

    ind = np.argsort(freq)
    freq = freq[ind]
    stokes = stokes[ind]
    wavelength = const.c.to('cm/s').value / freq
    mean_wavelength = np.mean(wavelength)

    vel = (wavelength - mean_wavelength) / mean_wavelength * const.c.to('km/s').value

    nl, nmus = stokes.shape

# Reinterpolate in a equidistant velocity axis
    new_vel = np.linspace(np.min(vel), np.max(vel), nl)    
    wavelength = mean_wavelength + mean_wavelength * new_vel / const.c.to('km/s').value    

    for i in range(nmus):
        interpolator = scipy.interpolate.interp1d(vel, stokes[:,i], kind='linear')
        stokes[:,i] = interpolator(new_vel)

    return new_vel, wavelength, stokes