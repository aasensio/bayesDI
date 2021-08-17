from configobj import ConfigObj
import sys
import os
import numpy as np
from subprocess import call
import glob
import kurucz
import zarr
from tqdm import tqdm

def lower_to_sep(string, separator='='):
    line=string.partition(separator)
    string=str(line[0]).lower()+str(line[1])+str(line[2])
    return string


def run_model(conf, mus, model, output):

    ones = [1] * len(mus)

    # Transform all keys to lowercase to avoid problems with
    # upper/lower case
    f = open(conf,'r')
    input_lines = f.readlines()
    f.close()
    input_lower = ['']
    for l in input_lines:
        input_lower.append(lower_to_sep(l)) # Convert keys to lowercase

    config = ConfigObj(input_lower)

    file = open('kurucz_models/conf.input','w')

    # Write general information
    file.write("'"+config['general']['type of computation']+"'\n")
    file.write('{0}\n'.format(len(mus)))

    file.write('{0}\n'.format(" ".join([str(x) for x in mus])))
    file.write('{0}\n'.format(" ".join([str(x) for x in ones])))
    file.write("'"+model+"'\n")
    file.write("'"+config['general']['file with linelist']+"'\n") 
    file.write("'"+output+"'\n")

    file.write(config['wavelength region']['first wavelength']+'\n')
    file.write(config['wavelength region']['last wavelength']+"\n")
    file.write(config['wavelength region']['wavelength step']+"\n")
    file.write(config['wavelength region']['wavelength chunk size']+"\n")

    file.close()
    
#	Run the code
    call(['./lte.mint', 'conf.input'], cwd='kurucz_models/')
    

if (__name__ == '__main__'):
    
    n_mus = 15
    n_v = 160

    mus = np.linspace(0.02, 1.0, n_mus)
    T = 3000 + 100 * np.arange(11)
    T = np.append(T, 4000+250*(np.arange(8)+1) )
    n_T = len(T)
    velocities = np.linspace(-80.0, 80.0, n_v)
    
    for i in tqdm(range(n_T)):
        
        model = f'kurucz_models/ATMOS/MARCS/t{T[i]:d}g3.5z-0.25.model'
        tmp = model.split('/')[-1]
        tmp = tmp[:-5] + 'spec'
        outfile = 'RESULTS_MARCS/'+tmp
        
        run_model('kurucz_models/conf.ini', mus, '/'.join(model.split('/')[1:]), outfile)

        vel, wl, spec = kurucz.read_kurucz_spec(f'kurucz_models/{outfile}')

        if (i == 0):
            nlambda, nmus = spec.shape
            fout = zarr.open('marcs.zarr', 'w')
            T_ds = fout.create_dataset("T", shape=(n_T,), dtype=np.float32)
            v_ds = fout.create_dataset("v", shape=(n_v,), dtype=np.float32)
            mu_ds = fout.create_dataset("mu", shape=(n_mus,), dtype=np.float32)
            velaxis_ds = fout.create_dataset("velaxis", shape=(nlambda,), dtype=np.float32)
            wl_ds = fout.create_dataset("wavelength", shape=(nlambda,), dtype=np.float32)
            spec_ds = fout.create_dataset("spec", shape=(n_v, n_T, nmus, nlambda), dtype=np.float32)

            T_ds[:] = T
            v_ds[:] = velocities
            mu_ds[:] = mus
            velaxis_ds[:] = vel
            wl_ds[:] = wl

            velocity_per_pxl = vel[1] - vel[0]
            freq_grid = np.fft.fftfreq(nlambda)        
                
            kernel = np.exp(-2*1j*np.pi*freq_grid[None,:] * velocities[:,None] / velocity_per_pxl)

        for j in range(n_mus):
            fi = np.fft.fft(spec[:, j])
            tmp = fi[None, :] * kernel
            spec_ds[:, i, j, :] = np.fft.ifft(tmp).real