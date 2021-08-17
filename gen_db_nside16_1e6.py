import numpy as np
from mpi4py import MPI
from enum import IntEnum
import h5py
from tqdm import tqdm
import di_marcs as di
import healpy as hp
import pickle

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def compute(star, T, R, out_lambda):
    
    n_angles = np.random.randint(low=6, high=20, size=1)[0]
    
    rot_velocity = np.random.uniform(low=10, high=80)

    min_inclination = 10.0
    max_inclination = 85.0
    cosi = np.random.uniform(low=np.cos(max_inclination*np.pi/180.0), high=np.cos(min_inclination*np.pi/180.0))
    sini = np.sqrt(1.0 - cosi**2)
    phases = np.random.uniform(low=0, high=2.0*np.pi, size=n_angles)
    phases = np.sort(phases)

    los = np.zeros((n_angles,2))
    los[:, 0] = np.arcsin(sini)
    los[:, 1] = phases

    stokesi = star.compute_stellar_spectrum(T, los, omega=rot_velocity, resolution=R, reinterpolate_lambda=out_lambda)
                    
    return T, stokesi, phases, rot_velocity, sini

def master_work(filename, filename_surfaces, batchsize=100, NSIDE=16, write_frequency=1000):

    n_healpix = hp.nside2npix(NSIDE)

    # f = zarr.open(filename, 'w')

    f_surface = h5py.File(filename_surfaces, 'r')
    n_surfaces = f_surface['T'].shape[0]

    stokesi_list = [None] * n_surfaces
    T_list = [None] * n_surfaces
    angles_list = [None] * n_surfaces
    velocity_list = [None] * n_surfaces
    sini_list = [None] * n_surfaces
        
    task_index = 0
    num_workers = size - 1
    closed_workers = 0
    n_batches = n_surfaces // batchsize

    print("*** Master starting with {0} workers".format(num_workers))
    with tqdm(initial=task_index, total=n_batches, ncols=140) as pbar:
        while closed_workers < num_workers:
            dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:
                # Worker is ready, so send it a task
                if task_index < n_surfaces:
                    dataToSend = {'index': task_index, 'batchsize': batchsize, 'surface': f_surface['T'][task_index, :]}
                    comm.send(dataToSend, dest=source, tag=tags.START)
                    task_index += batchsize
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            elif tag == tags.DONE:
                index = dataReceived['index']
                T_list[index] = dataReceived['T']                
                stokesi_list[index] = dataReceived['stokesi']
                angles_list[index] = dataReceived['angles']
                velocity_list[index] = dataReceived['velocity']
                sini_list[index] = dataReceived['sini']
                            
                pbar.update(1)
                
                # print(" * MASTER : got block {0} from worker {1} - saved {2}".format(index, source, index), flush=True)
                    
            elif tag == tags.EXIT:
                print(" * MASTER : worker {0} exited.".format(source))
                closed_workers += 1

            if (pbar.n / write_frequency == pbar.n // write_frequency):
                with open(f'{filename}_stokes.pk', 'wb') as filehandle:
                    pickle.dump(stokesi_list[0:task_index], filehandle)

                with open(f'{filename}_T.pk', 'wb') as filehandle:
                    pickle.dump(T_list[0:task_index], filehandle)

                with open(f'{filename}_angles.pk', 'wb') as filehandle:
                    pickle.dump(angles_list[0:task_index], filehandle)

                with open(f'{filename}_velocity.pk', 'wb') as filehandle:
                    pickle.dump(velocity_list[0:task_index], filehandle)

                with open(f'{filename}_sini.pk', 'wb') as filehandle:
                    pickle.dump(sini_list[0:task_index], filehandle)

    print("Master finishing")

    with open(f'{filename}_stokes.pk', 'wb') as filehandle:
        pickle.dump(stokesi_list, filehandle)

    with open(f'{filename}_T.pk', 'wb') as filehandle:
        pickle.dump(T_list, filehandle)

    with open(f'{filename}_angles.pk', 'wb') as filehandle:
        pickle.dump(angles_list, filehandle)

    with open(f'{filename}_velocity.pk', 'wb') as filehandle:
        pickle.dump(velocity_list, filehandle)

    with open(f'{filename}_sini.pk', 'wb') as filehandle:
        pickle.dump(sini_list, filehandle)


def slave_work(NSIDE=16, R=None, out_lambda=None):

    star = di.DopplerImaging(NSIDE)

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
            
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
            batchsize = dataReceived['batchsize']
            surface = dataReceived['surface']

            T = [] 
            angles = []
            velocity = []
            sini = []
            stokesi = []
            
            for i in range(batchsize):
                T_, stokesi_, angles_, velocity_, sini_ = compute(star, surface, R=R, out_lambda=out_lambda)

                T.append(T_)
                stokesi.append(stokesi_)
                angles.append(angles_)
                velocity.append(velocity_)                
                sini.append(sini_)                
                
            dataToSend = {'index': task_index, 'T': T, 'stokesi': stokesi, 'angles': angles, 'velocity': velocity, 'sini': sini}

            comm.send(dataToSend, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


if (__name__ == '__main__'):

    NSIDE = 16 

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
    
    # star = synth.synth_spot(16, n_pixel_map=48, clv=True)
    # absorption, stokesi, phases, rot_velocity, depth, width, absorption_healpix, sini, spot_vec, spot_radius, spot_depth = compute(star, 150, 30)

    # breakpoint()
    
# Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    todo = 'training'

    # Validation
    if (todo == 'validation'):
        if rank == 0:
            filename = '/scratch/aasensio/doppler_imaging/nside16/validation'
            filename_surfaces = '/scratch/aasensio/doppler_imaging/nside16/stars_T_spots_validation.h5'
            master_work(filename, filename_surfaces, batchsize=1, NSIDE=NSIDE, write_frequency=1000)
        
        else:
            slave_work(NSIDE=NSIDE, R=R, out_lambda=out_lambda)

    comm.Barrier()
    comm.Barrier()

    # Training
    if (todo == 'training'): 
        if rank == 0:
            filename = '/scratch/aasensio/doppler_imaging/nside16_1e6/stars'
            filename_surfaces = '/scratch/aasensio/doppler_imaging/nside16_1e6/stars_T_spots.h5'
            master_work(filename, filename_surfaces, batchsize=1, NSIDE=NSIDE, write_frequency=50000)
        
        else:
            slave_work(NSIDE=NSIDE, R=R, out_lambda=out_lambda)
