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
from modules import model_flow_ae_transformer as model
import platform
import argparse
from matplotlib.lines import Line2D

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
        
        print(" Reading all data...")

        filename = '/scratch1/aasensio/doppler_imaging/nside16/stars'
        
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
        
        print(" Normalizing data...")

        for i in tqdm(range(self.n_training)):
            ind = indices[i]
            self.T[i] = torch.tensor(T[ind][0].astype('float32'))
            self.T[i] = (self.T[i] - 4700.0) / 500.0
            self.velocity[i] = torch.tensor(velocity[ind][0] / 80.0)
            self.sini[i] = torch.tensor(sini[ind][0].astype('float32'))
            self.T_max[i] = torch.max(self.T[i])
            
            tmp = stokesi[ind][0]
            normalization = np.max(tmp, axis=-1)
            
            tmp = tmp / normalization[:, None]
            
            # Add noise            
            tmp += np.random.normal(loc=0, scale=1e-3, size=tmp.shape)

            mn = np.mean(tmp, axis=0)
            tmp = tmp - mn[None, :]

            self.stokesi_mn[i] = torch.tensor(((mn - 0.75) / 0.25).astype('float32'))  # The normalization extends the range to a larger range in the [0,1] interval
            self.stokesi_residual[i] = torch.tensor((tmp / 0.02).astype('float32'))   # Extracted from a quick analysis of the amplitudes
            self.wavelength[i] = torch.tensor(np.linspace(0.0, 1.0, len(mn)).astype('float32'))
                        
            self.angles[i] = torch.tensor(((angles[ind][0] - np.pi) / np.pi).astype('float32'))
            self.nangles[i] = torch.tensor(len(self.angles[i]))
        
        self.nangles_max = max(self.nangles)
        
    def __getitem__(self, index):

        # Fixed size
        velocity = self.velocity[index]
        sini = self.sini[index]
        T = self.T[index]
        T_max = self.T_max[index]
        stokesi_mn = self.stokesi_mn[index]
        nangles = self.nangles[index]
        wavelength = self.wavelength[index]

        # Variable size
        angles = self.angles[index]
        stokesi_residual = self.stokesi_residual[index]
                        
        return velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Doppler(object):
    def __init__(self, batch_size=64, gpu=0, smooth=0.05, validation_split=0.2, checkpoint_ae=None, warmup=False):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        self.warmup = warmup

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size                

        # Neural network                
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}        

        hyperparameters_encoder = {            
            'n_latent_enc': 256,
            'dropout_attention': 0.0,
            'nheads_attention': 2,
            'nlayers_attention': 5,
            'checkpoint_ae': checkpoint_ae,
        }

        base_transform_kwargs = {
            'hidden_dim': 256,
            'num_transform_blocks': 5,
            'activation': 'relu',
            'dropout_probability': 0.4,
            'batch_norm': True,
            'num_bins': 10,
            'tail_bound': 3.0,
            'apply_unconditional_transform': False,
            'transform_net': 'fc',
        }

        hyperparameters_flow = {
            'n_flow_steps': 5,
            'base_transform_kwargs': base_transform_kwargs,
        }

        self.latent = hyperparameters_encoder['n_latent_enc']
        if (self.warmup):
            self.n_warmup_steps = 10000
        
        self.model = model.Model(hyperparameters_encoder, hyperparameters_flow).to(self.device)
    
        self.dataset = Dataset(n_training=None)

        # Training set
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

         # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
        
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate, **kwargs)        

    def _get_lr_scale(self, epoch):
        d_model = self.latent
        n_warmup_steps = self.n_warmup_steps
        n_steps = epoch + 1
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def optimize(self, epochs, lr=3e-4):        

        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = f'weights_flow_ae_nside16/{current_time}.pth'

        print(' Model: {0}'.format(self.out_name))
        
        # Copy model
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        if (self.warmup):            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._get_lr_scale, last_epoch=-1)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.5)

        for epoch in range(1, epochs + 1):
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            # if (epoch == 1):
            #     dif = 0.0
            # else:
            #     dif = np.abs(valid_loss - best_loss)

            # # Save all models that produce a validation better than 0.5
            # if (dif < 0.5):

            if  (valid_loss < best_loss):
                best_loss = valid_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters_encoder': self.model.hyper_encoder,
                    'hyperparameters_flow': self.model.hyper_flow,
                    'optimizer': self.optimizer.state_dict(),
                }
                
                print("Saving model...")
                torch.save(checkpoint, f'{self.out_name}')

            # If we are not doing warmup, apply scheduler
            if (not self.warmup):
                self.scheduler.step()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0

        for batch_idx, (velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength) in enumerate(t):
                                    
            velocity, sini, wavelength = velocity.to(self.device), sini.to(self.device), wavelength.to(self.device)
            T, stokesi_mn, nangles = T.to(self.device), stokesi_mn.to(self.device), nangles.to(self.device)
            angles, stokesi_residual, T_max = angles.to(self.device), stokesi_residual.to(self.device), T_max.to(self.device)
                        
            self.optimizer.zero_grad()

            _, loss = self.model.loss(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, T, wavelength)
                                    
            loss.backward()

            # plot_grad_flow(self.model.named_parameters())
            # breakpoint()
            
            self.optimizer.step()

            # If we are doing warmup, apply scheduler
            if (self.warmup):
                self.scheduler.step()

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        return loss_avg

    def validate(self):
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        correct = 0
        total = 0
        n = 1
        with torch.no_grad():
            for batch_idx, (velocity, sini, T, stokesi_mn, nangles, angles, stokesi_residual, T_max, wavelength) in enumerate(t):
                                
                velocity, sini, wavelength = velocity.to(self.device), sini.to(self.device), wavelength.to(self.device)
                T, stokesi_mn, nangles = T.to(self.device), stokesi_mn.to(self.device), nangles.to(self.device)
                angles, stokesi_residual, T_max = angles.to(self.device), stokesi_residual.to(self.device), T_max.to(self.device)
                
                _, loss = self.model.loss(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, T, wavelength)
                
                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                                
                t.set_postfix(loss=loss_avg)            
   
        return loss_avg

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=64, type=int,
                    metavar='BATCH', help='Batch size')
    parser.add_argument('--split', '--split', default=0.08, type=float,
                    metavar='SPLIT', help='Validation split')
    parser.add_argument('--ae', '--ae', default='weights_ae_nside16/2021-06-01-14:11.pth', 
                    type=str, metavar='AE', help='AE weights')
    parser.add_argument('--warmup', dest='warmup', action='store_true')
    parser.add_argument('--no-warmup', dest='warmup', action='store_false')
    parser.set_defaults(warmup=False)
    
    parsed = vars(parser.parse_args())
    
    network = Doppler(
            batch_size=parsed['batch'], 
            gpu=parsed['gpu'], 
            validation_split=parsed['split'], 
            smooth=parsed['smooth'],
            checkpoint_ae=parsed['ae'],
            warmup=parsed['warmup'])

    network.optimize(parsed['epochs'], lr=parsed['lr'])