import shutil
import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
import h5py
from torch.autograd import Variable
import time
from tqdm import tqdm
import sys
from modules import model_ae as model
import platform
import argparse
import healpy as hp
import torch.distributions as td
from matplotlib.lines import Line2D

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(Dataset, self).__init__()
        
        print("Reading stars...")

        f = h5py.File(filename, 'r')
        
        self.n_training, _ = f['T'].shape        
        
        self.T = torch.tensor(f['T'][:].astype('float32'))
        self.T = (self.T - 4700.0) / 500.0
                                    
    def __getitem__(self, index):

        T = self.T[index:index+1, :]
                        
        return T

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Autoencoder(object):
    def __init__(self, batch_size=64, gpu=0, smooth=0.05, validation_split=0.2):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size        
                
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        
        self.dataset = Dataset(filename='/scratch1/aasensio/doppler_imaging/nside16/stars_T_spots.h5')

        hyperparameters = {
            'NSIDE': 16,
            'channels_enc': 32,
            'dim_latent_enc': 64,
            'n_steps_enc': 3,
            'dim_hidden_dec': 128,
            'dim_hidden_mapping': 128,
            'siren_num_layers': 3
        }
                
        self.model = model.Model(hyperparameters).to(self.device)
    
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

         # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
        
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)        

    def optimize(self, epochs, lr=3e-4):        

        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = f'weights_ae_nside16/{current_time}.pth'

        print(f'Model: {self.out_name}')
        print(" Number of params : ", sum(x.numel() for x in self.model.parameters()))

        # Copy model
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.5)

        self.beta = 0.0
        self.iteration = 0
        self.n_warmup_steps = 10000

        for epoch in range(1, epochs + 1):
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            if  (valid_loss < best_loss):
                best_loss = valid_loss

                hyperparameters = self.model.hyperparameters

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': hyperparameters,
                    'optimizer': self.optimizer.state_dict(),
                }
                
                print("Saving model...")
                torch.save(checkpoint, f'{self.out_name}')

            self.scheduler.step()

    def train(self, epoch):
        self.model.train()
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"Epoch {epoch}/{self.n_epochs}    - t={current_time}")
        t = tqdm(self.train_loader)
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, T in enumerate(t):
                                    
            T = T.to(self.device)
                        
            self.optimizer.zero_grad()
            
            out, z = self.model(T)
            
            loss = self.mse_loss(out.squeeze(), T.squeeze())
                                    
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()

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

            self.iteration += 1
            
        return loss_avg

    def validate(self):
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        correct = 0
        total = 0
        n = 1
        with torch.no_grad():
            for batch_idx, T in enumerate(t):
                                                
                T = T.to(self.device)
                
                out, z = self.model(T)
            
                loss = self.mse_loss(out.squeeze(), T.squeeze())

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
    parser.add_argument('--batch', '--batch', default=128, type=int,
                    metavar='BATCH', help='Batch size')
    parser.add_argument('--split', '--split', default=0.1, type=float,
                    metavar='SPLIT', help='Validation split')
    
    
    parsed = vars(parser.parse_args())

    network = Autoencoder(batch_size=parsed['batch'], gpu=parsed['gpu'], validation_split=parsed['split'], smooth=parsed['smooth'])
    network.optimize(parsed['epochs'], lr=parsed['lr'])