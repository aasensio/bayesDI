import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from modules import siren
from modules import spherical
import healpy as hp
import torch.distributions as td

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self, NSIDE, channels_in=16, channels_out=16, dropout=0.0):
        """
        Define all layers of a residual conv block
        """
        super(ResidualBlock, self).__init__()
                        
        activation = nn.LeakyReLU(0.2)

        self.layers = nn.ModuleList([])

        # self.layers.append(nn.BatchNorm1d(channels_in))
        self.layers.append(activation)
        self.layers.append(spherical.sphericalConv(NSIDE, channels_in, channels_out, nest=True))

        # self.layers.append(nn.BatchNorm1d(channels_out))
        self.layers.append(activation)
        # self.layers.append(nn.Dropout(dropout))
        self.layers.append(spherical.sphericalConv(NSIDE, channels_out, channels_out, nest=True))

        self.residual = nn.Conv1d(channels_in, channels_out, kernel_size=1)
                
    def forward(self, x):

        tmp = x
        
        for layer in self.layers:
            x = layer(x)
        
        return x + self.residual(tmp)

    def weights_init(self):        
        for module in self.modules():
            kaiming_init(module)

class EncoderSpherical(nn.Module):
    def __init__(self, NSIDE, channels=16, dim_latent=128, n_steps=4):
        """
        Define all layers
        """
        super(EncoderSpherical, self).__init__()
                        
        activation = nn.LeakyReLU(0.2)

        self.layers = nn.ModuleList([])
        
        self.layers.append(spherical.sphericalConv(NSIDE, 1, channels, nest=True))
        self.layers.append(ResidualBlock(NSIDE, channels_in=channels, channels_out=channels, dropout=0.0))
        self.layers.append(spherical.sphericalDown(NSIDE))
        NSIDE = NSIDE // 2
        
        for i in range(n_steps):
            self.layers.append(ResidualBlock(NSIDE, channels_in=channels, channels_out=2*channels, dropout=0.0))
            self.layers.append(spherical.sphericalDown(NSIDE))
            channels *= 2
            NSIDE = NSIDE // 2

        self.mu = nn.Linear(channels * hp.nside2npix(NSIDE), dim_latent)
        self.logvar = nn.Linear(channels * hp.nside2npix(NSIDE), dim_latent)
                
    def forward(self, x):

        for layer in self.layers:            
            x = layer(x)
                    
        x = x.view(x.size(0), 1, -1)

        return self.mu(x), self.logvar(x)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class Encoder(nn.Module):
    def __init__(self, NSIDE, channels=16, dim_latent=128, n_steps=4):
        """
        Define all layers
        """
        super(Encoder, self).__init__()
                        
        self.activation = nn.LeakyReLU(0.2)

        self.layers = nn.ModuleList([])
        
        self.layers.append(spherical.sphericalConv(NSIDE, 1, channels, nest=True, bias=True))
        self.layers.append(self.activation)
        self.layers.append(spherical.sphericalConv(NSIDE, channels, channels, nest=True, bias=True))
        self.layers.append(self.activation)
        self.layers.append(spherical.sphericalDown(NSIDE))
        NSIDE = NSIDE // 2
        
        for i in range(n_steps):
            self.layers.append(spherical.sphericalConv(NSIDE, channels, channels, nest=True, bias=True))
            self.layers.append(self.activation)            
            self.layers.append(spherical.sphericalDown(NSIDE))
            channels *= 2
            NSIDE = NSIDE // 2

        self.mu = nn.Linear(channels * hp.nside2npix(NSIDE), dim_latent)
        self.logvar = nn.Linear(channels * hp.nside2npix(NSIDE), dim_latent)
                
    def forward(self, x):
            
        for layer in self.layers:            
            x = layer(x)
                            
        x = x.view(x.size(0), 1, -1)

        return self.mu(x), self.logvar(x)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class EncoderCNN(nn.Module):
    def __init__(self, channels=16, dim_latent=128, n_steps=4):
        """
        Define all layers
        """
        super(EncoderCNN, self).__init__()
                        
        self.activation = nn.LeakyReLU()

        self.layers = nn.ModuleList([])
        
        self.layers.append(nn.Conv1d(1, channels, kernel_size=9, padding=4, bias=True))
        self.layers.append(self.activation)
        self.layers.append(nn.Conv1d(channels, channels, kernel_size=9, padding=4, bias=True))
        self.layers.append(self.activation)
        self.layers.append(nn.MaxPool1d(2))
        
        for i in range(n_steps):
            self.layers.append(nn.Conv1d(channels, channels, kernel_size=9, padding=4, bias=True))
            self.layers.append(self.activation)            
            self.layers.append(nn.MaxPool1d(2))

        self.mu = nn.Linear(channels * 3072 // 2**(n_steps+1), dim_latent)
        self.logvar = nn.Linear(channels * 3072 // 2**(n_steps+1), dim_latent)
                
    def forward(self, x):        
        for layer in self.layers:            
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        
        return self.mu(x), self.logvar(x)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class EncoderMLP(nn.Module):
    def __init__(self, dim_input=155, dim_output=128, dim_hidden = 256, n_steps=4):
        """
        Define all layers
        """
        super(EncoderMLP, self).__init__()
                        
        self.activation = nn.LeakyReLU()

        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(dim_input, dim_hidden))        
        self.layers.append(self.activation)

        for i in range(n_steps):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))            
            self.layers.append(self.activation)
        
        self.mu = nn.Linear(dim_hidden, dim_output)
        self.logvar = nn.Linear(dim_hidden, dim_output)
                
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = x.squeeze()
        
        return self.mu(x), self.logvar(x)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class DecoderSiren(nn.Module):
    def __init__(self, dim_condition=155, dim_output=1, dim_hidden = 128, dim_hidden_mapping = 128, siren_num_layers=6):
        """
        Define all layers
        """
        super(DecoderSiren, self).__init__()

        self.mapping = siren.MappingNetwork(dim_input=dim_condition, dim_hidden=dim_hidden_mapping, dim_out = dim_hidden, depth_hidden = 2)

        self.siren = siren.SirenNet(dim_in = 3, dim_hidden = dim_hidden, dim_out = dim_hidden, num_layers = siren_num_layers)
        self.to_T = nn.Linear(dim_hidden, dim_output)
                        
    def forward(self, xyz, latent):
                
        gamma, beta = self.mapping(latent)
        
        out = self.siren(xyz, gamma, beta)
        out = self.to_T(out)

        return out.squeeze()

class DecoderMLP(nn.Module):
    def __init__(self, dim_condition=155, dim_output=1, dim_hidden = 128, num_layers=2):
        """
        Define all layers
        """
        super(DecoderMLP, self).__init__()

        self.activation = nn.LeakyReLU()

        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(dim_condition, dim_hidden))
        # self.layers.append(nn.BatchNorm1d(dim_hidden))
        self.layers.append(self.activation)

        for i in range(num_layers):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            # self.layers.append(nn.BatchNorm1d(dim_hidden))
            self.layers.append(self.activation)
                
        self.layers.append(nn.Linear(dim_hidden, dim_output))
                        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x

class ModelSiren(nn.Module):
    def __init__(self, hyperparameters):
        """
        Define all layers
        """
        super(ModelSiren, self).__init__()

        self.hyperparameters = hyperparameters

        # Hyperparameters
        self.NSIDE = hyperparameters['NSIDE']
        self.channels_enc = hyperparameters['channels_enc']
        self.dim_latent_enc = hyperparameters['dim_latent_enc']
        self.n_steps_enc = hyperparameters['n_steps_enc']
        self.dim_hidden_dec = hyperparameters['dim_hidden_dec']
        self.dim_hidden_mapping = hyperparameters['dim_hidden_mapping']
        self.siren_num_layers = hyperparameters['siren_num_layers']
        
        # Director cosines
        npix = hp.nside2npix(self.NSIDE)
        xyz = torch.zeros((npix, 3))
        for i in range(npix):
            xyz[i, :] = torch.tensor(hp.pix2vec(self.NSIDE, i, nest=True))

        self.register_buffer('xyz', xyz, persistent=False)
        
        # Encoder
        self.encoder = EncoderSpherical(self.NSIDE, dim_latent=self.dim_latent_enc, channels=self.channels_enc, n_steps=self.n_steps_enc)
        self.encoder.weights_init()

        # self.encoder = EncoderCNN(dim_latent=self.dim_latent_enc, channels=self.channels_enc, n_steps=self.n_steps_enc)
        # self.encoder.weights_init()

        # self.encoder = EncoderMLP(dim_input=3072, dim_output=self.dim_latent_enc, dim_hidden = 256, n_steps=2)

        # Decoder
        self.decoder = DecoderSiren(dim_condition=self.dim_latent_enc, dim_output=1, dim_hidden=self.dim_hidden_dec, dim_hidden_mapping=self.dim_hidden_mapping, siren_num_layers=self.siren_num_layers)
        # self.decoder = DecoderMLP(dim_condition=self.dim_latent_enc, dim_output=3072, dim_hidden=self.dim_hidden_dec, num_layers=2)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        
        # Get the normal distribution and sample from it using reparameterization
        std = torch.exp(0.5 * logvar)
        q_z = td.normal.Normal(mu, std)
        z = q_z.rsample()

        return z, mu, logvar

    def decode(self, z):
        return self.decoder(self.xyz, z)
        
    def forward(self, x):
        
        # Get mean and diagonal covariance matrix from the encoder
        mu, logvar = self.encoder(x)

        # Get normal distribution and sample from it using reparameterization
        std = torch.exp(0.5 * logvar)
        q_z = td.normal.Normal(mu, std)
        z = q_z.rsample()
        
        # Get output from decoder with the sampled latent
        out = self.decoder(self.xyz, z)
        # out = self.decoder(z)
        
        return out, z, q_z, mu, logvar

if (__name__ == '__main__'):
    hyperparameters = {
            'NSIDE': 16,
            'channels_enc': 16,
            'dim_latent_enc': 128,
            'n_steps_enc': 3,                        
            'dim_hidden_dec': 128,
            'dim_hidden_mapping': 128,
            'siren_num_layers': 2,
            'device': 'cpu'
        }
    tmp = ModelSiren(hyperparameters)
    
    nspots = 5
    tstar = 5700.0
    smooth = 0.2
    NSIDE = hyperparameters['NSIDE']

    T = tstar * np.ones(hp.nside2npix(NSIDE))

    for i in range(nspots):
        vec = np.random.randn(3)
        vec /= np.sum(vec**2)
        radius = np.random.uniform(low=0.1, high=0.5)
        T_spot = np.random.uniform(low=0.5, high=1.0) * tstar
        
        px = hp.query_disc(NSIDE, vec, radius, nest=False)
            
        T[px] = T_spot
            
    # The smoothing works only on RINGED
    T = hp.sphtfunc.smoothing(T, sigma=smooth, verbose=False)

    T = hp.pixelfunc.reorder(T, inp='RING', out='NESTED')
    T = torch.tensor(T.astype('float32'))
    T = (T - 4000.0) / (7500.0 - 4000.0)

    out, z, q_z, mu, logvar = tmp(T[None, None, :])