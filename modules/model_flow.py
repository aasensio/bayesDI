import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from modules import flow
from modules import model_vae
from modules.Transformer import Encoder
from modules.millify import millify

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class SpectrumEncoder(nn.Module):
    def __init__(self, n_input=100, n_output=128, n_steps=4, channels=8, activation='relu'):
        """
        Define all layers
        """
        super(SpectrumEncoder, self).__init__()
                        
        if (activation == 'relu'):
            self.activation = nn.ReLU()
        if (activation == 'elu'):
            self.activation = nn.ELU()

        self.layers = nn.ModuleList([])

        self.layers.append(nn.Conv1d(2, channels, kernel_size=15, padding=7))
        self.layers.append(self.activation)
        self.layers.append(nn.Conv1d(channels, 4*channels, kernel_size=9, padding=4))
        self.layers.append(self.activation)

        for i in range(n_steps):
            self.layers.append(nn.Conv1d(4*channels, 4*channels, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.BatchNorm1d(4*channels))
            self.layers.append(self.activation)            
        
        self.n_enc = n_input // 2**n_steps * 4*channels
        self.final = nn.Linear(self.n_enc, n_output)        
                
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)            
        
        return self.final(x.view(-1, self.n_enc))

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class Mapping(nn.Module):
    def __init__(self, n_input=128, n_output=128):
        """
        Define all layers
        """
        super(Mapping, self).__init__()
                                
        self.to_gamma = nn.Linear(n_input, n_output)
        self.to_beta = nn.Linear(n_input, n_output)
                
    def forward(self, x):

        return self.to_gamma(x), self.to_beta(x)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class Model(nn.Module):
    def __init__(self, hyper_encoder, hyper_flow):
        """
        Define all layers
        """
        super(Model, self).__init__()

        self.hyper_encoder = hyper_encoder
        self.hyper_flow = hyper_flow        

        # Variational autoencoder for T
        self.checkpoint_vae = hyper_encoder['checkpoint_vae']        
        print("=> loading VAE '{}'".format(self.checkpoint_vae))
        checkpoint = torch.load(self.checkpoint_vae, map_location=lambda storage, loc: storage)

        hyper_vae = checkpoint['hyperparameters']
        self.vae_latent_dim = hyper_vae['dim_latent_enc']
        
        self.vae = model_vae.ModelSiren(hyper_vae)
        self.vae.load_state_dict(checkpoint['state_dict'])  
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("=> done")
            
        # Feature extraction for the spectra        
        self.n_input_enc = hyper_encoder['n_input_enc']
        self.n_latent_enc = hyper_encoder['n_latent_enc']
        self.n_steps_enc = hyper_encoder['n_steps_enc']
        self.channels_enc = hyper_encoder['channels_enc']
        self.activation_enc = hyper_encoder['activation_enc']
        self.encoder = SpectrumEncoder(n_input=self.n_input_enc, n_output=self.n_latent_enc, n_steps=self.n_steps_enc, channels=self.channels_enc, activation=self.activation_enc)
        self.encoder.weights_init()

        # FiLM
        self.film_angles = Mapping(n_input=1, n_output=self.n_latent_enc)
        self.film_rotation = Mapping(n_input=2, n_output=self.n_latent_enc)

        breakpoint()
                
        # Attention
        self.dropout_att = hyper_encoder['dropout_attention']
        self.nheads_att = hyper_encoder['nheads_attention']
        self.attention = nn.TransformerEncoderLayer(self.n_latent_enc, nhead=self.nheads_att, dim_feedforward=128, dropout=self.dropout_att, activation='relu')

        # Flow
        self.num_flow_steps = hyper_flow['n_flow_steps']
        self.base_transform_kwargs = hyper_flow['base_transform_kwargs']

        self.flow = flow.create_nsf_model(
            input_dim=hyper_vae['dim_latent_enc'], 
            context_dim=self.n_latent_enc,
            num_flow_steps=self.num_flow_steps, 
            base_transform_kwargs=self.base_transform_kwargs)

        # Show number of parameters of each network
        networks = [self.vae, self.encoder, self.flow, self.attention, self.film_angles, self.film_rotation]
        labels = ['VAE', 'SpectrumEncoder', 'Flow', 'Attention', 'FiLM angles', 'FiLM rotation']

        npar_total = 0
        npar_total_learn = 0
        for i, network in enumerate(networks):
            npar = sum(x.numel() for x in network.parameters())            
            npar_learn = sum(x.numel() for x in network.parameters() if x.requires_grad)
            npar_total += npar
            npar_total_learn += npar_learn
            print(f" Number of params {labels[i]:<16}             : {millify(npar, precision=3)} / {millify(npar_learn, precision=3)}")
        
        print(f" Number of total : {millify(npar_total, precision=3)} / {millify(npar_total_learn, precision=3)}")

    def context(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max):
        # Calculate the flow context vector from the observations

        n_batch, n_steps, n_lambda = stokesi_residual.shape        

        # Get mask for unobserved phases        
        mask = (angles != -999).unsqueeze(-1)

        # Clone the mean spectrum for all timesteps and flatten batch+time to apply the encoder for all timesteps and batch
        tmp_mn = stokesi_mn[:, None, :].expand(-1, n_steps, n_lambda).reshape(-1, n_lambda)
        tmp_residual = stokesi_residual.reshape(-1, n_lambda)
        
        tmp_stokes = torch.cat([tmp_mn[:, None, :], tmp_residual[:, None, :]], dim=1)

        # Encode all observations
        out = self.encoder(tmp_stokes).view(n_batch, n_steps, -1)

        # Mask all not observed values
        out = out.masked_fill(mask == 0, 0.0)

        # Now add a conditioning on the phase angle using FiLM
        film_angles_gamma, film_angles_beta = self.film_angles(angles[:, :, None])
        film_angles_gamma = film_angles_gamma.masked_fill(mask == 0, 0.0)
        film_angles_beta = film_angles_beta.masked_fill(mask == 0, 0.0)

        out = out * film_angles_gamma + film_angles_beta
                        
        # We now apply an attention mechanism based on a series of TransformerEncoder layers 
        # to produce a unique latent vector by attending to all timesteps
        # We apply a mask to only attend to the time steps in each element of the batch        
        
        # (Seq, Batch, Feature)
        attention = self.attention(out.permute(1,0,2), src_key_padding_mask=(angles == -999)).permute(1,0,2)

        attention = attention.masked_fill(mask == 0, 0.0)

        context = torch.mean(attention, dim=1)
        
        # Add a conditioning using FiLM for velocity, sini and T_max
        tmp = torch.cat([velocity[:, None], sini[:, None]], dim=-1) #, T_max[:, None]], dim=-1)
        film_rotation_gamma, film_rotation_beta = self.film_rotation(tmp)
                
        context = context * film_rotation_gamma + film_rotation_beta

        return context
               
    def loss(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, T):
        
        # Get context from observations
        context = self.context(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max)
    
        # Apply the flow to get the logprob
        latent_T, mu, logvar = self.vae.encode(T[:, None, :])

        loss = torch.mean(-self.flow.log_prob(latent_T.squeeze(), context=context))

        return latent_T, loss

    def sample(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, nsamples, xyz):

        n_batch, _, _ = stokesi_residual.shape
        
        # Get context from observations
        context = self.context(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max)
    
        latent_T_samples = self.flow.sample(nsamples, context=context)        

        tmp = self.vae.decode(latent_T_samples.view(-1, self.vae_latent_dim))

        out = tmp.view(n_batch, nsamples, -1)


        # out = []

        # for i in range(n_batch):
        #     tmp = self.vae.decode(latent_T_samples[i, :, :])
        #     out.append(tmp[None, :, :])

        # out = torch.cat(out, dim=0)
        
        return latent_T_samples, out
