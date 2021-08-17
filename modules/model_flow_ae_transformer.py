import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from tqdm import tqdm
from modules import flow
from modules import model_ae
from modules.Transformer import Encoder as TransformerEncoder
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
    def __init__(self, hyper_encoder, hyper_flow, old=False):
        """
        Define all layers
        """
        super(Model, self).__init__()

        self.hyper_encoder = hyper_encoder
        self.hyper_flow = hyper_flow        

        # Variational autoencoder for T
        self.checkpoint_ae = hyper_encoder['checkpoint_ae']        
        print("=> loading AE '{}'".format(self.checkpoint_ae))
        checkpoint = torch.load(self.checkpoint_ae, map_location=lambda storage, loc: storage)

        hyper_ae = checkpoint['hyperparameters']
        self.ae_latent_dim = hyper_ae['dim_latent_enc']
        
        self.ae = model_ae.Model(hyper_ae)
        self.ae.load_state_dict(checkpoint['state_dict'])  
        self.ae.eval()
        for param in self.ae.parameters():
            param.requires_grad = False
        print("=> done")
            
        # Feature extraction for the spectra                
        self.n_latent_enc = hyper_encoder['n_latent_enc']                
        self.dropout_att = hyper_encoder['dropout_attention']
        self.nheads_att = hyper_encoder['nheads_attention']
        self.nlayers_att = hyper_encoder['nlayers_attention']

        self.StokesEmbedding = nn.Linear(3, self.n_latent_enc)
        if (old):
            self.SpectralEncoder = TransformerEncoder(self.nlayers_att, self.nheads_att, self.n_latent_enc, 128, dropout=self.dropout_att, norm_in=True)
        else:
            self.SpectralEncoder = TransformerEncoder(self.nlayers_att, self.nheads_att, self.n_latent_enc, self.n_latent_enc*2, dropout=self.dropout_att, norm_in=True)

        # FiLM
        self.film_angles = Mapping(n_input=1, n_output=self.n_latent_enc)
        self.film_rotation = Mapping(n_input=2, n_output=self.n_latent_enc)
                
        # Attention
        
        # self.attention = nn.TransformerEncoderLayer(self.n_latent_enc, nhead=self.nheads_att, dim_feedforward=128, dropout=self.dropout_att, activation='relu')
        if (old):
            self.TimeEncoder = TransformerEncoder(self.nlayers_att, self.nheads_att, self.n_latent_enc, 128, dropout=self.dropout_att, norm_in=True)
        else:
            self.TimeEncoder = TransformerEncoder(self.nlayers_att, self.nheads_att, self.n_latent_enc, self.n_latent_enc*2, dropout=self.dropout_att, norm_in=True)

        # Flow
        self.n_latent_T = hyper_ae['dim_latent_enc']
        self.num_flow_steps = hyper_flow['n_flow_steps']
        self.base_transform_kwargs = hyper_flow['base_transform_kwargs']

        self.flow = flow.create_nsf_model(
            input_dim=hyper_ae['dim_latent_enc'], 
            context_dim=self.n_latent_enc,
            num_flow_steps=self.num_flow_steps, 
            base_transform_kwargs=self.base_transform_kwargs,
            learn_normal=False)

        # Show number of parameters of each network
        networks = [self.ae, self.StokesEmbedding, self.SpectralEncoder, self.TimeEncoder, self.flow, self.film_angles, self.film_rotation]
        labels = ['AE', 'Embedding', 'SpectralEncoder', 'TimeEncoder', 'Flow', 'FiLM angles', 'FiLM rotation']

        npar_total = 0
        npar_total_learn = 0
        for i, network in enumerate(networks):
            npar = sum(x.numel() for x in network.parameters())            
            npar_learn = sum(x.numel() for x in network.parameters() if x.requires_grad)
            npar_total += npar
            npar_total_learn += npar_learn
            print(f" Number of params {labels[i]:<16}             : {millify(npar, precision=3)} / {millify(npar_learn, precision=3)}")
        
        print(f" Number of total : {millify(npar_total, precision=3)} / {millify(npar_total_learn, precision=3)}")

    def context(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength):
        """
        Calculate the flow context vector from the observations
        """
        
        n_batch, n_steps, n_lambda = stokesi_residual.shape                

        # Clone the mean spectrum for all timesteps and flatten batch+time to apply the encoder for all timesteps and batch
        tmp_wavelength = wavelength[:, None, :].expand(-1, n_steps, n_lambda).reshape(-1, n_lambda).unsqueeze(-1)
        tmp_mn = stokesi_mn[:, None, :].expand(-1, n_steps, n_lambda).reshape(-1, n_lambda).unsqueeze(-1)
        tmp_residual = stokesi_residual.reshape(-1, n_lambda).unsqueeze(-1)
        
        # Serialize all Stokes parameters for all time steps and treat wavelengths as a sequence
        tmp_stokes = torch.cat([tmp_wavelength, tmp_mn, tmp_residual], dim=-1)

        # Compute masks, both spectral and in time
        mask_spectral = (tmp_mn[:, :, 0] == 0.0).unsqueeze(1).unsqueeze(2)
        mask_time = (angles != -999).unsqueeze(-1)

        # First embedding
        # [B*T, S, 3] -> [B*T, S, n_emb]
        out = self.StokesEmbedding(tmp_stokes)
        
        # First Transformer Encoder to encode spectral information
        # The mask needs to indicate with False those spectral points to attend to
        # [B*T, S, n_emb] -> [B*T, S, n_emb]
        out = self.SpectralEncoder(out, mask_spectral)
        
        # [B*T, S, n_emb] -> [B*T, n_emb] -> [B, T, n_emb]
        out = torch.mean(out, dim=1).reshape((n_batch, n_steps, -1))

        # Now we mask all unavailable times
        out = out.masked_fill(mask_time == 0, 0.0)
        
        # Add an embedding based on the phase angle using FiLM
        film_angles_gamma, film_angles_beta = self.film_angles(angles[:, :, None])
        film_angles_gamma = film_angles_gamma.masked_fill(mask_time == 0, 0.0)
        film_angles_beta = film_angles_beta.masked_fill(mask_time == 0, 0.0)

        out = out * film_angles_gamma + film_angles_beta
                        
        # Second Transformer Encoder to encode time information
        # It produce a unique latent vector by attending to all timesteps
        # We apply a mask to only attend to the time steps in each element of the batch
        # The mask needs to indicate with False those spectral points to attend to
        # [B, T, n_emb] -> [B, T, n_emb]        
        out = self.TimeEncoder(out, (~mask_time[:, :, 0]).unsqueeze(1).unsqueeze(2))

        # Mask all unavailable times
        out = out.masked_fill(mask_time == 0, 0.0)

        # Create a unique context vector by averaging all phases
        context = torch.sum(out, dim=1) / nangles[:, None]
        
        # Add a conditioning using FiLM for velocity and sini
        tmp = torch.cat([velocity[:, None], sini[:, None]], dim=-1)
        film_rotation_gamma, film_rotation_beta = self.film_rotation(tmp)
                
        context = context * film_rotation_gamma + film_rotation_beta

        return context
               
    def loss(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, T, wavelength):
        
        # Get context from observations
        context = self.context(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength)
    
        # Apply the flow to get the logprob
        latent_T = self.ae.encode(T[:, None, :])
        
        loss = torch.mean(-self.flow.log_prob(latent_T.squeeze(), context=context))

        return latent_T, loss

    def sample(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength, nsamples, xyz, batch_size=100):

        n_batch, _, _ = stokesi_residual.shape
        
        # Get context from observations
        context = self.context(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength)
    
        latent_T_samples, logprob = self.flow.sample_and_log_prob(nsamples, context=context)        

        num_batches = nsamples // batch_size
        num_leftover = nsamples % batch_size

        # Ugly trick to check what happens when the dispersion is smaller
        # mn = torch.mean(latent_T_samples, dim=1, keepdim=True)
        # latent_T_samples = (latent_T_samples - mn) * 0.3 + mn

        out_t = []
        left = 0
        for i in range(num_batches):
            tmp = self.ae.decoder(xyz, latent_T_samples[:, left:left+batch_size, :].reshape(-1, self.ae_latent_dim))
            out_t.append(tmp.reshape((n_batch, batch_size, -1)))
            left += batch_size
        
        if num_leftover > 0:
            tmp = self.ae.decoder(xyz, latent_T_samples[:, left:left+num_leftover, :].reshape(-1, self.ae_latent_dim))
            out_t.append(tmp.reshape((n_batch, batch_size, -1)))


        # tmp = self.ae.decoder(xyz, latent_T_samples.view(-1, self.ae_latent_dim))
        
        out = torch.cat(out_t, dim=1)
        
        return latent_T_samples, out, logprob

    def find_MAP(self, velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength, device, xyz):
        
        print("Computing MAP...")
        # Get context from observations
        context = self.context(velocity, sini, nangles, angles, stokesi_mn, stokesi_residual, T_max, wavelength)
    
        # Apply the flow to get the logprob
        latent_T = torch.zeros((1, self.n_latent_T), requires_grad=True, device=device)
        
        self.optimizer = torch.optim.Adam([latent_T], lr=0.1)

        losses = []

        t = tqdm(range(100))

        for loop in t:

            def closure():
                self.optimizer.zero_grad()
                loss = torch.mean(-self.flow.log_prob(latent_T, context=context))
                losses.append(loss.item())
                loss.backward()

                return loss

            self.optimizer.step(closure)

            t.set_postfix(iter=loop, loss=losses[-1])
        
        out = self.ae.decoder(xyz, latent_T)

        return latent_T, out, losses[-1]