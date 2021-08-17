import torch
import torch.nn as nn
import torch.nn.functional as F
import healpy as hp
import matplotlib.pyplot as pl
import numpy as np

class sphericalConv(nn.Module):
    def __init__(self, NSIDE, in_channels, out_channels, bias=True, nest=True):
        """
        Convolutional neural networks on the HEALPix sphere: a
        pixel-based algorithm and its application to CMB data analysis
        N. Krachmalnicoff1, 2? and M. Tomasi3, 4

        """
        super(sphericalConv, self).__init__()

        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)

        self.neighbours = torch.zeros(9 * self.npix, dtype=torch.long)
        for i in range(self.npix):
            # neighbours = [i]
            # neighbours.extend(hp.pixelfunc.get_all_neighbours(self.NSIDE, i, nest=nest))
            
            neighbours = hp.pixelfunc.get_all_neighbours(self.NSIDE, i, nest=nest)
            neighbours = np.insert(neighbours, 4, i)

            neighbours[neighbours == -1] = neighbours[4]

            self.neighbours[9*i:9*i+9] = torch.tensor(neighbours)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)
        
    def forward(self, x):
        
        vec = x[:, :, self.neighbours]        
        
        tmp = self.conv(vec)

        return tmp

class sphericalDown(nn.Module):
    def __init__(self, NSIDE):
        super(sphericalDown, self).__init__()
        
        self.pool = nn.AvgPool1d(4)
                
    def forward(self, x):
                
        return self.pool(x)

class sphericalUp(nn.Module):
    def __init__(self, NSIDE):
        super(sphericalUp, self).__init__()
                
    def forward(self, x):
        
        return torch.repeat_interleave(x, 4, dim=-1)

if (__name__ == '__main__'):
    
    NSIDE = 16
    pl.close('all')

    nspots = 5
    tstar = 5700.0
    smooth = 0.2
    
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

    conv = sphericalConv(NSIDE, 1, 1, bias=False, nest=True)    
    down = sphericalDown(NSIDE)
        
    with torch.no_grad():
        out = conv(T[None, None, :])
        out2 = down(out)
        

    hp.mollview(out[0, 0, :].numpy(), nest=True)
    hp.mollview(out2[0, 0, :].detach().numpy(), nest=True)
    
    # hp.mollview(im[0, 1, :].numpy())

    # hp.mollview(out[0, 0, :].detach().numpy(), nest=True)
    # hp.mollview(out[0, 1, :].detach().numpy())
    # pl.show()