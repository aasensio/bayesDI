import math
import torch
from torch import nn
import torch.nn.functional as F

# from einops import rearrange, repeat

# https://github.com/lucidrains/pi-GAN-pytorch/blob/main/pi_gan_pytorch/pi_gan_pytorch.py

# helper

def exists(val):
    return val is not None

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 30., c = 6., is_first = False, use_bias = True, activation = 'sine'):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is 'sine' else Nones        

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)

        # FiLM modulation

        if exists(gamma):
            out = out * gamma[:, None, :]

        if exists(beta):
            out = out + beta[:, None, :]

        if (self.activation is not None):
            out = self.activation(out)
        return out

# mapping network
class MappingNetwork(nn.Module):
    def __init__(self, *, dim_input, dim_hidden, dim_out, depth_hidden = 3):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_input, dim_hidden))
        self.layers.append(nn.LeakyReLU(0.2))

        for i in range(depth_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nn.LeakyReLU(0.2))
        
        self.to_gamma = nn.Linear(dim_hidden, dim_out)
        self.to_beta = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.to_gamma(x), self.to_beta(x)

# siren network
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.last_layer = nn.Linear(dim_hidden, dim_out, bias = use_bias)

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)

        return self.last_layer(x)

# generator

# class SirenGenerator(nn.Module):
#     def __init__(self, dim_input, dim_hidden, siren_num_layers = 6):
#         super().__init__()
        
#         self.siren = SirenNet(
#             dim_in = dim_input,
#             dim_hidden = dim_hidden,
#             dim_out = dim_hidden,
#             num_layers = siren_num_layers
#         )

#         self.to_output = nn.Linear(dim_hidden, 1)

#     def forward(self, coors, gamma, beta):
#         x = self.siren(coors, gamma, beta)
#         output = self.to_output(x)

#         return output

# class Generator(nn.Module):
#     def __init__(
#         self,
#         *,
#         image_size,
#         dim,
#         dim_hidden,
#         siren_num_layers
#     ):
#         super().__init__()
#         self.image_size = image_size

#         coors = torch.stack(torch.meshgrid(
#             torch.arange(image_size),
#             torch.arange(image_size)
#         ))

#         coors = rearrange(coors, 'c h w -> (h w) c')
#         self.register_buffer('coors', coors)

#         self.G = SirenGenerator(
#             image_size = image_size,
#             dim = dim,
#             dim_hidden = dim_hidden,
#             siren_num_layers = siren_num_layers
#         )

#     def forward(self, x, ray_direction):
#         device, b = x.device, x.shape[0]
#         coors = repeat(self.coors, 'n c -> b n c', b = b).float()
#         ray_direction = repeat(ray_direction, 'b c -> b n c', n = coors.shape[1])
#         return self.G(x, ray_direction, coors)
