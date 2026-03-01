from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


def minmaxscaler(data):
    min_val = torch.min(data, dim=1, keepdim=True)[0]
    max_val = torch.max(data, dim=1, keepdim=True)[0]

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def z_norm(data):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)

    normalized_data = (data - mean) / std
    return normalized_data



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

        


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPEncoder, self).__init__()
        self.activate = nn.GELU()
        
        self.mlp_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activate,
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        out = self.mlp_encoder(x)
        return out 



class MLPDecoder(nn.Module):
    def __init__(self, opt, output_dim):
        super(MLPDecoder, self).__init__()
        self.opt = opt
        if self.opt.rec == 'cat':
            self.input_dim = self.opt.output_dim_S + self.opt.output_dim_D
        elif self.opt.rec == 'add':
            self.input_dim = opt.output_dim_S
        elif self.opt.rec == 'hadamard':
            self.input_dim = opt.output_dim_S 
            
        self.hidden_dim = opt.hidden_dim_dec
        self.output_dim = output_dim

        self.activate = nn.GELU()

        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activate,
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.mlp_decoder.apply(weights_init_normal)


    def forward(self, domain_h, semantic_h):
        if self.opt.rec == 'cat':
            input = torch.cat([domain_h, semantic_h], dim=1)
        elif self.opt.rec == 'add':
            input = domain_h + semantic_h
        elif self.opt.rec == 'hadamard':
            input = domain_h * semantic_h

        out = self.mlp_decoder(input)
        return out 


class ProtoMomentumScheduler:
    def __init__(self, total_epochs: int,
                 start: float = 0.50, 
                 end: float = 0.99,
                 mode: str = "cosine"):

        self.T = max(1, int(total_epochs))
        self.s = start
        self.e = end
        self.mode = mode

    def __call__(self, epoch_one_based: int) -> float:
        # epoch_one_based ∈ [1, T]
        if self.T <= 1:
            t = 1.0
        else:
            t = ( min( max(epoch_one_based, 1) , self.T) - 1 ) / (self.T - 1)

        if self.mode == "linear":
            return self.s + (self.e - self.s) * t
        elif self.mode == "cosine":
            import math
            return self.s + (self.e - self.s) * (1 - math.cos(math.pi * t)) * 0.5
        else:  # "exp"
            return float(self.s * ((self.e / self.s) ** t))
