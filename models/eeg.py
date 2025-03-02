import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import weights_init_normal
from .eeg_encoders import *
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class Semantic_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Semantic_Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activate(x)
        out = self.linear2(x)
        return out 


class Domain_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Domain_Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activate(x)
        out = self.linear2(x)
        return out 


class EEG_Decoder(nn.Module):
    def __init__(self, opt, output_dim):
        super(EEG_Decoder, self).__init__()
        self.opt = opt
        if self.opt.rec == 'cat':
            self.input_dim = self.opt.output_dim_S + self.opt.output_dim_D
        elif self.opt.rec == 'add':
            self.input_dim = opt.output_dim_S

        self.hidden_dim = opt.hidden_dim_dec
        self.output_dim = output_dim

        self.activate = nn.GELU()

        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activate,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activate,
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.decoder.apply(weights_init_normal)

    def forward(self, domain_h, semantic_h):
        if self.opt.rec == 'cat':
            input = torch.cat([domain_h, semantic_h], dim=1)
        elif self.opt.rec == 'add':
            input = domain_h + semantic_h
        out = self.decoder(input)
        return out 
    

class EEG_Net(nn.Module):
    def __init__(self, opt):
        super(EEG_Net, self).__init__()
        self.eeg_Backbone = ATMS_Encoder()
        self.semantic_encoder = Semantic_Encoder(1440, opt.hidden_dim_S, opt.output_dim_S)
        self.domain_encoder = Domain_Encoder(1440, opt.hidden_dim_D, opt.output_dim_D)

        self.eeg_Backbone.apply(weights_init_normal)
        self.semantic_encoder.apply(weights_init_normal)
        self.domain_encoder.apply(weights_init_normal)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if opt.geo_loss: 
            self.register_buffer('eeg_prototypes', torch.randn((1654, 512)))

    def update_memory(self, eeg_features, labels):
        device = torch.device('cuda') if eeg_features.is_cuda else torch.device('cpu')
        unique_labels = labels.unique(sorted = True)
        mean_features = torch.zeros((len(unique_labels), eeg_features.size(1))).to(device)

        # update prototype
        for i, label in enumerate(unique_labels):
            indices = (labels == label)
            mean_features[i, :] = eeg_features[indices].mean(dim = 0)

        for idx, label in enumerate(unique_labels):
            self.eeg_prototypes[label] = 0.5 * self.eeg_prototypes[label] + 0.5 * mean_features[idx]
        self.eeg_prototypes[unique_labels] = self.eeg_prototypes[unique_labels] / self.eeg_prototypes[unique_labels].norm(dim = -1, keepdim= True)
        del mean_features
        return
       
    
    def forward(self, x):
        eeg_feature = self.eeg_Backbone(x)
        eeg_zs = self.semantic_encoder(eeg_feature)
        eeg_zd = self.domain_encoder(eeg_feature)

        return eeg_feature, eeg_zs, eeg_zd


class EEG_Net_Con(nn.Module):
    def __init__(self, opt):
        super(EEG_Net_Con, self).__init__()
        
        self.eeg_Backbone = ATMS_Encoder()
        self.semantic_encoder = Semantic_Encoder(1440, 512, 512)

        self.eeg_Backbone.apply(weights_init_normal)
        self.semantic_encoder.apply(weights_init_normal)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if opt.geo_loss: 
            self.register_buffer('eeg_prototypes', torch.randn((1654, 512)))

    def update_memory(self, eeg_features, labels):
        device = torch.device('cuda') if eeg_features.is_cuda else torch.device('cpu')
        unique_labels = labels.unique(sorted = True)
        mean_features = torch.zeros((len(unique_labels), eeg_features.size(1))).to(device)

        # update prototype
        for i, label in enumerate(unique_labels):
            indices = (labels == label)
            mean_features[i, :] = eeg_features[indices].mean(dim = 0)

        for idx, label in enumerate(unique_labels):
            self.eeg_prototypes[label] = 0.5 * self.eeg_prototypes[label] + 0.5 * mean_features[idx]
        self.eeg_prototypes[unique_labels] = self.eeg_prototypes[unique_labels] / self.eeg_prototypes[unique_labels].norm(dim = -1, keepdim= True)
        del mean_features
        return
    
    def forward(self, x):
        eeg_feature = self.eeg_Backbone(x)
        eeg_zs = self.semantic_encoder(eeg_feature)
        return eeg_zs


# eegtest = torch.randn(2,1,17,33)
# eeg_Backbone = Enc_eeg()
# eeg_projector = Proj_eeg()
# print(eeg_projector(eeg_Backbone(eegtest)).shape)

