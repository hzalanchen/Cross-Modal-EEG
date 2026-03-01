import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .eeg_encoders import *


class EEG_Net(nn.Module):
    def __init__(self, opt):
        super(EEG_Net, self).__init__()
        self.eeg_Backbone = ATMS_Encoder()
        self.semantic_encoder = MLPEncoder(1024, opt.hidden_dim_S, opt.output_dim_S)
        self.domain_encoder = MLPEncoder(1024, opt.hidden_dim_D, opt.output_dim_D)

        self.eeg_Backbone.apply(weights_init_normal)
        self.semantic_encoder.apply(weights_init_normal)
        self.domain_encoder.apply(weights_init_normal)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if opt.geo_loss: 
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.manual_seed(42)

            eeg_prototypes = torch.randn((1654, 512))
            eeg_prototypes = eeg_prototypes / eeg_prototypes.norm(dim=-1, keepdim=True)
            self.register_buffer('eeg_prototypes', eeg_prototypes)

            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)

            self.mom_sched = ProtoMomentumScheduler(
                total_epochs = opt.train_epochs - getattr(opt, "geo_last_epochs", 25),
                start=opt.base_alpha
            )
            self.base_alpha = self.mom_sched(1)
    

    def set_mom_sched(self, epoch_one_based):
        if hasattr(self, "mom_sched"):
            self.base_alpha = self.mom_sched(epoch_one_based)

    def set_alpha_value(self, new_alpha_value):
        self.base_alpha = new_alpha_value

    
    @torch.no_grad()
    def update_memory(self, eeg_features, labels):
        unique_labels = labels.unique(sorted = True)
        mean_features = torch.zeros((len(unique_labels), eeg_features.size(1))).to(eeg_features.device)

        # update prototype
        for i, label in enumerate(unique_labels):
            indices = (labels == label)
            mean_features[i, :] = eeg_features[indices].mean(dim = 0)

        for idx, label in enumerate(unique_labels):
            self.eeg_prototypes[label] = self.base_alpha * self.eeg_prototypes[label] + (1 - self.base_alpha) * mean_features[idx]
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
        self.semantic_encoder = MLPEncoder(1024, 512, 512)

        self.eeg_Backbone.apply(weights_init_normal)
        self.semantic_encoder.apply(weights_init_normal)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if opt.geo_loss: 
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.manual_seed(42)

            eeg_prototypes = torch.randn((1654, 512))
            eeg_prototypes = eeg_prototypes / eeg_prototypes.norm(dim=-1, keepdim=True)
            self.register_buffer('eeg_prototypes', eeg_prototypes)

            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)

            self.mom_sched = ProtoMomentumScheduler(
                total_epochs = opt.train_epochs - getattr(opt, "geo_last_epochs", 25),
                start=opt.base_alpha
            )
            self.base_alpha = self.mom_sched(1)


    def set_mom_sched(self, epoch_one_based):
        if hasattr(self, "mom_sched"):
            self.base_alpha = self.mom_sched(epoch_one_based)


    def set_alpha_value(self, new_alpha_value):
        self.base_alpha = new_alpha_value


    @torch.no_grad()
    def update_memory(self, eeg_features, labels):
        device = torch.device('cuda') if eeg_features.is_cuda else torch.device('cpu')
        unique_labels = labels.unique(sorted = True)
        mean_features = torch.zeros((len(unique_labels), eeg_features.size(1))).to(device)

        # update prototype
        for i, label in enumerate(unique_labels):
            indices = (labels == label)
            mean_features[i, :] = eeg_features[indices].mean(dim = 0)

        for idx, label in enumerate(unique_labels):
            self.eeg_prototypes[label] = self.base_alpha * self.eeg_prototypes[label] + (1 - self.base_alpha) * mean_features[idx]
        self.eeg_prototypes[unique_labels] = self.eeg_prototypes[unique_labels] / self.eeg_prototypes[unique_labels].norm(dim = -1, keepdim= True)
        del mean_features
        return

    
    def forward(self, x):
        eeg_feature = self.eeg_Backbone(x)
        eeg_zs = self.semantic_encoder(eeg_feature)
        return eeg_zs




class EEG_Net_CLIP_Con(nn.Module):
    def __init__(self, opt):
        super(EEG_Net_CLIP_Con, self).__init__()
        
        self.eeg_Backbone = ATMS_Encoder()
        self.semantic_encoder = Proj_eeg(embedding_dim=1024, proj_dim=1024)

        self.eeg_Backbone.apply(weights_init_normal)
        self.semantic_encoder.apply(weights_init_normal)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        eeg_feature = self.eeg_Backbone(x)
        eeg_zs = self.semantic_encoder(eeg_feature)
        return eeg_zs