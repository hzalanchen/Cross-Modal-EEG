from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import *
from .modules import z_norm, xavier_weights_init, weights_init_normal


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


class Semantic_Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Semantic_Encoder, self).__init__()
        self.activate = nn.GELU()
        
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activate,
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        out = self.semantic_encoder(x)
        return out 


class Domain_Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Domain_Encoder, self).__init__()
        self.activate = nn.GELU()
        
        self.domain_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activate,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        out = self.domain_encoder(x)
        return out 
    

class Visual_Decoder(nn.Module):
    def __init__(self, opt, output_dim):
        super(Visual_Decoder, self).__init__()
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

        
class Visual(nn.Module):
    def __init__(self, opt):
        super(Visual, self).__init__()
        # self.visual_backbone = self.visual_backbone_net()
        self.visual_backbone = nn.Identity()
        self.semantic_encoder = Semantic_Encoder(1024, opt.hidden_dim_S, opt.output_dim_S)
        self.domain_encoder = Domain_Encoder(1024, opt.hidden_dim_D, opt.output_dim_D)

        self.semantic_encoder.apply(weights_init_normal)
        self.domain_encoder.apply(weights_init_normal)

    
    def visual_backbone_net(self, pretrained = True, load_path = '/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'):
        if pretrained:
            backbone_net = resnet50()
            backbone_net.load_state_dict(torch.load(load_path))
            backbone_net.fc = torch.nn.Identity()
        else:
            pass

        return backbone_net

    def forward(self, image):
        image_feature = self.visual_backbone(image)
        image_s = self.semantic_encoder(image_feature)
        image_d = self.domain_encoder(image_feature)

        return image_feature, image_s, image_d


class Visual_Con(nn.Module):
    def __init__(self, opt):
        super(Visual_Con, self).__init__()
        self.visual_backbone = nn.Identity()
        self.semantic_encoder = Semantic_Encoder(1024, 512, 512)
        self.semantic_encoder.apply(weights_init_normal)

    def forward(self, image):
        image_feature = self.visual_backbone(image)
        image_s = self.semantic_encoder(image_feature)
        return image_s

