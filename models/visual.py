import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class Visual(nn.Module):
    def __init__(self, opt):
        super(Visual, self).__init__()
        self.visual_backbone = nn.Identity()
        self.semantic_encoder = MLPEncoder(1024, opt.hidden_dim_S, opt.output_dim_S)
        self.domain_encoder = MLPEncoder(1024, opt.hidden_dim_D, opt.output_dim_D)

        self.semantic_encoder.apply(weights_init_normal)
        self.domain_encoder.apply(weights_init_normal)

        if opt.geo_loss_visual: 
            visual_prototypes = torch.randn((1654, 512))
            visual_prototypes = visual_prototypes / visual_prototypes.norm(dim=-1, keepdim=True)
            self.register_buffer('visual_prototypes', visual_prototypes)

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
    def update_memory(self, visual_features, labels):
        unique_labels = labels.unique(sorted = True)
        mean_features = torch.zeros((len(unique_labels), visual_features.size(1))).to(visual_features.device)

        # update prototype
        for i, label in enumerate(unique_labels):
            indices = (labels == label)
            mean_features[i, :] = visual_features[indices].mean(dim = 0)

        for idx, label in enumerate(unique_labels):
            self.visual_prototypes[label] = self.base_alpha * self.visual_prototypes[label] + (1 - self.base_alpha) * mean_features[idx]
        self.visual_prototypes[unique_labels] = self.visual_prototypes[unique_labels] / self.visual_prototypes[unique_labels].norm(dim = -1, keepdim= True)
        del mean_features
        return

    def forward(self, image):
        image_feature = self.visual_backbone(image)
        image_s = self.semantic_encoder(image_feature)
        image_d = self.domain_encoder(image_feature)

        return image_feature, image_s, image_d


class Visual_Con(nn.Module):
    def __init__(self, opt):
        super(Visual_Con, self).__init__()
        self.visual_backbone = nn.Identity()
        self.semantic_encoder = MLPEncoder(1024, 512, 512)
        self.semantic_encoder.apply(weights_init_normal)

        if opt.geo_loss_visual: 
            visual_prototypes = torch.randn((1654, 512))
            visual_prototypes = visual_prototypes / visual_prototypes.norm(dim=-1, keepdim=True)
            self.register_buffer('visual_prototypes', visual_prototypes)

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
    def update_memory(self, visual_features, labels):
        unique_labels = labels.unique(sorted = True)
        mean_features = torch.zeros((len(unique_labels), visual_features.size(1))).to(visual_features.device)

        # update prototype
        for i, label in enumerate(unique_labels):
            indices = (labels == label)
            mean_features[i, :] = visual_features[indices].mean(dim = 0)

        for idx, label in enumerate(unique_labels):
            self.visual_prototypes[label] = self.base_alpha * self.visual_prototypes[label] + (1 - self.base_alpha) * mean_features[idx]
        self.visual_prototypes[unique_labels] = self.visual_prototypes[unique_labels] / self.visual_prototypes[unique_labels].norm(dim = -1, keepdim= True)
        del mean_features
        return


    def forward(self, image):
        image_feature = self.visual_backbone(image)
        image_s = self.semantic_encoder(image_feature)
        return image_s


class Visual_CLIP_Con(nn.Module):
    def __init__(self, opt):
        super(Visual_CLIP_Con, self).__init__()
        self.visual_backbone = nn.Identity()

    def forward(self, image):
        return image