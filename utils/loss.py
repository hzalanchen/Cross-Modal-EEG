from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import modality_gap_sample

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        https://arxiv.org/abs/2006.12013
        This class provides the CLUB estimation to I(X,Y) 
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    # x_dim 512 y_dim 512
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Tanh())


    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0) / 2.
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class CLUBSample_CO_Estimate(nn.Module):
    # x_dim 512 y_dim 512
    def __init__(self, x_dim, y_dim, hidden_size, lambda_dml=1.0):
        super(CLUBSample_CO_Estimate, self).__init__()

        self.p_mu_A = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim)
        )
        self.p_logvar_A = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh()
        )

        self.p_mu_B = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim)
        )
        self.p_logvar_B = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh()
        )

        self.active_head = 'A'           
        self.lambda_dml = float(lambda_dml)

        self.reset_parameters()

    def reset_parameters(self):
        def init_weights_A(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.p_mu_A.apply(init_weights_A)
        self.p_logvar_A.apply(init_weights_A)

        def init_weights_B(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) 

        self.p_mu_B.apply(init_weights_B)
        self.p_logvar_B.apply(init_weights_B)


    def get_mu_logvar_A(self, x_samples):
        mu_A = self.p_mu_A(x_samples)
        logvar_A = self.p_logvar_A(x_samples)
        return mu_A, logvar_A


    def get_mu_logvar_B(self, x_samples):
        mu_B = self.p_mu_B(x_samples)
        logvar_B = self.p_logvar_B(x_samples)
        return mu_B, logvar_B


    def get_mu_logvar(self, x_samples):
        if self.active_head == 'A':
            return self.get_mu_logvar_A(x_samples)
        else:
            return self.get_mu_logvar_B(x_samples)

    def use_head(self, head: str):
        self.active_head = 'A' if head == 'A' else 'B'


    @staticmethod
    def gaussian_kl(mu1, logvar1, mu2, logvar2):
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)

        kl = 0.5 * ( (logvar2 - logvar1) + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
        return kl.sum(dim=-1).mean()

    def dml_symmetric_kl(self, x_samples):
        mu_A, logvar_A = self.get_mu_logvar_A(x_samples)
        mu_B, logvar_B = self.get_mu_logvar_B(x_samples)
        return self.gaussian_kl(mu_A, logvar_A, mu_B, logvar_B) + \
               self.gaussian_kl(mu_B, logvar_B, mu_A, logvar_A)


    def loglikeli_A(self, x_samples, y_samples):
        mu_A, logvar_A = self.get_mu_logvar_A(x_samples)
        return (-(mu_A - y_samples)**2 / logvar_A.exp() - logvar_A).sum(dim=1).mean(dim=0) / 2.

    def loglikeli_B(self, x_samples, y_samples):
        mu_B, logvar_B = self.get_mu_logvar_B(x_samples)
        return (-(mu_B - y_samples)**2 / logvar_B.exp() - logvar_B).sum(dim=1).mean(dim=0) / 2.

    def forward_A(self, x_samples, y_samples):
        mu_A, logvar_A = self.get_mu_logvar_A(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = - (mu_A - y_samples)**2 / logvar_A.exp()
        negative = - (mu_A - y_samples[random_index])**2 / logvar_A.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def forward_B(self, x_samples, y_samples):
        mu_B, logvar_B = self.get_mu_logvar_B(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = - (mu_B - y_samples)**2 / logvar_B.exp()
        negative = - (mu_B - y_samples[random_index])**2 / logvar_B.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.


    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0) / 2.

    def forward(self, x_samples, y_samples):
        return 0.5 * (self.forward_A(x_samples, y_samples) +
                      self.forward_B(x_samples, y_samples))

    # def forward(self, x_samples, y_samples):
    #     mi_A = self.forward_A(x_samples, y_samples)
    #     mi_B = self.forward_B(x_samples, y_samples)
        
    #     # For training stability.
    #     eps = 1e-4
    #     mi_A = torch.clamp_min(mi_A, eps)
    #     mi_B = torch.clamp_min(mi_B, eps)

    #     return 0.5 * (mi_A + mi_B)

    def learning_loss(self, x_samples, y_samples):
        loss_A = - self.loglikeli_A(x_samples, y_samples)  # maximize -> add minus
        loss_B = - self.loglikeli_B(x_samples, y_samples)
        dml = self.dml_symmetric_kl(x_samples)             # minimize
        return loss_A + loss_B + self.lambda_dml * dml


class ConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, eeg_features, logit_scale):

        device = torch.device('cuda') if image_features.is_cuda else torch.device('cpu')
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size).to(device)
       
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_features, eeg_features.t())
        logits_per_eeg =  logits_per_image.t()

        loss_im = self.contrastive_loss(logits_per_image, labels)
        loss_eeg = self.contrastive_loss(logits_per_eeg, labels)
        
        return (loss_im + loss_eeg) / 2.

    def contrastive_loss(self, logits, labels):
        
        loss = F.cross_entropy(logits, labels)
        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, eeg_features, labels, logit_scale, mask=None):

        device = torch.device('cuda') if image_features.is_cuda else torch.device('cpu')
        batch_size = image_features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)

        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(image_features, eeg_features.t())
        logits_per_eeg =  logits_per_image.t()

        loss_im = self.contrastive_loss(logits_per_image, mask)
        loss_eeg = self.contrastive_loss(logits_per_eeg, mask)
        
        return (loss_im + loss_eeg) / 2.

    def contrastive_loss(self, logits, mask):

        mask_nosim = torch.ones_like(mask) - mask
        exp_logits = torch.exp(logits)
        probs = (exp_logits * mask) / exp_logits.sum(dim = 1, keepdim=True)
        
        # log value cannot be negatvie
        probs = probs + mask_nosim
        log_prob = torch.log(probs).sum(1) / mask.sum(1)

        return -log_prob.mean()


class SupConLoss_orginal(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_orginal, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class Geometry_Variance(nn.Module):
    def __init__(self):
        super(Geometry_Variance, self).__init__()
        self.temperature = 0.07

    def forward(self, image_features, labels, eeg_prototype, mask=None):

        device = torch.device('cuda') if image_features.is_cuda else torch.device('cpu')
        batch_size = image_features.shape[0]

        unique_label = labels.unique()
        unique_label = unique_label.contiguous().view(-1, 1)
        mask = torch.eq(unique_label, labels).float().to(device)

        unique_label = unique_label.squeeze(1)
        eeg_prototype_cls = eeg_prototype[unique_label.long()]

        # logit_scale = logit_scale.exp()
        # dis_matrix = logit_scale * torch.matmul(eeg_prototype_cls, image_features.t())
        dis_matrix = torch.cdist(eeg_prototype_cls, image_features, p=2)
        cls_dis = dis_matrix * mask

        cls_means = torch.sum(cls_dis, dim = 1) / torch.sum(mask, dim = 1)
        cls_means = cls_means.unsqueeze(-1).repeat(1, cls_dis.shape[-1])
        cls_dis = cls_dis + (1 - mask) * cls_means

        # MSE
        cls_variance = ((cls_dis - cls_means)**2).sum(dim = 1 ) / mask.sum(dim = 1) 
        non_zero = torch.count_nonzero(cls_variance) + 1e-4

        geo_loss = torch.sum(cls_variance) / non_zero
        return geo_loss


class Geometry_Gaps(nn.Module):
    def __init__(self):
        super(Geometry_Gaps, self).__init__()
        self.temperature = 0.07

    def forward(self, image_features, labels, eeg_prototype, mask=None):
        
        device = torch.device('cuda') if image_features.is_cuda else torch.device('cpu')
        batch_size = image_features.shape[0]

        unique_label = labels.unique()
        unique_label = unique_label.contiguous().view(-1, 1)
        mask = torch.eq(unique_label, labels).float().to(device)
        
        unique_label = unique_label.squeeze(1)
        eeg_prototype_cls = eeg_prototype[unique_label.long()]
        
        # logit_scale = logit_scale.exp()
        # dis_matrix = logit_scale * torch.matmul(eeg_prototype_cls, image_features.t())
        dis_matrix = torch.cdist(eeg_prototype_cls, image_features, p=2)
        cls_dis = dis_matrix * mask
        cls_gaps = torch.sum(cls_dis, dim = 1) / torch.sum(mask, dim = 1)
        geo_loss = cls_gaps.mean()

        return geo_loss


class Intra_Geometry_Variance(nn.Module):
    # Distance variance regularization.
    def __init__(self):
        super(Intra_Geometry_Variance, self).__init__()
        self.temperature = 0.07

    def forward(self, image_features, eeg_features, labels, mask=None):

        device = torch.device('cuda') if image_features.is_cuda else torch.device('cpu')
        batch_size = image_features.shape[0]
        unique_label = labels.unique()
        unique_label = unique_label.contiguous().view(-1, 1)
        mask = torch.eq(unique_label, labels).float().to(device)

        # cal batch eeg features centroid
        unique_label = unique_label.squeeze(1)
        bs_eeg_prototypes = torch.zeros((len(unique_label), eeg_features.size(1))).to(device)

        for i, label in enumerate(unique_label):
            indices = (labels == label)
            bs_eeg_prototypes[i, :] = eeg_features[indices].mean(dim = 0)

        dis_matrix = torch.cdist(bs_eeg_prototypes, image_features, p=2)
        cls_dis = dis_matrix * mask

        cls_means = torch.sum(cls_dis, dim = 1) / torch.sum(mask, dim = 1)
        cls_means = cls_means.unsqueeze(-1).repeat(1, cls_dis.shape[-1])
        cls_dis = cls_dis + (1 - mask) * cls_means
        # MSE
        cls_variance = ((cls_dis - cls_means)**2).sum(dim = 1 ) / mask.sum(dim = 1) 
        non_zero = torch.count_nonzero(cls_variance) + 1e-4

        geo_loss = torch.sum(cls_variance) / non_zero
        return geo_loss



# new
class Geometry_Variance_New(nn.Module):
    def __init__(self, distance_type: str = "euclidean", eps: float = 1e-6):
        super().__init__()
        self.distance_type = distance_type.lower()
        self.eps = eps

    def _compute_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: [U, D]   Prototype
        B: [B, D]   feature
        return: [U, B]  dist matrix
        """
        if self.distance_type == "cosine":
            sim = A @ B.t()                     
            dist = 1.0 - sim                    
        elif self.distance_type == "euclidean":
            dist = torch.cdist(A, B, p=2)       
        elif self.distance_type == "l1":
            dist = torch.cdist(A, B, p=1)             
        else:
            raise ValueError(f"Unsupported distance_type: {self.distance_type}")
        return dist

    def forward(self, image_features: torch.Tensor,
                      labels: torch.Tensor,
                      eeg_prototypes: torch.Tensor) -> torch.Tensor:

        device = image_features.device
        unique_label = labels.unique(sorted=True)
        Cb = eeg_prototypes[unique_label.long()]  # [U, D]

        mask = (labels.unsqueeze(0) == unique_label.unsqueeze(1)).float()

        cnt = mask.sum(dim=1)
        valid_cls_mask = (cnt >= 2)

        if not valid_cls_mask.any():
            return torch.tensor(0.0, device=device)
        
        # MSE
        dist_matrix = self._compute_distance(Cb, image_features)  # [U, B]
        dist_sum = (dist_matrix * mask).sum(dim=1)
        cls_means = dist_sum / (cnt + self.eps)
        
        target_radius = cls_means.detach()
        diff = dist_matrix - target_radius.unsqueeze(1)
        sq_diff = (diff ** 2) * mask
        
        cls_variances = sq_diff.sum(dim=1) / (cnt + self.eps)
        valid_variances = cls_variances[valid_cls_mask]

        return valid_variances.mean()


class Geometry_Gaps_Consistency(nn.Module):
    def __init__(self, distance_type: str = "cosine", eps: float = 1e-6):
        super().__init__()
        self.distance_type = distance_type.lower()
        self.eps = eps

    def _compute_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: [U, D]   Prototype
        B: [B, D]   feature
        return: [U, B]  dist matrix
        """
        if self.distance_type == "cosine":
            sim = A @ B.t()                     
            dist = 1.0 - sim                    
        elif self.distance_type == "euclidean":
            dist = torch.cdist(A, B, p=2)       
        elif self.distance_type == "l1":
            dist = torch.cdist(A, B, p=1)             
        else:
            raise ValueError(f"Unsupported distance_type: {self.distance_type}")
        return dist

    def forward(self, semantic_features: torch.Tensor,
                      labels: torch.Tensor,
                      semantic_prototypes: torch.Tensor) -> torch.Tensor:

        device = semantic_features.device
        unique_label = labels.unique(sorted=True)
        Cb = semantic_prototypes[unique_label.long()] 

        mask = (labels.unsqueeze(0) == unique_label.unsqueeze(1)).float()
        
        dist_matrix = self._compute_distance(Cb, semantic_features)  # [U, B]
        
        sum_dist_per_class = (dist_matrix * mask).sum(dim=1)
        
        counts = mask.sum(dim=1)
        mean_dist_per_class = sum_dist_per_class / counts
        loss_gaps = mean_dist_per_class.mean()

        return loss_gaps


class Geometry_Std_Consistency(nn.Module): # 
    def __init__(self, distance_type: str = "cosine", eps: float = 1e-6):
        super().__init__()
        self.distance_type = distance_type.lower()
        self.eps = eps

    def _compute_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: [U, D]   Prototype
        B: [B, D]   feature
        return: [U, B]  dist matrix
        """
        if self.distance_type == "cosine":
            sim = A @ B.t()                     
            dist = 1.0 - sim                    
        elif self.distance_type == "euclidean":
            dist = torch.cdist(A, B, p=2)       
        elif self.distance_type == "l1":
            dist = torch.cdist(A, B, p=1)             
        else:
            raise ValueError(f"Unsupported distance_type: {self.distance_type}")
        return dist

    def forward(self, semantic_features: torch.Tensor,
                      labels: torch.Tensor,
                      semantic_prototypes: torch.Tensor) -> torch.Tensor:

        device = semantic_features.device
        unique_label = labels.unique(sorted=True)
        Cb = semantic_prototypes[unique_label.long()]  # [U, D]

        mask = (labels.unsqueeze(0) == unique_label.unsqueeze(1)).float()

        cnt = mask.sum(dim=1)
        valid_cls_mask = (cnt >= 2)

        if not valid_cls_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        dist_matrix = self._compute_distance(Cb, semantic_features)  # [U, B]
        dist_sum = (dist_matrix * mask).sum(dim=1)
        cls_means = dist_sum / cnt
        
        target_radius = cls_means.detach()
        diff = dist_matrix - target_radius.unsqueeze(1)
        sq_diff = (diff ** 2) * mask

        cls_variances = sq_diff.sum(dim=1) / cnt
        
        # Variance -> Standard Deviation
        cls_stds = torch.sqrt(cls_variances + self.eps)
        valid_stds = cls_stds[valid_cls_mask]

        return valid_stds.mean()


class Geometry_Mean_Std_Consistency(nn.Module):
    def __init__(self, distance_type: str = "cosine", lambda_std: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.distance_type = distance_type.lower()
        self.lambda_std = lambda_std
        self.eps = eps


    def _compute_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: [U, D]   Prototypes
        B: [B, D]   Features
        return: [U, B] dist matrix
        """
        if self.distance_type == "cosine":
            sim = A @ B.t()
            dist = 1.0 - sim                    
        elif self.distance_type == "euclidean":
            dist = torch.cdist(A, B, p=2)       
        elif self.distance_type == "l1":
            dist = torch.cdist(A, B, p=1)             
        else:
            raise ValueError(f"Unsupported distance_type: {self.distance_type}")
        return dist


    def forward(self, semantic_features: torch.Tensor,
                      labels: torch.Tensor,
                      semantic_prototypes: torch.Tensor) -> torch.Tensor:
        
        device = semantic_features.device
        unique_label = labels.unique(sorted=True)
        Cb = semantic_prototypes[unique_label.long()] 
        
        mask = (labels.unsqueeze(0) == unique_label.unsqueeze(1)).float()
        dist_matrix = self._compute_distance(Cb, semantic_features)
        
        counts = mask.sum(dim=1)

        valid_class_mask = (counts >= 2).float()
        num_valid_classes = valid_class_mask.sum()
        
        if num_valid_classes == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # sum_dist: [U]
        sum_dist_per_class = (dist_matrix * mask).sum(dim=1)
        mean_dist_per_class = sum_dist_per_class / (counts)
        
        # mu : [U, 1]
        mu_expanded = mean_dist_per_class.detach().unsqueeze(1)
        diff_sq = ((dist_matrix - mu_expanded) * mask) ** 2
        
        # Variance/Std : [U]
        var_per_class = diff_sq.sum(dim=1) / (counts)
        std_per_class = torch.sqrt(var_per_class + self.eps)
        
        loss_mean = (mean_dist_per_class * valid_class_mask).sum() / num_valid_classes
        loss_std = (std_per_class * valid_class_mask).sum() / num_valid_classes
        
        print(f"Valid Classes: {int(num_valid_classes.item())}/{len(unique_label)}, "
              f"Mean Loss: {loss_mean.item():.4f}, Std Loss: {loss_std.item():.4f}")
        return loss_mean + self.lambda_std * loss_std

