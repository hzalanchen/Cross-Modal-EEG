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
    # x_dim 512 y_dim 256
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
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
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean() #.sum(dim=1).mean(dim=0)
    

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


def cal_dis_var(image_features, eeg_features, labels):
        '''
        Input: L2 Norm feauture, image_features, eeg_features
        '''
        device = torch.device('cuda') if image_features.is_cuda else torch.device('cpu')
        batch_size = image_features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)            
        # eye = torch.eye(batch_size).to(device)
        # inverse_eye = 1 - eye

        # modality gap
        # Euclidean distance
        dis_matrix = torch.cdist(image_features, eeg_features, p=2)# .pow(2)

        same_class_distances = []
        for i in range(mask.shape[0]):
            # Use a mask to select the distance values of samples from the same category in the current row
            distances = dis_matrix[i][mask[i] == 1]
            # Add to the list
            if distances.numel() > 1:
                same_class_distances.append(distances)
        
        variances = []
        for distances in same_class_distances:
            variance = torch.var(distances, unbiased=True)
            variances.append(variance)
        variances_tensor = torch.tensor(variances)

        return torch.mean(variances_tensor)


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

