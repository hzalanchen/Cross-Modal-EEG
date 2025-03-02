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


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            m.bias.data.fill_(0.001)
    elif classname.find('BatchNorm') != -1:
        pass


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Projector(nn.Module):
    def __init__(self, dim_in, dim_hidden = 2048, dim_out = 1654):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.activate = nn.GELU()
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, dim_out)


    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        out = self.linear3(x)
        return out

