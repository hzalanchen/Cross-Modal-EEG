import os 
import torch
import torch.nn.functional as F
import random
import numpy as np
import wandb
import shutil
import logging
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return 1e-6 + (1e-4 - 1e-6) * self.current_step / self.warmup_steps
        else:
            return 1e-4

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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


def save_checkpoint(state, outdir, filename):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, filename)  
    torch.save(state, checkpoint_file) 


def save_checkpoint_copy(state, is_best, outdir):
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file) 
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def modality_gap_sample(image_embedding, eeg_embedding):
    '''
    Input: L2 Norm feature
    '''
    feature_gap = image_embedding.unsqueeze(1) - eeg_embedding.unsqueeze(0)
    modality_distance = torch.norm(feature_gap, p = 2, dim = 2)

    
    return modality_distance


def modality_gap_bs(image_embedding, eeg_embedding):
    '''
    input: Tensor
    '''
    batch_size = image_embedding.size()[0]
    center_gap = image_embedding.mean(dim = 0) - eeg_embedding.mean(dim = 0)
    modality_gap = F.normalize(center_gap, p = 2, dim = -1)

    modality_distance = torch.norm(center_gap)
   
    return modality_distance


# Record
class WandbLogger:
    def __init__(self, project_name, run_name, config):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config

    def initialize(self):
        wandb.init(project=self.project_name, name= self.run_name ,config=self.config)

    def log_metrics(self, step = None, **kwargs):
        if step is not None:
            wandb.log(kwargs, step=step)
        else:
            wandb.log(kwargs)

    def finish(self):
        wandb.finish()


def setLogger(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(logfile, mode = 'a')
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
