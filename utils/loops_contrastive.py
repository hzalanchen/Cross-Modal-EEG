import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .util import AverageMeter, accuracy


def inference(image_features, eeg_features):
    similarity = (100.0 * eeg_features @ image_features.T).softmax(dim = -1)
    return similarity


def train(epoch, device, train_loader, module_list, criterions, optimizers, opt):
    """training"""
    ConLoss = AverageMeter()
    GeoLoss1 = AverageMeter()

    # model
    module_list.train()
    visual_net = module_list[0]
    eeg_net = module_list[1]
    
    # criterion
    conloss = criterions[0]
    if opt.geo_loss: geoloss1 = criterions[1]
    
    # optimizer
    net_optim = optimizers[0]

    for idx, (image, eeg, labels) in enumerate(train_loader):
        
        image = image.to(device)
        eeg = eeg.to(device)
        labels = labels.to(device)
        batch_size = image.size()[0]

        # First Forward
        image_s= visual_net(image)
        eeg_s= eeg_net(eeg)

        image_s = image_s / image_s.norm(dim=-1, keepdim=True)
        eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)
        
        # EEG prototype update
        if opt.geo_loss:
            with torch.no_grad():
                eeg_net.update_memory(eeg_s, labels)

        if opt.main_loss == 'conloss':
            con_loss = conloss(image_s, eeg_s, eeg_net.logit_scale)
        elif opt.main_loss == 'supconloss':
            con_loss = conloss(image_s, eeg_s, labels, eeg_net.logit_scale)
        
        if opt.geo_loss: 
            dis_var = geoloss1(image_s, labels, eeg_net.eeg_prototypes.detach())
            loss = con_loss + opt.lambda1 * dis_var
        else:
            loss = con_loss
            dis_var = torch.Tensor([0.])
        
        # total loss
        net_optim.zero_grad()
        loss.backward()
        net_optim.step()
        
        ConLoss.update(con_loss.item())
        GeoLoss1.update(dis_var.item())

    print('* Training Epoch {epoch} finished: ConLoss: {loss:.3f}'.format(epoch = epoch , loss=ConLoss.avg))
    return ConLoss.avg, GeoLoss1.avg


def validate(epoch, device, val_loader, val_images, module_list, criterions, opt = None):
    """validation"""
    top1 = AverageMeter()
    top5 = AverageMeter()
    ConLoss = AverageMeter()

    # switch to evaluate mode
    module_list.eval()
    visual_net = module_list[0]
    eeg_net = module_list[1]

    # criterion
    conloss = criterions[0]
    geoloss1 = criterions[1]
    geoloss2 = criterions[2]


    with torch.no_grad():
        image_concept= visual_net(val_images)
        image_concept = image_concept / image_concept.norm(dim=-1, keepdim=True)
        for idx, (image, eeg, target) in enumerate(val_loader):

            image = image.to(device)
            eeg = eeg.to(device)
            target = target.to(device)
            batch_size = target.size()[0]

            # compute output
            image_s= visual_net(image)
            eeg_s= eeg_net(eeg)
            eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)
            # cal
            con_loss = conloss(image_s, eeg_s, target)
            similarity = inference(image_concept, eeg_s)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(similarity, target, topk=(1, 5))
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            ConLoss.update(con_loss.item())

        print(' * Val Epoch {epoch} finished: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% SupConLoss {SupConLoss.avg:.4f}'.format(epoch = epoch, top1=top1, top5=top5, SupConLoss=SupConLoss))
    return top1.avg, top5.avg, ConLoss.avg


def test(epoch, device, test_loader, test_images, module_list, opt=None):
    """testing"""

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    module_list.eval()
    visual_net = module_list[0]
    eeg_net = module_list[1]

    with torch.no_grad():
        image_s = visual_net(test_images)
        image_s = image_s / image_s.norm(dim=-1, keepdim=True)
        for idx, (eeg, target) in enumerate(test_loader):

            eeg = eeg.to(device)
            target = target.to(device)
            batch_size = target.size()[0]
            # compute output
            eeg_s= eeg_net(eeg)
            eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)
            # cal
            similarity = inference(image_s, eeg_s)
            # measure accuracy
            acc1, acc5 = accuracy(similarity, target, topk=(1, 5))
            
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
        print(' * Test finished: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg
