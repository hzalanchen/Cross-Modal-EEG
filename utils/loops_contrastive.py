import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .util import AverageMeter, accuracy, compute_grad_norm


def inference(image_features, eeg_features):
    similarity = (100.0 * eeg_features @ image_features.T).softmax(dim = -1)
    return similarity


def train(epoch, device, train_loader, module_list, criterions, optimizers, opt):
    """training"""
    ConLoss = AverageMeter()
    GeoLoss = AverageMeter()

    # model
    module_list.train()
    visual_net, eeg_net = module_list

    # criterion
    conloss = criterions[0]
    if opt.geo_loss: geoloss = criterions[-1]
    
    # optimizer
    net_optim = optimizers[0]

    for idx, (image, eeg, labels) in enumerate(train_loader):
        batch_size = image.size()[0]
        image = image.to(device)
        eeg = eeg.to(device)
        labels = labels.to(device)

        # First Forward
        image_s = visual_net(image)
        eeg_s = eeg_net(eeg)

        image_s = image_s / image_s.norm(dim=-1, keepdim=True)
        eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)
        
        if opt.main_loss == 'conloss': con_loss = conloss(image_s, eeg_s, eeg_net.logit_scale)
        elif opt.main_loss == 'supconloss': con_loss = conloss(image_s, eeg_s, labels, eeg_net.logit_scale)
        
        if opt.geo_loss:
            geo_last_epochs = getattr(opt, "geo_last_epochs", 25)
            geo_active = (epoch > opt.train_epochs - geo_last_epochs)
            if geo_active:
                eeg_net.set_alpha_value(0.99)
                eeg_net.update_memory(eeg_s, labels)
                geo_lossValue = geoloss(image_s, labels, eeg_net.eeg_prototypes.detach())
                loss = (
                    con_loss + 
                    opt.lambda1 * geo_lossValue 
                )
            else: 
                eeg_net.set_mom_sched(epoch)
                eeg_net.update_memory(eeg_s, labels)
                loss = con_loss 
                geo_lossValue = torch.zeros((), device=image.device)
        else:
            loss = con_loss 
            geo_lossValue = torch.zeros((), device=image.device)

        # total loss
        net_optim.zero_grad()
        loss.backward()
        net_optim.step()
        
        ConLoss.update(con_loss.item())
        GeoLoss.update(geo_lossValue.item())

        if idx % opt.print_freq == 0:
            print('Trainig phase : Epoch: [{0}][{1}/{2}] '
                  'Con loss: {3:.4f}, Geo loss1: {4:.6f} '.format(
                   epoch, idx, len(train_loader),
                   con_loss.item(), geo_lossValue.item()))
            sys.stdout.flush()
    print('* Training Epoch {epoch} finished: ConLoss: {loss:.3f}'.format(epoch = epoch , loss=ConLoss.avg))
    return ConLoss.avg, GeoLoss.avg


def test(epoch, device, test_loader, test_images, module_list, opt=None, phase='Test'):
    """testing or validation"""

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
            acc1, acc5 = accuracy(similarity, target, topk=(1, 5)) # acc
            # measure accuracy
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
        print(' * {phase} finished: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'.format(phase=phase, top1=top1, top5=top5))
    return top1.avg, top5.avg
