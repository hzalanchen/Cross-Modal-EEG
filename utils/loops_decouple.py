import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .util import AverageMeter, accuracy, l2norm, compute_grad_norm


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)


def inference(image_features, eeg_features):
    similarity = (100 * eeg_features @ image_features.T).softmax(dim = -1)
    return similarity


def train(epoch, device, train_loader, module_list, criterions, optimizers, opt):
    """training"""
    VisualRecLoss = AverageMeter()
    EEGRecLoss = AverageMeter()
    ConLoss = AverageMeter()
    ImageMiLoss = AverageMeter()
    EEGMiLoss = AverageMeter()
    GeoLoss = AverageMeter()

    # model
    module_list.train()
    visual_net, eeg_net, visual_recnet, eeg_recnet = module_list
    
    # criterion
    conloss, recloss, visual_mi_net, eeg_mi_net = criterions[: 4]
    if opt.geo_loss: geoloss = criterions[-1]

    # optimizer
    net_optim, visual_mioptim, eeg_mioptim = optimizers

    for idx, (image, eeg, labels) in enumerate(train_loader):
        batchsize = image.size()[0]
        image = image.to(device)
        eeg = eeg.to(device)
        labels = labels.to(device)

        image_feature, image_s, image_d = visual_net(image)
        eeg_feauture, eeg_s, eeg_d = eeg_net(eeg)

        image_s = l2norm(image_s); image_d = l2norm(image_d)
        eeg_s  = l2norm(eeg_s); eeg_d = l2norm(eeg_d)

        # Stage1: loglikeli
        for _ in range(10):
            visual_mi_net.train(); eeg_mi_net.train()
            image_li = visual_mi_net.learning_loss(image_d.detach(), image_s.detach())
            eeg_li = eeg_mi_net.learning_loss(eeg_d.detach(), eeg_s.detach())

            visual_mioptim.zero_grad(); eeg_mioptim.zero_grad()
            image_li.backward(); eeg_li.backward()
            visual_mioptim.step(); eeg_mioptim.step()

        # Stage2: MI Maximizations
        if opt.main_loss == 'conloss': con_loss = conloss(image_s , eeg_s , eeg_net.logit_scale)
        elif opt.main_loss == 'supconloss': con_loss = conloss(image_s , eeg_s , labels, eeg_net.logit_scale)

        # Reconstruction Loss(Cyclic Consistency) + MI Minimization
        visual_mi_net.eval(); eeg_mi_net.eval()
        image_miloss = visual_mi_net(image_d, image_s)
        eeg_miloss = eeg_mi_net(eeg_d, eeg_s)

        #Reconstruction Loss.
        image_hat = visual_recnet(image_d, eeg_s)
        eeg_hat = eeg_recnet(eeg_d, image_s)

        # image_hat = l2norm(image_hat); eeg_hat = l2norm(eeg_hat)
        target_img_feature = l2norm(image_feature).detach(); target_eeg_feature = l2norm(eeg_feauture).detach()

        visual_rec_loss = recloss(image_hat, target_img_feature)
        eeg_rec_loss = recloss(eeg_hat, target_eeg_feature)
        visual_rec_loss = visual_rec_loss.sum(dim=1).mean()
        eeg_rec_loss = eeg_rec_loss.sum(dim=1).mean()


        if opt.geo_loss:
            geo_last_epochs = getattr(opt, "geo_last_epochs", 25)
            geo_active = (epoch > opt.train_epochs - geo_last_epochs)
            if geo_active:
                eeg_net.set_alpha_value(0.99)
                eeg_net.update_memory(eeg_s, labels)
                geo_lossValue = geoloss(image_s, labels, eeg_net.eeg_prototypes.detach())
                loss = ( 
                    con_loss + 
                    opt.lambda1 * geo_lossValue + 
                    opt.lambda3 * (image_miloss + eeg_miloss)  +  
                    opt.lambda4 * (visual_rec_loss +  eeg_rec_loss)
                )
            else:
                eeg_net.set_mom_sched(epoch)
                eeg_net.update_memory(eeg_s, labels)
                loss = (
                    con_loss + 
                    opt.lambda3 * (image_miloss + eeg_miloss)  +  
                    opt.lambda4 * (visual_rec_loss +  eeg_rec_loss)
                )
                geo_lossValue = torch.zeros((), device=image.device)
        else:
            loss = (
                con_loss + 
                opt.lambda3 * (image_miloss + eeg_miloss)  +  
                opt.lambda4 * (visual_rec_loss +  eeg_rec_loss) 
            )
            geo_lossValue = torch.zeros((), device=image.device)
            
        # backward 
        net_optim.zero_grad()
        loss.backward()
        net_optim.step()

        ConLoss.update(con_loss.item())
        GeoLoss.update(geo_lossValue.item())
        VisualRecLoss.update(visual_rec_loss.item())
        EEGRecLoss.update(eeg_rec_loss.item())
        ImageMiLoss.update(image_miloss.item())
        EEGMiLoss.update(eeg_miloss.item())

        # print info
        if idx % opt.print_freq == 0:
            print('Trainig phase : Epoch: [{0}][{1}/{2}] '
                  'Con loss: {3:.4f}, Geo loss1: {4:.6f} '
                  'Visual Mi loss: {5:.4f}, EEG Mi loss {6:.4f}, '
                  'Visual Rec loss: {7:.4f}, EEG Rec loss {8:.4f}, '.format(
                   epoch, idx, len(train_loader),
                   con_loss.item(), geo_lossValue.item(),
                   image_miloss.item(), eeg_miloss.item(),
                   visual_rec_loss.item(), eeg_rec_loss.item()))
            sys.stdout.flush()
    print('* Training Epoch {epoch} finished: ConLoss: {loss:.3f}'.format(epoch = epoch , loss=ConLoss.avg))
    return ConLoss.avg, GeoLoss.avg, ImageMiLoss.avg, EEGMiLoss.avg, VisualRecLoss.avg, EEGRecLoss.avg 


def test(epoch, device, test_loader, test_images, module_list, opt=None, phase='Test'):
    """testing or validation"""

    top1 = AverageMeter()
    top5 = AverageMeter()

    module_list.eval()
    visual_net = module_list[0]
    eeg_net = module_list[1]

    with torch.no_grad():
        _, image_s, _ = visual_net(test_images)
        image_s = image_s / image_s.norm(dim=-1, keepdim=True)
        for idx, (eeg, target) in enumerate(test_loader):

            eeg = eeg.to(device)
            target = target.to(device)
            batch_size = target.size()[0]
            
            _, eeg_s, _ = eeg_net(eeg)
            eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)

            similarity = inference(image_s, eeg_s)
            acc1, acc5 = accuracy(similarity, target, topk=(1, 5))
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
        print(' * {phase} finished: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'.format(phase=phase, top1=top1, top5=top5))
        
    return top1.avg, top5.avg


