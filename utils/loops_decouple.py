import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .util import AverageMeter, accuracy

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
    GeoLoss1 = AverageMeter()

    # model
    module_list.train()
    visual_net = module_list[0]
    eeg_net = module_list[1]
    visual_recnet = module_list[2]
    eeg_recnet = module_list[3]

    # criterion
    conloss = criterions[0]
    recloss = criterions[1]
    visual_mi_net = criterions[2]
    eeg_mi_net = criterions[3]
    if opt.geo_loss: geoloss1 = criterions[4]

    # optimizer
    net_optim = optimizers[0]
    visual_mioptim = optimizers[1]
    eeg_mioptim = optimizers[2]

    for idx, (image, eeg, labels) in enumerate(train_loader):
        image = image.to(device)
        eeg = eeg.to(device)
        labels = labels.to(device)
        batch_size = image.size()[0]

        image_feature, image_s, image_d = visual_net(image)
        eeg_feauture, eeg_s, eeg_d = eeg_net(eeg)

        image_s = image_s / image_s.norm(dim=-1, keepdim=True)
        image_d = image_d / image_d.norm(dim=-1, keepdim=True)
        eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)
        eeg_d = eeg_d / eeg_d.norm(dim=-1, keepdim=True)

        # EEG prototype update
        if opt.geo_loss:
            with torch.no_grad():
                eeg_net.update_memory(eeg_s, labels)

        # Stage1: loglikeli
        for _ in range(4):
            visual_mi_net.train()
            eeg_mi_net.train()
            image_li = visual_mi_net.learning_loss(image_d.detach(), image_s.detach())
            eeg_li = eeg_mi_net.learning_loss(eeg_d.detach(), eeg_s.detach())

            visual_mioptim.zero_grad()
            eeg_mioptim.zero_grad()

            image_li.backward()
            eeg_li.backward()

            visual_mioptim.step()
            eeg_mioptim.step()


        # Stage2: MI Maximization
        if opt.main_loss == 'conloss':
            con_loss = conloss(image_s, eeg_s, eeg_net.logit_scale)
        elif opt.main_loss == 'supconloss':
            con_loss = conloss(image_s, eeg_s, labels, eeg_net.logit_scale)

        # Reconstruction Loss(Cyclic Consistency) + MI Minimization
        visual_mi_net.eval()
        eeg_mi_net.eval()

        image_miloss = visual_mi_net(image_d, image_s)
        eeg_miloss = eeg_mi_net(eeg_d, eeg_s)

        #Reconstruction Loss.
        image_hat = visual_recnet(image_d, eeg_s)
        eeg_hat = eeg_recnet(eeg_d, image_s)

        visual_rec_loss = recloss(image_hat, image_feature)
        eeg_rec_loss = recloss(eeg_hat, eeg_feauture)

        if opt.geo_loss:
            dis_var = geoloss1(image_s, labels, eeg_net.eeg_prototypes.detach())
            loss = con_loss + opt.lambda1 * dis_var + opt.lambda3 * (image_miloss + eeg_miloss)  +  opt.lambda4 * (visual_rec_loss +  eeg_rec_loss)
        else:
            loss =  con_loss + opt.lambda3 * (image_miloss + eeg_miloss)  +  opt.lambda4 * (visual_rec_loss +  eeg_rec_loss)
            dis_var = torch.Tensor([0.])
        # print(f"The Hyperparameters: {opt.lambda1} and {opt.lambda2} and {opt.lambda3} and {opt.lambda4}")
        # backward 
        net_optim.zero_grad()
        loss.backward()
        net_optim.step()

        ConLoss.update(con_loss.item())
        GeoLoss1.update(dis_var.item())
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
                   con_loss.item(), dis_var.item(),
                   image_miloss.item(), eeg_miloss.item(),
                   visual_rec_loss.item(), eeg_rec_loss.item()))
            sys.stdout.flush()
    print('* Training Epoch {epoch} finished: ConLoss: {loss:.3f}'.format(epoch = epoch , loss=ConLoss.avg))
    return ConLoss.avg, GeoLoss1.avg, ImageMiLoss.avg, EEGMiLoss.avg, VisualRecLoss.avg, EEGRecLoss.avg 

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
    geoloss1 = criterions[4]
    geoloss2 = criterions[5]

    with torch.no_grad():
        _, image_concept, _= visual_net(val_images)
        image_concept = image_concept / image_concept.norm(dim=-1, keepdim=True)
        for idx, (image, eeg, target) in enumerate(val_loader):

            image = image.to(device)
            eeg = eeg.to(device)
            target = target.to(device)
            batch_size = target.size()[0]

            # compute output
            _, image_s, _= visual_net(image)
            _, eeg_s, _= eeg_net(eeg)
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
            # compute output
            _, eeg_s, _ = eeg_net(eeg)
            eeg_s = eeg_s / eeg_s.norm(dim=-1, keepdim=True)
            # cal
            similarity = inference(image_s, eeg_s)
            # measure accuracy
            acc1, acc5 = accuracy(similarity, target, topk=(1, 5))
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
        print(' * Test finished: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


