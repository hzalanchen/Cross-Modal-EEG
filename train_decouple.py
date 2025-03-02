import os
import csv
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from utils.datasets import *
from utils.loss import CLUB, CLUBSample, ConLoss, SupConLoss ,Geometry_Variance
from utils.loops_decouple import train, validate, test 
from utils.util import save_checkpoint, setLogger, WandbLogger, setup_seed
from config import parse_args
from einops import rearrange

from models import eeg
from models import visual

# set cuda
gpus = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.deterministic = True
else:
    device = torch.device('cpu')
    gpu_index = -1 


class VE_SID():
    def __init__(self, opt, nsub):
        self.opt = opt
        self.nSub = nsub
        self.batch_size_val = 200
        self.batch_size_test = 200

        self.eeg_data_path =  '/chz/data/Things-EEG/Preprocessed_data_250Hz/'
        self.image_data_path = './image_features/'
        # EEG feature dim: 1440 , Image feature dim : 1024
        self.eeg_feature_dim = 1440
        self.image_feature_dim = 1024

        # model
        self.EEG_Net = eeg.EEG_Net(self.opt).to(device)
        self.Visual_Net = visual.Visual(self.opt).to(device)
        self.EEG_Rec = eeg.EEG_Decoder(self.opt, self.eeg_feature_dim).to(device)
        self.Visual_Rec = visual.Visual_Decoder(self.opt, self.image_feature_dim).to(device)
        self.modules = nn.ModuleList([self.Visual_Net, self.EEG_Net, self.Visual_Rec, self.EEG_Rec])
        
        # criterion
        if opt.main_loss == 'conloss':
            self.main_loss = ConLoss()
        elif opt.main_loss == 'supconloss':
            self.main_loss = SupConLoss()
        self.recloss = nn.MSELoss()
        self.visual_mi_net = CLUBSample(opt.output_dim_D, opt.output_dim_S, 512).to(device)
        self.eeg_mi_net = CLUBSample(opt.output_dim_D, opt.output_dim_S, 512).to(device)
        if opt.geo_loss:
            self.geoloss1 = Geometry_Variance()
            self.criterions = [self.main_loss, self.recloss, self.visual_mi_net, self.eeg_mi_net, self.geoloss1]
            geo_info = f'geovar_{self.opt.lambda1}'
        else:
            self.criterions = [self.main_loss, self.recloss, self.visual_mi_net, self.eeg_mi_net]
            geo_info = f''

        # optimizers
        self.net_optim = optim.AdamW(self.modules.parameters(),lr=self.opt.lr, betas=(0.9,0.98))
        self.visual_mioptim = optim.Adam(self.visual_mi_net.parameters(), lr = self.opt.mi_lr)
        self.eeg_mioptim = optim.Adam(self.eeg_mi_net.parameters(), lr = self.opt.mi_lr)
        # self.semantic_mi_net = optim.AdamW(self.semantic_mi_net.parameters(), lr = self.opt.mi_lr)
        self.optimizers = [self.net_optim, self.visual_mioptim, self.eeg_mioptim]
        #self.Con_Record = []
        #self.IM_MI = []
        #self.EEG_MI = []
        #self.Semantic_MI = []
        #self.IM_Rec = []
        #self.EEG_Rec = []

        self.save_model_path = f'./saves/checkpoints/Decouple_Sub{format(self.nSub, "02")}_{self.opt.main_loss}_MI_{self.opt.lambda3}_Rec_{self.opt.lambda4}_{geo_info}/'
        if not os.path.exists(self.save_model_path): os.makedirs(self.save_model_path)

        self.results_path = f'./saves/results/Decouple_Sub{format(self.nSub, "02")}_{self.opt.main_loss}_MI_{self.opt.lambda3}_Rec_{self.opt.lambda4}_{geo_info}/'
        if not os.path.exists(self.results_path):os.makedirs(self.results_path)
        self.logger = setLogger(os.path.join(self.results_path, f'sub_{format(self.nSub, "02")}_logs'))
        self.wandblogger = WandbLogger("VE-SID", f'Decouple_Sub{format(self.nSub, "02")}_MI_{self.opt.lambda3}_Rec_{self.opt.lambda4}__{geo_info}', self.opt)
        self.wandblogger.initialize()

        
    def initial_EEG_prototype(self):  
        eeg_datasets_path = os.path.join(self.eeg_data_path, 'sub-'+format(self.nSub, '02'), 'preprocessed_eeg_training.npy')
        eeg_datasets = np.load(eeg_datasets_path, allow_pickle=True)['preprocessed_eeg_data']
        eeg_datasets = np.mean(eeg_datasets, axis = 1)        
        eeg_datasets = torch.from_numpy(eeg_datasets).float().to(device)
        #(16540 63 250) 
        print(eeg_datasets.shape)
        batch_size = 40
        eeg_prototype_list = []
        self.EEG_Net.eval()
        with torch.no_grad():
            for i in range(0, len(eeg_datasets), batch_size):
                batch_eegs = eeg_datasets[i : i + batch_size]
                _, batch_eegs_features, _ = self.EEG_Net(batch_eegs)
                eeg_prototype_list.append(batch_eegs_features)
            
        eeg_prototypes = torch.cat(eeg_prototype_list, dim = 0)
        print("eeg_prototypes tensor features shape", eeg_prototypes.shape)
        eeg_prototypes = rearrange(eeg_prototypes, '(b h) f -> b h f', h = 10)
        eeg_prototypes = eeg_prototypes.mean(dim = 1)
        # print("eeg_prototypes shape2", eeg_prototypes.shape)
        eeg_prototypes = eeg_prototypes / eeg_prototypes.norm(dim = -1, keepdim=True)
        self.EEG_Net.register_buffer('eeg_prototypes', eeg_prototypes)
        # self.EEG_Net.eeg_prototypes.copy_(eeg_prototypes)
        del eeg_datasets, eeg_prototype_list, eeg_prototypes
        print(f"The shape of EEG prototypes: {self.EEG_Net.eeg_prototypes.shape}")

     
    def get_datasets(self):
        image_eeg_pairs_train = get_image_eeg_pair_Rep(self.image_data_path, self.eeg_data_path, sub=format(self.nSub, '02'))
        train_datasets = TrainDataset(image_eeg_pairs_train)

        test_img_feature, test_eeg = get_test_eeg_Rep(self.image_data_path, self.eeg_data_path, sub=format(self.nSub, '02'))
        test_datasets = TestDataset(test_eeg)

        trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = self.opt.batch_size, shuffle = True)
        testloader = torch.utils.data.DataLoader(test_datasets, batch_size = self.batch_size_test, shuffle = False)

        return train_datasets, test_datasets, trainloader, testloader, test_img_feature


    def loops(self):

        train_datasets, test_datasets, trainloader, testloader, test_img_feature = self.get_datasets()
        print(f"The number of train_datasets: {len(train_datasets)}. The number of test_datasets: {len(test_datasets)}")
        print(f"The dimension of visual feature: {train_datasets[0][0].shape}, The dimension of eeg feature: {train_datasets[0][1].shape}")
        print(f"The concept shape of test image feature {test_img_feature.shape}")
        self.logger.info(f"The number of train_datasets: {len(train_datasets)}. The number of test_datasets: {len(test_datasets)}")
        self.logger.info(f"The dimension of visual feature: {train_datasets[0][0].shape}, The dimension of eeg feature: {train_datasets[0][1].shape}")
        self.logger.info(f"The concept shape of test image feature {test_img_feature.shape}")
        test_img_feature = test_img_feature.to(device)
        
        results = []
        model_dict = {}
        best_record = {}
        best_record['test_top1_acc'] = 0
        for epoch in range(1, self.opt.train_epochs + 1): 
            print(f"This is training phase, Epoch : {epoch}, learning_rate : {self.optimizers[0].param_groups[0]['lr']}")
            train_conloss, dis_var, v_miloss, e_miloss, v_recloss, e_recloss= train(epoch, device, trainloader, self.modules, self.criterions, self.optimizers, self.opt)
            test_top1_acc, test_top5_acc = test(epoch, device, testloader, test_img_feature, self.modules, self.opt)

            if (epoch + 1) % 10 == 0:
                model_dict['EEG_Net'] = self.EEG_Net.state_dict()
                model_dict['Viusal_Net'] = self.Visual_Net.state_dict()
                save_checkpoint(model_dict, self.save_model_path, f'models_checkpoints_{epoch + 1}.pth')

            if test_top1_acc > best_record['test_top1_acc']:
                best_record['test_top1_acc'] = test_top1_acc
                best_record['test_top5_acc'] = test_top5_acc
                best_record['epoch'] = epoch
                best_record['lr'] = self.optimizers[0].param_groups[0]['lr']

            epoch_results = {
                "epoch": epoch,
                "train_conloss":train_conloss,
                "test_top1_acc":test_top1_acc,
                "test_top5_acc":test_top5_acc,
            }

            results.append(epoch_results)

            self.wandblogger.log_metrics(test_acc1 = test_top1_acc, test_acc5 = test_top5_acc, train_conloss = train_conloss)
            self.logger.info(f"Subject {self.nSub} : Epoch {epoch}, test_acc1 :{test_top1_acc}, test_acc5:{test_top5_acc}, train_conloss = {train_conloss} \
                             geo_loss1: {dis_var}, visual_mi_loss: {v_miloss}, eeg_mi_loss: {e_miloss}, visual_rec_loss: {v_recloss}, eeg_rec_loss: {e_recloss}")
        self.logger.info(f"* Subject {self.nSub} The best test Top1 acc is {best_record['test_top1_acc']}; Top5 acc is {best_record['test_top5_acc']}, epoch is {best_record['epoch']}")
        self.logger.handlers.clear()
        self.wandblogger.finish()

        # Save results to a CSV file
        results_file = os.path.join(self.results_path, f'sub_{format(self.nSub, "02")}_result.csv')
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {results_file}')
        return results

def main():
    opt = parse_args()
    num_sub = opt.num_sub
    cal_num = 0
    for i in range(1, num_sub + 1):
        cal_num += 1
        print(f'Subject {i}, Seed is {opt.seed}')
        setup_seed(opt.seed)
        ie = VE_SID(opt, i)
        results= ie.loops()

if __name__ == "__main__":
    main()