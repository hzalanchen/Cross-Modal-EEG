import os
import csv
import copy
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.datasets import *
from utils.util import *
from utils.loss import CLUBSample_CO_Estimate, CLUBSample, ConLoss, SupConLoss, Geometry_Gaps_Consistency 
from utils.loops_decouple import train, test 
from config import parse_args 
from einops import rearrange
from models import eeg
from models import visual
from models.modules import *

# set cuda
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    gpu_index = -1 


class VE_SID():
    def __init__(self, opt, nsub):
        self.opt = opt
        self.nSub = nsub
        self.batch_size_val = 200
        self.batch_size_test = 200

        self.eeg_data_path =  '/chz/data/VISUALEEG/Things-EEG/Preprocessed_data_250Hz/'
        self.image_data_path = './image_features/'
        # EEG feature dim: 1024 , Image feature dim : 1024
        self.eeg_feature_dim = 1024
        self.image_feature_dim = 1024

        # model
        self.Visual_Net = visual.Visual(self.opt).to(device)
        self.EEG_Net = eeg.EEG_Net(self.opt).to(device)
        self.Visual_Rec = MLPDecoder(self.opt, self.image_feature_dim).to(device)
        self.EEG_Rec = MLPDecoder(self.opt, self.eeg_feature_dim).to(device)
        self.modules = nn.ModuleList([self.Visual_Net, self.EEG_Net, self.Visual_Rec, self.EEG_Rec])
        
        # criterion
        if opt.main_loss == 'conloss': self.main_loss = ConLoss()
        elif opt.main_loss == 'supconloss': self.main_loss = SupConLoss()
        self.recloss = nn.MSELoss(reduction='none')

        self.visual_mi_net = CLUBSample(opt.output_dim_D, opt.output_dim_S, 768).to(device)
        self.eeg_mi_net = CLUBSample(opt.output_dim_D, opt.output_dim_S, 768).to(device)

        if opt.geo_loss:
            self.geoloss = Geometry_Gaps_Consistency(self.opt.geo_loss_dis)
            self.criterions = [self.main_loss, self.recloss, self.visual_mi_net, self.eeg_mi_net, self.geoloss]
            geo_info = f'geogaps_last{self.opt.geo_last_epochs}epochs_{self.opt.geo_loss_dis}_{self.opt.lambda1}'
        else:
            self.criterions = [self.main_loss, self.recloss, self.visual_mi_net, self.eeg_mi_net]
            geo_info = f''

        # optimizers
        self.net_optim = optim.AdamW(self.modules.parameters(), lr=self.opt.lr)
        self.visual_mioptim = optim.AdamW(self.visual_mi_net.parameters(), lr=self.opt.mi_lr, betas=(0.5, 0.999))
        self.eeg_mioptim = optim.AdamW(self.eeg_mi_net.parameters(), lr=self.opt.mi_lr, betas=(0.5, 0.999))
        self.optimizers = [self.net_optim, self.visual_mioptim, self.eeg_mioptim]

        run_session = f"Decouple_{self.opt.exp_setting}_seed{self.opt.seed}_lr{self.opt.lr}_MI_{self.opt.lambda3}_Rec_{self.opt.lambda4}_{geo_info}"
        run_name = f'Decouple_{self.opt.exp_setting}_Sub{format(self.nSub, "02")}'
        self.save_model_path = f'./saves/checkpoints/{run_session}/{run_name}/'; ensure_path(self.save_model_path)
        self.results_path = f'./saves/results/{run_session}/{run_name}/' ; ensure_path(self.results_path)

        # logger
        self.logger = setLogger(os.path.join(self.results_path, f'sub_{format(self.nSub, "02")}_logs'))
        self.logger.info("================= Options ===================")
        for k, v in vars(self.opt).items(): self.logger.info(f'{str(k):<25}: {str(v)}')
        self.logger.info("=============================================")
        # self.wandblogger = WandbLogger("VE-SID-MR-2026-LOSO", run_name, self.opt)
        # self.wandblogger.initialize()

     
    def get_datasets(self):
        if self.opt.exp_setting == 'intra-subject':
            image_eeg_pairs_train, test_img_feature, test_eeg = get_train_test_datasets(self.image_data_path, self.eeg_data_path, sub=format(self.nSub, '02'))
            train_datasets = TrainDataset(image_eeg_pairs_train)
            test_datasets = TestDataset(test_eeg)

            trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=self.opt.batch_size, num_workers=8, shuffle=True)
            testloader = torch.utils.data.DataLoader(test_datasets, batch_size=self.batch_size_test, shuffle=False)

            return train_datasets, test_datasets, trainloader, testloader, test_img_feature
        
        elif self.opt.exp_setting == 'inter-subject':
            train_image_eeg_pairs, val_eeg, test_img_feature, test_eeg = get_train_val_test_datasets_LOSO(self.image_data_path, self.eeg_data_path, sub=format(self.nSub, '02'))
            train_datasets = TrainDataset(train_image_eeg_pairs)
            val_datasets = TestDataset(val_eeg)
            test_datasets = TestDataset(test_eeg)

            trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=self.opt.batch_size, num_workers=8, shuffle=True)
            valloader = torch.utils.data.DataLoader(val_datasets, batch_size=self.batch_size_val, shuffle=False)
            testloader = torch.utils.data.DataLoader(test_datasets, batch_size=self.batch_size_test, shuffle=False)
            
            return train_datasets, val_datasets, test_datasets, trainloader, valloader, testloader, test_img_feature


    def loops_intra_subject(self):
        train_datasets, test_datasets, trainloader, testloader, test_img_feature = self.get_datasets()
        print(f"The number of train_datasets: {len(train_datasets)}. The number of test_datasets: {len(test_datasets)}")
        print(f"The dimension of visual feature: {train_datasets[0][0].shape}, The dimension of eeg feature: {train_datasets[0][1].shape}")
        print(f"The concept shape of test image feature {test_img_feature.shape}")
        self.logger.info(f"The number of train_datasets: {len(train_datasets)}. The number of test_datasets: {len(test_datasets)}")
        self.logger.info(f"The dimension of visual feature: {train_datasets[0][0].shape}, The dimension of eeg feature: {train_datasets[0][1].shape}")
        self.logger.info(f"The concept shape of test image feature {test_img_feature.shape}")
        test_img_feature = test_img_feature.to(device)
        
        results = []
        ckpt_state = {
            'loss': float('inf'),  
            'epoch': None,
            'lr': None,
            'EEG_Net': None,
            'Visual_Net': None
        }
        for epoch in range(1, self.opt.train_epochs + 1): 
            current_lr = self.optimizers[0].param_groups[0]['lr']
            print(f"This is training phase, Epoch : {epoch}, learning_rate : {current_lr}")
            train_conloss, geo_loss, v_miloss, e_miloss, v_recloss, e_recloss = train(epoch, device, trainloader, self.modules, self.criterions, self.optimizers, self.opt)
            test_top1_acc, test_top5_acc = test(epoch, device, testloader, test_img_feature, self.modules, self.opt)

            if train_conloss < ckpt_state['loss']:
                ckpt_state['loss'] = train_conloss
                ckpt_state['epoch'] = epoch
                ckpt_state['lr'] = current_lr

                ckpt_state['EEG_Net'] = copy.deepcopy(self.EEG_Net.state_dict())
                ckpt_state['Visual_Net'] = copy.deepcopy(self.Visual_Net.state_dict())

                save_checkpoint(
                    ckpt_state,
                    self.save_model_path,
                    f'ckpt_best_trainconloss.pth'
                )

            epoch_results = {
                "epoch": epoch,
                "train_conloss": train_conloss,
                "test_top1_acc": test_top1_acc,
                "test_top5_acc": test_top5_acc,
            }
            results.append(epoch_results)

            self.logger.info(f"Subject {self.nSub} : Epoch {epoch}, test_acc1: {test_top1_acc}, test_acc5: {test_top5_acc}, train_conloss = {train_conloss}, "
                             f"geo_loss: {geo_loss}, visual_mi_loss: {v_miloss}, eeg_mi_loss: {e_miloss}, visual_rec_loss: {v_recloss}, eeg_rec_loss: {e_recloss}")

        # Test with best model
        if ckpt_state['EEG_Net'] is not None and ckpt_state['Visual_Net'] is not None:
            self.EEG_Net.load_state_dict(ckpt_state['EEG_Net'])
            self.Visual_Net.load_state_dict(ckpt_state['Visual_Net'])

            test_top1_acc, test_top5_acc = test(ckpt_state['epoch'], device, testloader, test_img_feature, self.modules, self.opt)

            print(f"[Test Results] Using best-train-loss weights from epoch {ckpt_state['epoch']}; "
                  f"Test top-1: {test_top1_acc}, Test top-5: {test_top5_acc}")
            self.logger.info(f"Subject {self.nSub} : Test results test_top_acc1: {test_top1_acc}, test_top_acc5: {test_top5_acc}")
            
            results.append({
                "best_epoch": ckpt_state['epoch'],
                "best_train_conloss": ckpt_state['loss'],
                "top1_acc_results": test_top1_acc,
                "top5_acc_results": test_top5_acc,
            })
        else:
            raise ValueError("No best-train-loss state captured")

        self.logger.handlers.clear()

        # Save results to a CSV file
        train_rows = results[:-1]
        final_row = results[-1]
        fieldnames = list(train_rows[0].keys()) + list(final_row.keys())

        results_file = os.path.join(self.results_path, f'sub_{format(self.nSub, "02")}_result.csv')
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(train_rows)
            writer.writerow(final_row)
        print(f'Results saved to {results_file}')
        return results


    def loops_inter_subject(self):
        train_datasets, val_datasets, test_datasets, trainloader, valloader, testloader, test_img_feature = self.get_datasets()
        print(f"The number of train_datasets: {len(train_datasets)}. The number of val_datasets: {len(val_datasets)}. The number of test_datasets: {len(test_datasets)}")
        print(f"The dimension of visual feature: {train_datasets[0][0].shape}, The dimension of eeg feature: {train_datasets[0][1].shape}")
        print(f"The concept shape of test image feature {test_img_feature.shape}")
        self.logger.info(f"The number of train_datasets: {len(train_datasets)}. The number of val_datasets: {len(val_datasets)}. The number of test_datasets: {len(test_datasets)}")
        self.logger.info(f"The dimension of visual feature: {train_datasets[0][0].shape}, The dimension of eeg feature: {train_datasets[0][1].shape}")
        self.logger.info(f"The concept shape of test image feature {test_img_feature.shape}")
        test_img_feature = test_img_feature.to(device)

        results = []
        ckpt_state = {
            'val_top1_acc': 0.0, 
            'epoch': None,
            'lr': None,
            'EEG_Net': None,
            'Visual_Net': None
        }
        
        for epoch in range(1, self.opt.train_epochs + 1):
            current_lr = self.optimizers[0].param_groups[0]['lr']
            print(f"This is training phase, Epoch : {epoch}, learning_rate : {current_lr}")
            train_conloss, geo_loss, v_miloss, e_miloss, v_recloss, e_recloss = train(epoch, device, trainloader, self.modules, self.criterions, self.optimizers, self.opt)
            # Validation
            val_top1_acc, val_top5_acc = test(epoch, device, valloader, test_img_feature, self.modules, self.opt, phase='Validation')
            # Test
            test_top1_acc, test_top5_acc = test(epoch, device, testloader, test_img_feature, self.modules, self.opt, phase='Test')


            if val_top1_acc > ckpt_state['val_top1_acc']:
                ckpt_state['val_top1_acc'] = val_top1_acc
                ckpt_state['epoch'] = epoch
                ckpt_state['lr'] = current_lr
                ckpt_state['EEG_Net'] = copy.deepcopy(self.EEG_Net.state_dict())
                ckpt_state['Visual_Net'] = copy.deepcopy(self.Visual_Net.state_dict())
                
                save_checkpoint(
                    ckpt_state,
                    self.save_model_path,
                    f'ckpt_best_val_acc.pth'
                )
            
            epoch_results = {
                "epoch": epoch,
                "train_conloss": train_conloss,
                "geo_loss": geo_loss,
                "v_miloss": v_miloss,
                "e_miloss": e_miloss,
                "v_recloss": v_recloss,
                "e_recloss": e_recloss,
                "val_top1_acc": val_top1_acc,
                "val_top5_acc": val_top5_acc,
                "test_top1_acc": test_top1_acc,
                "test_top5_acc": test_top5_acc,
            }
            results.append(epoch_results)

            self.logger.info(f"Subject {self.nSub} : Epoch {epoch}, train_conloss = {train_conloss}, geo_loss = {geo_loss}, "
                             f"v_miloss: {v_miloss}, e_miloss: {e_miloss}, v_recloss: {v_recloss}, e_recloss: {e_recloss}, "
                             f"val_acc1: {val_top1_acc}, val_acc5: {val_top5_acc}, test_acc1: {test_top1_acc}, test_acc5: {test_top5_acc}")
        
        # Final Test with best validation model
        if ckpt_state['EEG_Net'] is not None and ckpt_state['Visual_Net'] is not None:
            self.EEG_Net.load_state_dict(ckpt_state['EEG_Net'])
            self.Visual_Net.load_state_dict(ckpt_state['Visual_Net'])

            test_top1_acc, test_top5_acc = test(ckpt_state['epoch'], device, testloader, test_img_feature, self.modules, self.opt, phase='Final Test')
            print(f"[Final Test Results] Using best-val-acc weights from epoch {ckpt_state['epoch']}; "
                  f"Val top-1: {ckpt_state['val_top1_acc']}, Test top-1: {test_top1_acc}, Test top-5: {test_top5_acc}")
            self.logger.info(f"Subject {self.nSub} : Test results test_top_acc1: {test_top1_acc}, test_top_acc5: {test_top5_acc}")
            
            results.append({
                "best_epoch": ckpt_state['epoch'],
                "best_val_top1_acc": ckpt_state['val_top1_acc'],
                "top1_acc_results": test_top1_acc,
                "top5_acc_results": test_top5_acc,
            })
        else:
            raise ValueError("No best-val-acc state captured")
        
        self.logger.handlers.clear()
        self.wandblogger.finish()
        
        # Save results to a CSV file
        train_rows = results[:-1]
        final_row = results[-1]
        fieldnames = list(train_rows[0].keys()) + list(final_row.keys())
        
        results_file = os.path.join(self.results_path, f'sub_{format(self.nSub, "02")}_result.csv')
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(train_rows)
            writer.writerow(final_row)
        print(f'Results saved to {results_file}')
        return results


    def loops(self):
        if self.opt.exp_setting == 'intra-subject':
            return self.loops_intra_subject()
        elif self.opt.exp_setting == 'inter-subject':
            return self.loops_inter_subject()
        else:
            raise ValueError(f"Unknown exp_setting: {self.opt.exp_setting}. Expected 'intra-subject' or 'inter-subject'.")


def main():
    opt = parse_args()
    num_sub = opt.num_sub
    setup_seed(opt.seed)
    for i in range(1, num_sub + 1):
        ie = VE_SID(opt, i)
        ie.loops()


if __name__ == "__main__":
    main()
