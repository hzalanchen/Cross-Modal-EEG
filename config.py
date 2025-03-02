import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Cross_Modal_EEG')
    parser.add_argument('--EEGdatasetting', default='nice', type=str, help='EEG preprocess setting of nice or bravl')
    parser.add_argument('--num_sub', default=10, type=int, help='number of subjects used in the experiments.')
    parser.add_argument('--train_epochs', default=50, type=int, metavar='N',help='number of total train epochs to run')
    parser.add_argument('--print_freq', default=5, type=int, metavar='N', help='number of loop to print the corresponds information.')
    parser.add_argument('--print_freq_val', default=5, type=int, metavar='N',help='number of loop in validation to print the corresponds information.')
    parser.add_argument('--workers', default=0, type=int, metavar='N',help='number of data loading workers')
    parser.add_argument('-bs', '--batch-size', default=1024, type=int, metavar='N',help='mini-batch size (default: 1024)')
    
    # Network
    # Semantic Encoder
    parser.add_argument('--hidden_dim_S', default = 512, type = int, help='The dimension of the hidden layer of semantic encoder')
    parser.add_argument('--output_dim_S', default = 512, type = int, help='The dimension of the embedding of semantic output')
    parser.add_argument('--rec', default = 'add', type = str, help='Modal data concatenation mode')
    # Domain Encoder
    parser.add_argument('--hidden_dim_D', default = 512, type = int, help='The dimension of the hidden layer of domain encoder')
    parser.add_argument('--output_dim_D', default = 512, type = int, help='The dimension of the embedding of domain output')
    # Decoder
    parser.add_argument('--hidden_dim_dec', default = 512, type = int,help='The dimension of the hidden layer of decoder')

    # optimizer and scheduler
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--mi_lr', default=3e-4, type=float,help='learning rate of MI_NET', dest='mi_lr')
    
    #parser.add_argument('--step_size', default=10, type=int,metavar='N', help='Scheduler step size')
    #parser.add_argument('--decay_epochs', default = [50, 70, 90], nargs = '+', type = int, help = 'milestones for multisteps lr decay')
    #parser.add_argument('--gamma', default=0.1, type=float,metavar='W', help='The gammas value of scheduler')

    parser.add_argument('--seed', default=2024, type=int,help='seed for initializing training. ')
    parser.add_argument('--temperature', default=0.07, type=float,help='Initial softmax temperature (default: 0.07)')

    parser.add_argument("--main-loss", type = str, default = 'conloss', help = "Multimodal-Contrastive Learning Loss (conloss or supconloss)")
    parser.add_argument("--geo-loss", type = bool, default = False, help = "Whether to use geometric loss")
    parser.add_argument("--lambda1", type = float, default = 0., help = "Geometric Variance loss hyperparameter lambda 1 (default 0.5)")
    #parser.add_argument("--lambda2", type = float, default = 0, help = "Geometric Gaps loss hyperparameter lambda 2 (default 0.5)")
    parser.add_argument("--lambda3", type = float, default = 1., help = "MI loss hyperparameter lambda 3")
    parser.add_argument("--lambda4", type = float, default = 2., help = "Reconstruct loss lambda 4")
    
    opt = parser.parse_args()
    return opt 

