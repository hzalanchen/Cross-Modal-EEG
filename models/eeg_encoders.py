
import math
import torch
import torch.nn as nn
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import warnings

class EEGNetv4_Encoder(nn.Module):
    # input (bs, channel, time_point)
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.eegnet = EEGNetv4(
            in_chans=self.shape[0],
            n_classes=1024,  # EEG Feature Dim
            input_window_samples=self.shape[1],
            final_conv_length='auto',
            pool_mode='mean',
            F1=8, # The filter number of temporal conv
            D=20, # Depth factor 
            F2=160, # The number of output channels of the separable convolution layer
            kernel_length=4,
            third_kernel_size=(4, 2),
            drop_prob=0.25
        )
    def forward(self, data):
        data = data.unsqueeze(0)
        data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        #print(data.shape)
        prediction = self.eegnet(data)
        #print(prediction.shape)
        return prediction


class ATCNet_Encoder(nn.Module):
    # input dim (bs, channel, time_point, 1)
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.eegATCNet = ATCNet(n_chans=self.shape[0], 
                                n_outputs=1024,
                                input_window_seconds=1.0,
                                sfreq=250.,
                                conv_block_n_filters=8,
                                conv_block_kernel_length_1=32,
                                conv_block_kernel_length_2=8,
                                conv_block_pool_size_1=4,
                                conv_block_pool_size_2=3,
                                conv_block_depth_mult=2,
                                conv_block_dropout=0.3,
                                n_windows=5,
                                att_head_dim=4,
                                att_num_heads=2,
                                att_dropout=0.5,
                                tcn_depth=2,
                                tcn_kernel_size=4,
                                tcn_n_filters=16,
                                tcn_dropout=0.3,
                                tcn_activation=nn.ELU(),
                                concat=False,
                                max_norm_const=0.25,
                                chs_info=None,
                                n_times=None,
                                n_channels=None,
                                n_classes=None,
                                input_size_s=None,
                                add_log_softmax=True)
        
    def forward(self, data):
        prediction = self.eegATCNet(data)
        return prediction


class EEGConformer_Encoder(nn.Module):
    # input dim (bs, channel, time_point)
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.eegConformer = EEGConformer(n_outputs=None, 
                                   n_chans=self.shape[0], 
                                   n_filters_time=40, 
                                   filter_time_length=10, 
                                   pool_time_length=25, 
                                   pool_time_stride=5, 
                                   drop_prob=0.25, 
                                   att_depth=2, 
                                   att_heads=1, 
                                   att_drop_prob=0.5, 
                                   final_fc_length=1760, 
                                   return_features=False, 
                                   n_times=None, 
                                   chs_info=None, 
                                   input_window_seconds=None,
                                   n_classes=1024, 
                                   input_window_samples=self.shape[1], 
                                   add_log_softmax=True)
    def forward(self, data):
        # data = data.unsqueeze(0)
        # data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        # print(data.shape)
        prediction = self.eegConformer(data)
        return prediction
 

class ShallowFBCSPNet_Encoder(nn.Module):
    # input dim (bs, channel, time_point)
    def __init__(self):
        super().__init__()
        self.shape = (63, 250)
        self.ShallowFBCSPNet = ShallowFBCSPNet(n_chans=self.shape[0],
                                         n_outputs=1024,
                                         n_times=self.shape[1], 
                                         n_filters_time=20, 
                                         filter_time_length=20,
                                         n_filters_spat=20,
                                         pool_time_length=25, 
                                         pool_time_stride=5, 
                                         final_conv_length='auto', 
                                         pool_mode='mean', 
                                         split_first_layer=True,
                                         batch_norm=True, 
                                         batch_norm_alpha=0.1, 
                                         drop_prob=0.5,
                                         chs_info=None, 
                                         input_window_seconds=1.0, 
                                         sfreq=250, 
                                         add_log_softmax=True)
    def forward(self, data):
        prediction = self.ShallowFBCSPNet(data)
        return prediction
    
# NICE
class PatchEmbedding_NICE(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class NICE_Encoder(nn.Sequential):
    # input dim (bs,1,channel, time_point)
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding_NICE(emb_size),
            FlattenHead()
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


# ATM-S
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x


class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        # print(src.shape)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]


class ATMS_Encoder(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS_Encoder, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = NICE_Encoder()
        # self.proj_eeg = Proj_eeg()        
      
         
    def forward(self, x):
        x = x.squeeze()
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[0](x)
        # attention - input(bs, channel, time_point). after nice_encoder - input(bs, 1, channel, time_point)
        x = x.unsqueeze(1)
        eeg_embedding = self.enc_eeg(x)
        # out = self.proj_eeg(eeg_embedding)
        return eeg_embedding  
      


# ATM-E
class PatchEmbedding_ATME(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.shape = (63, 250)
        self.tsconv = EEGNetv4(
            in_chans=self.shape[0],
            n_classes=1440,   
            input_window_samples=self.shape[1],
            final_conv_length='auto',
            pool_mode='mean',
            F1=8,
            D=20,
            F2=160,
            kernel_length=4,
            third_kernel_size=(4, 2),
            drop_prob=0.25
        )


    def forward(self, x):
        x = x.unsqueeze(3)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        
        return x


class Enc_eeg_ATME(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding_ATME(emb_size),
            FlattenHead()
        )


class ATME_Encoder(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATME_Encoder, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg_ATME()
        # self.proj_eeg = Proj_eeg()        
    
    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print(f'After enc_eeg shape: {eeg_embedding.shape}')
        # out = self.proj_eeg(eeg_embedding)
        return eeg_embedding  
    

# if __name__ == "__main__":
#     testeeg = torch.randn(4, 63, 250)
#     eegnet = ATMS_Encoder()
#     print(eegnet(testeeg).shape)
