import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_utils import move_data_to_device, make_cqt_attn_mask, make_mel_attn_mask

from hpptnet import Transformer, CQTSpectrogram, ConvTrans
from conformer import Conformer, HAT, FCUDown, FCUUp
from einops import rearrange, repeat
from models import init_bn, init_gru, init_layer, ConvBlock
import config

cqt_config = config.cqt_config
mel_config = config.mel_config

spectrogram_extractor = Spectrogram(n_fft=config.mel_config['n_fft'], hop_length=config.hop_length, win_length=config.mel_config['n_fft'], window='hann', center=True, pad_mode='reflect', freeze_parameters=True).to('cuda')
logmel_extractor = LogmelFilterBank(**config.mel_config, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True).to('cuda')
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, pool_size=None, pool_type=None):
        return x

# conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> trans (-> F.avg_pool2d)
class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum, freq_attn_mask=None, num_heads=1, stride=(1, 1), norm='BatchNorm', kernel_size=(3, 3), residual=False):
        super(ConvTransBlock, self).__init__()
        self.residual = residual
        if residual:
            logging.info('='*20 + ' residual ' + '='*20)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1,1),
                               padding='same', bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels, momentum) if norm == 'InstanceNorm' else nn.BatchNorm2d(out_channels, momentum)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=((kernel_size[0]-stride[0])//2,(kernel_size[1]-stride[1])//2), bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels, momentum) if norm == 'InstanceNorm' else nn.BatchNorm2d(out_channels, momentum)

        self.trans = Transformer(out_channels, num_heads, freq_attn_mask)

        if self.residual:
            self.projection = Identity()
            if in_channels != out_channels or not (stride == 1 or stride == (1,1)):
                self.projection = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels, momentum)
                )

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        # conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu (-> F.avg_pool2d)
        # logging.info(f'block input: {input.shape}')
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        # logging.info(f'block x1   : {x.shape}')
        # [1, 48, 251, 352]
        # logging.info(f'block x2   : {x.shape}')
        if self.residual:
            input = self.projection(input)
            x = F.relu_(x + input)
        conv_x = x
        b, c, t, f = conv_x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, f, c) # b*t, f, c
        x = self.trans(x).reshape(b, t, f, c).permute(0, 3, 1, 2)
        
        if True: # NOTE: resi_transfomer
            # x = F.relu_(x + conv_x)
            pass
        # logging.info(f'block x3   : {x.shape}')
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        return x

# AcousticModel.png
class AcousticModelCnnTrans(nn.Module):
    #                   88             0.01
    def __init__(self, classes_num, momentum, model_type='ConvBlock', cqt_or_mel=True, bins=0, norm='BatchNorm', bin_ratios=[1,1,0.5,0.25], bin_deltas=[0,0,0,0], decoder='default', pool_type='avg'):
        super(AcousticModelCnnTrans, self).__init__()

        self.decoder = decoder
        self.model_type = model_type
        freq_attn_masks = [make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[0])]

        self.pool_types = ['', '', '', '']
        self.pool_sizes = [(1,1), (1,1), (1,1), (1,1)]
        self.pool_stride = True # True = pool | False = cnn_stride
        out_channels = [48, 64, 96, 128]
        kernel_sizes = [(3, 3), (3, 3), (3, 3), (3, 3)] # kernel_sizes = [(3, 9), (3, 9), (3, 5), (3, 3)]
        for n in range(len(bin_ratios) - 1):
            if bin_ratios[n + 1] != bin_ratios[n]:
                assert bin_ratios[n + 1] < bin_ratios[n], f'wrong bin_ratios: {bin_ratios}'
                self.pool_types[n] = pool_type
                self.pool_sizes[n] = (1, int(bin_ratios[n]/bin_ratios[n+1]))
        logging.info(f'pool_types: {self.pool_types}\t\tpool_sizes:{self.pool_sizes}')
        
        if model_type == 'Conv_trans':  # 前三层全1 第四层掩码 每层都是352
            #  TODO:  CNN*4 -> CNN*3+(CNN+Trans)*1参数缩减 Trans增多
            self.conv_block1 = ConvTransBlock(
                in_channels=1, out_channels=out_channels[0], momentum=momentum, freq_attn_mask=None, num_heads=num_heads)
            self.conv_block2 = ConvTransBlock(
                in_channels=out_channels[0], out_channels=out_channels[1], momentum=momentum, freq_attn_mask=None, num_heads=num_heads)
            self.conv_block3 = ConvTransBlock(
                in_channels=out_channels[1], out_channels=out_channels[2], momentum=momentum, freq_attn_mask=None, num_heads=num_heads)
            self.conv_block4 = ConvTransBlock(
                in_channels=out_channels[2], out_channels=out_channels[3], momentum=momentum, freq_attn_mask=freq_attn_masks[0], num_heads=num_heads)
        elif model_type in ['Three_blocks', 'Four_blocks', 'Conv_quarter_trans', 'ConvBlock_Transblock']:  # (CNN+Trans)*(3|4)
            if model_type == 'Three_blocks':
                bin_ratios = [1,0.5,0.25]
                self.strides = [(1,1),(1,1),(1,1)] # self.strides = [(1,1),(1,2),(1,2)]
                out_channels = [32, 64, 128] # [48, 96, 128] # 
                freq_attn_masks = [make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[0]), # 352
                    make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[1]),  # 176
                    make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[2])]  # 88
            
            elif model_type in ['Four_blocks', 'Conv_quarter_trans', 'ConvBlock_Transblock']:  # (CNN+Trans)*4  
                # TODO: mask: all_one -> sparse_delta3 -> sparse_delta1 -> sparse_delta0
                self.strides = [(1,1),(1,1),(1,1),(1,1)] # self.strides = [(1,1),(1,1),(1,2),(1,2)] # 
                out_channels = [48, 64, 96, 128] #[16, 32, 64, 128] # 
                kernel_sizes = [(3, 3), (3, 3), (3, 3), (3, 3)] # kernel_sizes = [(3, 9), (3, 9), (3, 5), (3, 3)]
                # kernel_sizes = [(3, 5), (3, 5), (3, 3), (3, 3)] 
                sparses = [False, True, True, True]
                # sparses = [False, False, False, False]
                logging.info('='*50)
                logging.info(f'out_channels: {out_channels}')
                logging.info(f'kernel_sizes: {kernel_sizes}')
                logging.info(f'sparses= {sparses}')
                logging.info('='*50)
                if cqt_or_mel:
                    freq_attn_masks = [make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[0], bin_delta=bin_deltas[0], sparse=sparses[0]), # 352
                        make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[1], bin_delta=bin_deltas[1], sparse=sparses[1]),  # 352
                        make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[2], bin_delta=bin_deltas[2], sparse=sparses[2]),  # 176
                        make_cqt_attn_mask(cqt_config, bin_ratio=bin_ratios[3], bin_delta=bin_deltas[3], sparse=sparses[3])]  # 88
                else:
                    freq_attn_masks = [make_mel_attn_mask(mel_config, bin_ratio=bin_ratios[0], sparse=sparses[0]), # 352
                        make_mel_attn_mask(mel_config, bin_ratio=bin_ratios[1], sparse=sparses[1]),  # 352
                        make_mel_attn_mask(mel_config, bin_ratio=bin_ratios[2], sparse=sparses[2]),  # 176
                        make_mel_attn_mask(mel_config, bin_ratio=bin_ratios[3], sparse=sparses[3])]  # 88
                
            for n in range(len(bin_ratios) - 1):
                if bin_ratios[n + 1] != bin_ratios[n]:
                    assert bin_ratios[n + 1] < bin_ratios[n], f'wrong bin_ratios: {bin_ratios}'
                    if self.pool_stride:
                        self.pool_types[n] = pool_type
                        self.pool_sizes[n] = (1, int(bin_ratios[n]/bin_ratios[n+1]))
                    elif n > 0:
                        self.strides[n] = (1, int(bin_ratios[n-1]/bin_ratios[n]))
            logging.info(f'pool_types: {self.pool_types}\tpool_sizes:{self.pool_sizes}\t\tstrides: {self.strides}')
            logging.info(f'self.pool_stride: {self.pool_stride} (True = pool | False = cnn_stride)')
            logging.info(f'num_heads: {num_heads}')
            if model_type in ['Conv_quarter_trans', 'ConvBlock_Transblock']: # 只保留最后一层的trans
                self.conv_block1 = ConvBlock(
                    in_channels=1, out_channels=out_channels[0], momentum=momentum)
                self.conv_block2 = ConvBlock(
                    in_channels=out_channels[0], out_channels=out_channels[1], momentum=momentum)
                self.conv_block3 = ConvBlock(
                    in_channels=out_channels[1], out_channels=out_channels[2], momentum=momentum)
            else:
                self.conv_block1 = ConvTransBlock(in_channels=1, out_channels=out_channels[0], momentum=momentum, freq_attn_mask=freq_attn_masks[0], num_heads=num_heads, stride=self.strides[0], kernel_size=kernel_sizes[0], norm=norm)
                self.conv_block2 = ConvTransBlock(in_channels=out_channels[0], out_channels=out_channels[1], momentum=momentum, freq_attn_mask=freq_attn_masks[1], num_heads=num_heads, stride=self.strides[1], kernel_size=kernel_sizes[1], norm=norm)
                self.conv_block3 = ConvTransBlock(in_channels=out_channels[1], out_channels=out_channels[2], momentum=momentum, freq_attn_mask=freq_attn_masks[2], num_heads=num_heads, stride=self.strides[2], kernel_size=kernel_sizes[2])
            if len(out_channels) == 3:
                self.conv_block4 = Identity()
            else:
                if model_type == 'ConvBlock_Transblock':
                    self.conv_block4 = ConvBlock(
                        in_channels=out_channels[2], out_channels=out_channels[3], momentum=momentum)
                    self.trans = nn.Sequential(Transformer(out_channels[3], num_heads, freq_attn_masks[3]),
                                               Transformer(out_channels[3], num_heads, freq_attn_masks[3]), 
                                               Transformer(out_channels[3], num_heads, freq_attn_masks[3]),
                                               Transformer(out_channels[3], num_heads, freq_attn_masks[3]))
                else:
                    self.conv_block4 = ConvTransBlock(in_channels=out_channels[2], out_channels=out_channels[3], 
                        momentum=momentum, freq_attn_mask=freq_attn_masks[3], num_heads=num_heads, stride=self.strides[3], kernel_size=kernel_sizes[3], norm=norm)
        else: # All ConvBlock
            self.conv_block1 = ConvBlock(
                in_channels=1, out_channels=out_channels[0], momentum=momentum)
            self.conv_block2 = ConvBlock(
                in_channels=out_channels[0], out_channels=out_channels[1], momentum=momentum)
            self.conv_block3 = ConvBlock(
                in_channels=out_channels[1], out_channels=out_channels[2], momentum=momentum)
            self.conv_block4 = ConvBlock(
                in_channels=out_channels[2], out_channels=out_channels[3], momentum=momentum)

            # (TCN+Trans)*3
            if 'trans' in model_type or 'Trans' in model_type:
                self.conv_trans = self._make_trans(
                    config.dilations, num_heads, freq_attn_masks[0], 128, config.convtrans_channel, config.output_channel, norm=norm)
            else:
                self.conv_trans = Identity()

        if decoder == 'hpp':
            """ 
            self.lstm = BiLSTM(128, 128//2)
            self.linear = nn.Linear(128, 1) 
            """
            self.fc5 = Identity()
            
            self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2,
                bias=True, batch_first=True, dropout=0., bidirectional=True)
            self.fc = nn.Linear(512, 1, bias=True)
        else:
            if bins == 0:
                bins = cqt_config['n_bins'] if cqt_or_mel else mel_config['n_bins']
            midfeat = int(0.5 + bins * bin_ratios[-1]) * out_channels[-1]
            self.fc5 = nn.Linear(midfeat, 768, bias=False)
            self.bn5 = nn.InstanceNorm1d(768) if norm == 'InstanceNorm' else nn.BatchNorm1d(768, momentum=momentum) 
            self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                bias=True, batch_first=True, dropout=0., bidirectional=True)
            self.fc = nn.Linear(512, classes_num, bias=True)

            self.init_weight()

        logging.info('='*50)
        logging.info(model_type)
        logging.info(f'out_channels: {out_channels}')
        logging.info(f'kernel_sizes: {kernel_sizes}')
        logging.info('='*50)

    def _make_trans(self, dilations, num_heads, freq_attn_mask, conv_channel, convtrans_channel, out_channel, norm='BatchNorm'):
        conv_trans = []
        for idx, dilation in enumerate(dilations):
            if idx == 0:
                conv_trans.append(ConvTrans(conv_channel, dilation, num_heads, freq_attn_mask, out_channel=convtrans_channel))
            if idx == len(dilations) - 1:
                conv_trans.append(ConvTrans(convtrans_channel, dilation, num_heads, stride=4))
            else:
                conv_trans.append(ConvTrans(convtrans_channel, dilation, num_heads, freq_attn_mask))
        # downsampling
        down = [
            nn.Conv2d(convtrans_channel, out_channel, 3, (4, 1), 1, bias=False),
            nn.InstanceNorm2d(out_channel) if norm == 'InstanceNorm' else nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]
        conv_trans.extend(down)
        conv_trans = nn.Sequential(*conv_trans)

        return conv_trans
    

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, mel_bins)
          #input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """
        
        # ConvBlock * 4
        # [6, 1, 1001, 352]
        x = self.conv_block1(input, pool_size=self.pool_sizes[0], pool_type=self.pool_types[0])
        x = F.dropout(x, p=0.2, training=self.training)
        # [6, 48, 1001, 352]
        x = self.conv_block2(x, pool_size=self.pool_sizes[1], pool_type=self.pool_types[1])
        x = F.dropout(x, p=0.2, training=self.training)
        # [6, 64, 1001, 352]
        x = self.conv_block3(x, pool_size=self.pool_sizes[2], pool_type=self.pool_types[2])
        x = F.dropout(x, p=0.2, training=self.training)
        # [6, 96, 1001, 352]
        x = self.conv_block4(x, pool_size=self.pool_sizes[3], pool_type=self.pool_types[3])
        x = F.dropout(x, p=0.2, training=self.training)
        # [6, 128, 1001, 352] 
        
        if self.model_type == 'ConvBlock_Transblock':
            b, c, t, f = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, f, c)
            x = self.trans(x).reshape(b, t, f, c).permute(0, 3, 1, 2)
        
        # flatten -> Trans -> FC -> GRU -> FC
        if self.decoder == 'hpp':
            #[2, 128, 501, 88]
            b, c, t, f = x.size() 
            # => [b x f x T x c] 
            x = torch.permute(x, [0, 3, 2, 1])

            # => [(b*f) x T x c]
            x = x.reshape([b*f, t, c])
            self.gru.flatten_parameters()
            (x, _) = self.gru(x)
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            # [176, 501, 512]
            x = torch.sigmoid(self.fc(x))
            # [176, 501, 1] => [2, 501, 88]
            output = x.reshape([b, f, t]).permute([0, 2, 1])
        else:
            # (b, 128, t, 88) -> (b, t, 128 * 88)
            x = x.transpose(1, 2).flatten(2) # [2, 501, 11264]
            # -> (b, t, 768)    
            x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            # [6, 1001, 768]
            # -> (b, t, 512) -> (b, t, 88)
            self.gru.flatten_parameters()
            (x, _) = self.gru(x)
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            output = torch.sigmoid(self.fc(x))
        # [6, 1001, 88]
        return output

# Regress.png
class Regress_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num, model_type='', norm='BatchNorm'):
        super(Regress_CRNN, self).__init__()

        assert norm in ['BatchNorm', 'InstanceNorm'], f'unsupported norm method: {norm}'
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        cqt_bins = 352
        fmin = 30
        fmax = sample_rate // 2
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # NOTICE: hppt_trans -> 3*3conv 压缩后给fc5
        momentum = 0.01

        self.cqt_or_mel = True

        if self.cqt_or_mel:
            self.cqt = CQTSpectrogram(**cqt_config, log_scale=config.log_scale)
            bins = cqt_bins 
        else:
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
            bins = mel_bins
        bin_ratios = config.bin_ratios
        bin_deltas = config.bin_deltas
        logging.info(f'cqt_or_mel: {self.cqt_or_mel} (True: cqt | False: log-mel)')
        logging.info(f'bin_ratios:{bin_ratios}\tbin_deltas:{bin_deltas}')
        print(f'bin_ratios:{bin_ratios}\tbin_deltas:{bin_deltas}')
        
        self.bn0 = nn.InstanceNorm2d(bins, momentum) if norm == 'InstanceNorm' else nn.BatchNorm2d(bins, momentum)
        self.frame_model = AcousticModelCnnTrans(classes_num, momentum, model_type, cqt_or_mel=self.cqt_or_mel, norm=norm, bins=bins, bin_ratios=bin_ratios, bin_deltas=bin_deltas)
        self.reg_onset_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, cqt_or_mel=self.cqt_or_mel, norm=norm, bins=bins, bin_ratios=bin_ratios, bin_deltas=bin_deltas)
        self.reg_offset_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, cqt_or_mel=self.cqt_or_mel, norm=norm, bins=bins, bin_ratios=bin_ratios, bin_deltas=bin_deltas)
        self.velocity_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, cqt_or_mel=self.cqt_or_mel, norm=norm, bins=bins, bin_ratios=bin_ratios, bin_deltas=bin_deltas)
        
        
        self.reg_onset_gru = nn.GRU(input_size=classes_num * 2, hidden_size=256, num_layers=1,
                                    bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=classes_num * 3, hidden_size=256, num_layers=1,
                                bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input, extract_onset=False):
        """
        Args:
          input: (batch_size, data_length) # [*, 160000]

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        if self.cqt_or_mel:
            # NOTE:  hppt为  x = self.cqt(audio)[:, None]
            n_fft = cqt_config['sr'] / cqt_config['fmin'] / (2 ** (1 / cqt_config['bins_per_octave']) - 1)
            n_fft = 2 ** int(math.ceil(math.log2(n_fft)))
            input = torch.tensor(np.pad(input.cpu().numpy(), ((0, 0), (n_fft//2, n_fft//2)))).to('cuda')
            # [6, 352, 296] -> [6, 1, 296, 352]
            x = repeat(self.cqt(input), 'b f t -> b c t f', c=1)
        else:
            # (batch_size, 1, time_steps, freq_bins)
            x = self.spectrogram_extractor(input)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3) 
        x = self.bn0(x)  # BatchNorm2d 
        x = x.transpose(1, 3)

        # (batch_size, 1, time_steps, mel_bins) =>
        # (batch_size, time_steps, classes_num)
        frame_output = self.frame_model(x)
        # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)
        # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)
        # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)

        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5)
                      * velocity_output.detach()), dim=2)
        self.reg_onset_gru.flatten_parameters()
        (x, _) = self.reg_onset_gru(x)
        # extract features of onset
        if extract_onset:
            logging.info(f'ouput reg_onset_gru : {x.shape}')
            return {'onset_features': x}
        
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(),
                      reg_offset_output.detach()), dim=2)
        self.frame_gru.flatten_parameters()
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        # (batch_size, time_steps, classes_num)
        frame_output = torch.sigmoid(self.frame_fc(x))
        output_dict = {
            'reg_onset_output': reg_onset_output,
            'reg_offset_output': reg_offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}

        return output_dict
    
class Regress_Conformer(nn.Module):
    def __init__(self, frames_per_second, classes_num, model_type='Conformer', norm='BatchNorm'):
        super(Regress_Conformer, self).__init__()
        assert norm in ['BatchNorm', 'InstanceNorm'], f'unsupported norm method: {norm}'
        self.model_type = model_type

        # NOTE: hppt_trans -> 3*3conv 压缩后给fc5
        momentum = 0.01

        conformer_bias = True
        self.cqt_or_mel = False
        self.sony_mel = False # self.cqt_or_mel = False 时才有效
        self.x_and_xt = False # cat(x, x_t)
        self.x_or_xt = False # True: conv output x | False: trans output x_t
        self.incre_xt = False # True: x_t 特征维度和 x 一样递增 | False: x_t特征维度保持64
        self.x_add_xt = False # True: x = (x + x_t) / 2
        self.freq_mask = False
        self.has_token = False # 若True则mask上补token 目前False更好
        self.easy_decoder = True # True: no concat | False: 额外gru velocity = (onset ** 0.5) * velocity
        self.hard_decoder = False # True: frame = (onset ** 0.5) * frame, (frame ** 0.5) * offset
        self.offset_or_frame = False # True: offset = (frame ** 0.5) * offset | False: frame = (((onset + offset) / 2) ** 0.5) * frame
        self.concate = False
        self.is_velocity = False
        self.invalid_rnn = False
        self.invalid_guidance = False

        logging.info(f'cqt_or_mel: {self.cqt_or_mel} (True: cqt | False: log-mel)')
        if not self.cqt_or_mel:
            logging.info(f'==> sony_mel: {self.sony_mel} ')
        logging.info(f'x_and_xt: {self.x_and_xt} (True: concat(x, x_t) | False: x_or_xt available)')
        logging.info(f'x_or_xt: {self.x_or_xt} (True: conv output x | False: trans output x_t)')
        logging.info(f'incre_xt: {self.incre_xt}')
        logging.info(f'x_add_xt: {self.x_add_xt}')
        logging.info(f'has_token: {self.has_token}')
        logging.info(f'easy_decoder: {self.easy_decoder}')
        if not self.easy_decoder:
            logging.info(f'==> velocity = (onset ** 0.5) * velocity')
            logging.info(f'hard_decoder: {self.hard_decoder} (True: frame = (onset ** 0.5) * frame, (frame ** 0.5) * offset)')
            if not self.hard_decoder:
                logging.info(f'====> offset_or_frame: {self.offset_or_frame}' + \
                             f'(True: offset = (frame ** 0.5) * offset | False: frame = (((onset + offset) / 2) ** 0.5) * frame)')
            if self.concate:
                logging.info(f'concate : {self.concate} (* => concate)')
            if self.invalid_guidance:
                logging.info(f'invalid_guidance : {self.invalid_guidance} ( => all "*" invalid)')
                
        logging.info(f'is_velocity: {self.is_velocity}')
        logging.info(f'invalid_rnn: {self.invalid_rnn}')
        logging.info(f'freq_mask: {self.freq_mask}')
        
        # self.conformer = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=depth, num_heads=6, mlp_ratio=4, qkv_bias=True)
        self.conv_base_channel = 16
        self.channel_ratio = 4
        self.embed_dim = 64
        self.gru_size = 64
        
        attn_mask = []
        depth = 12
        num_heads = 1
        num_layers = 2
        logging.info(f'depth: {depth}\tnum_heads: {num_heads}\tnum_layers: {num_layers}')
        for i in range(0,12): #0~7
            attn_mask.append(None)
        if self.freq_mask:
            attn_mask[0] = make_cqt_attn_mask(352)
            attn_mask[1] = make_cqt_attn_mask(352)
            attn_mask[2] = make_cqt_attn_mask(352)
            attn_mask[3] = make_cqt_attn_mask(352)
            attn_mask[4] = make_cqt_attn_mask(352)
            attn_mask[5] = make_cqt_attn_mask(352)
            attn_mask[6] = make_cqt_attn_mask(352)
            attn_mask[7] = make_cqt_attn_mask(352)
            # attn_mask[4] = make_cqt_attn_mask(176)
            # attn_mask[5] = make_cqt_attn_mask(176)
            # attn_mask[6] = make_cqt_attn_mask(176)
            # attn_mask[7] = make_cqt_attn_mask(176)
            # attn_mask[8] = make_cqt_attn_mask(88)
            # attn_mask[9] = make_cqt_attn_mask(88)
            # attn_mask[10] = make_cqt_attn_mask(88)
            # attn_mask[11] = make_cqt_attn_mask(88)
            # self.freq_mask = make_mel_attn_mask(mel_config)
        logging.info(f'==> freq_mask = {["None" if mask is None else str(mask.shape) for mask in attn_mask]}')
        
        if self.cqt_or_mel:
            self.cqt = CQTSpectrogram(**config.cqt_config, log_scale=config.log_scale)
            bins = config.cqt_config['n_bins'] 
            logging.info(config.cqt_config)
        else:
            self.spectrogram_extractor = Spectrogram(n_fft=config.mel_config['n_fft'], hop_length=config.hop_length, win_length=config.mel_config['n_fft'], window='hann', center=True, pad_mode='reflect', freeze_parameters=True).to('cuda')
            self.logmel_extractor = LogmelFilterBank(**config.mel_config, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True).to('cuda')
            bins = config.mel_config['n_mels'] 
            logging.info(config.mel_config)
            if self.sony_mel:
                # NOTE: pad_mode = reflect -> constant 
                self.spectrogram_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, n_fft=config.mel_config['n_fft'], win_length=config.mel_config['n_fft'], hop_length=config.hop_length, pad_mode='constant', n_mels=config.mel_config['n_mels'] , norm='slaney')
        
        if "four_branch" in self.model_type:
            self.onset_conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias)
            self.offset_conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias)
            self.frame_conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias)
            self.velocity_conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias)
        elif "dual_branch" in self.model_type:
            self.onset_conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias)
            self.conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias)
        else:
            self.conformer = Conformer(num_heads=num_heads, base_channel=self.conv_base_channel, # patch_size=16, 
                channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias, incre_xt=self.incre_xt, cqt_or_mel=False)
        
        if self.x_and_xt:
            gru_input = self.embed_dim + self.conv_base_channel * self.channel_ratio * 2 * 2
        elif self.x_or_xt or self.incre_xt:
            gru_input = self.conv_base_channel * self.channel_ratio * 2 * 2
        else:
            gru_input = self.embed_dim
            
        if self.x_add_xt:
            if self.x_or_xt: # (conv output) x + (pw_conv trans output) x_t_p
                self.add_conv = FCUUp(self.embed_dim, gru_input, has_token=self.has_token)
            else:
                self.add_conv = FCUDown(self.conv_base_channel * self.channel_ratio * 2 * 2, gru_input, has_token=self.has_token)
            
        self.reg_onset_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers)
        self.reg_offset_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=(self.easy_decoder or (not self.hard_decoder and not self.offset_or_frame)), 
                                             num_layers=num_layers)
        self.frame_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=(self.easy_decoder or (not self.hard_decoder and self.offset_or_frame)), 
                                        num_layers=num_layers)
        self.velocity_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=self.easy_decoder,  num_layers=num_layers, 
                                           is_velocity=(False if self.hard_decoder else self.is_velocity))

        if not self.easy_decoder:
            self.velocity_decoder = ConformerGru(input_size=192, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers, 
                                           is_velocity=(False if self.hard_decoder else self.is_velocity))
            # self.note_decoder = ConformerGru(input_size=192, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers)
            self.note_decoder = ConformerGru(input_size=320, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers)
            if self.hard_decoder:
                self.offset_decoder = ConformerGru(input_size=192, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers)
        # self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length) # [*, 160000]

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        if self.cqt_or_mel:
            # # NOTICE:  hppt为  x = self.cqt(audio)[:, None]
            # n_fft = cqt_config['sr'] / cqt_config['fmin'] / (2 ** (1 / cqt_config['bins_per_octave']) - 1)
            # n_fft = 2 ** int(math.ceil(math.log2(n_fft)))
            # input = torch.tensor(np.pad(input.cpu().numpy(), ((0, 0), (n_fft//2, n_fft//2)))).to('cuda')
            # # [6, 352, 296] -> [6, 1, 296, 352]
            x = repeat(self.cqt(input), 'b f t -> b c t f', c=1)
        else:
            # (batch_size, 1, time_steps, freq_bins)
            x = self.spectrogram_extractor(input)  
            if not self.sony_mel:
                x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
            else:   
                x = (torch.log(x + 1e-8)).unsqueeze(1).transpose(2,3) # [2, 229, 101] -> [2, 1, 101, 229]
            # x = input

        # x = x.transpose(1, 3) 
        # x = self.bn0(x)  # BatchNorm2d 
        # x = x.transpose(1, 3)

        if "four_branch" in self.model_type:
            [onset, onset_t] = self.onset_conformer(x)
            [offset, offset_t] = self.offset_conformer(x)
            [frame, frame_t] = self.frame_conformer(x)
            [velocity, velocity_t] = self.velocity_conformer(x)
            
            if self.x_and_xt:
                onset = torch.cat((onset, onset_t), dim=-1)
                offset = torch.cat((offset, offset_t), dim=-1)
                frame = torch.cat((frame, frame_t), dim=-1)
                velocity = torch.cat((velocity, velocity_t), dim=-1)
            elif not self.x_or_xt:
                onset = onset_t
                frame = frame_t
                offset = offset_t
                velocity = velocity_t
            
            frame_output = self.frame_model(frame)
            onset_output = self.reg_onset_model(onset)
            offset_output = self.reg_offset_model(offset)
            velocity_output = self.velocity_model(velocity)
        elif "dual_branch" in self.model_type:
            [onset, onset_t] = self.onset_conformer(x)
            [x, x_t] = self.conformer(x)
            
            if self.x_and_xt:
                onset = torch.cat((onset, onset_t), dim=-1)
                x = torch.cat((x, x_t), dim=-1)
            elif not self.x_or_xt:
                onset = onset_t
                x = x_t
            
            onset_output = self.reg_onset_model(onset)
            frame_output = self.frame_model(x)
            offset_output = self.reg_offset_model(x)
            velocity_output = self.velocity_model(x)
        else:
            [x, x_t] = self.conformer(x) # [1, 88, 401, 256], [1, 88, 401, 64]
            if self.x_add_xt:
                N, _, T, _ = x.shape
                if self.x_or_xt:
                    x_t = self.add_conv(x_t.transpose(1,2).flatten(start_dim=0, end_dim=1), N, T).transpose(1,3)
                else:
                    x = self.add_conv(x.transpose(1,3)) # [401, 88, 64] 
                    x = x.reshape(N, T, x.shape[1], x.shape[2]).transpose(1,2).contiguous() 
                x = (x + x_t)/2
            elif self.x_and_xt:
                x = torch.cat((x, x_t), dim=-1)
            elif not self.x_or_xt:
                x = x_t
            frame_output = self.frame_model(x)
            onset_output, onset_x = self.reg_onset_model(x, True)
            offset_output, offset_x = self.reg_offset_model(x, True)
            velocity_output = self.velocity_model(x)

        if not self.easy_decoder:
            # velocity [1, 88, 401, 64] | onset [1, 401, 88]
            if self.invalid_guidance:
                velocity_output = self.velocity_decoder(velocity_output)
                offset_output = self.offset_decoder(offset_output)
                frame_output = self.note_decoder(frame_output)
            else:
                if self.concate:

                    velocity_output = torch.cat((velocity_output, onset_x.detach()), dim=-1)
                    velocity_output = self.velocity_decoder(velocity_output)
                else:
                    velocity_output = (onset_output.detach().transpose(0, 2) ** 0.5) * velocity_output.transpose(0, 3)
                    velocity_output = self.velocity_decoder(velocity_output.transpose(0, 3))

                if self.hard_decoder: # frame *= onset | offset *= frame
                    if self.concate:
                        frame_output = torch.cat((frame_output, onset_x.detach()), dim=-1)
                        frame_output, frame_x = self.note_decoder(frame_output, True)
                        offset_output = torch.cat((offset_output, frame_x.detach()), dim=-1)
                        offset_output = self.offset_decoder(offset_output)
                    else:
                        frame_output = (onset_output.detach().transpose(0, 2) ** 0.5) * frame_output.transpose(0, 3)
                        frame_output = self.note_decoder(frame_output.transpose(0, 3))
                        offset_output = (frame_output.detach().transpose(0, 2) ** 0.5) * offset_output.transpose(0, 3)
                        offset_output = self.offset_decoder(offset_output.transpose(0, 3))
                elif self.offset_or_frame: # offset *= frame
                    offset_output = (frame_output.detach().transpose(0, 2) ** 0.5) * offset_output.transpose(0, 3)
                    offset_output = self.note_decoder(offset_output.transpose(0, 3))
                else: # frame *= (onset + offset)/2
                    if self.concate:
                        frame_output = torch.cat((frame_output, onset_x.detach(), offset_x.detach()), dim=-1)
                        frame_output = self.note_decoder(frame_output)
                    else:
                        frame_output = (((onset_output.detach().transpose(0, 2) + offset_output.detach().transpose(0, 2)) / 2) ** 0.5) * frame_output.transpose(0, 3)
                        frame_output = self.note_decoder(frame_output.transpose(0, 3))
            
        output_dict = {
            'reg_onset_output': onset_output,
            'reg_offset_output': offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}

        return output_dict

class Regress_HAT(nn.Module):
    def __init__(self, frames_per_second, classes_num, model_type='Conformer', norm='BatchNorm', pre_spec = False):
        super(Regress_HAT, self).__init__()
        assert norm in ['BatchNorm', 'InstanceNorm'], f'unsupported norm method: {norm}'
        self.model_type = model_type

        momentum = 0.01
        
        conformer_bias = True
        self.pre_spec = pre_spec
        self.cqt_or_mel = True
        self.incre_xt = False # True: x_t 特征维度和 x 一样递增 | False: x_t特征维度保持64
        self.freq_mask = False
        self.pos_freq = False # 频率位置编码
        self.has_token = False # 若True则mask上补token 目前False更好
        self.easy_decoder = True # 参考Sony方案直接解码, 单层gru输出后无concat
        self.hard_decoder = False # easy_decoder=False 才有效, 给velocity一个单独的gru Concat(velocity, GRU(detach(velocity), onset))
        self.is_velocity = False
        self.invalid_rnn = False
        self.offset_or_frame = False
        
        print(f'pre_spec: {self.pre_spec}')
        logging.info(f'cqt_or_mel: {self.cqt_or_mel} (True: cqt | False: log-mel)')
        logging.info(f'pre_spec: {self.pre_spec}')
        logging.info(f'incre_xt: {self.incre_xt}')
        logging.info(f'has_token: {self.has_token}')
        logging.info(f'easy_decoder: {self.easy_decoder}')
        if not self.easy_decoder:
            logging.info(f'==> hard_decoder: {self.hard_decoder}')
        logging.info(f'is_velocity: {self.is_velocity}')
        logging.info(f'invalid_rnn: {self.invalid_rnn}')
        logging.info(f'freq_mask: {self.freq_mask}')
        logging.info(f'pos_freq: {self.pos_freq}')
        
        # self.conformer = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
        self.conv_base_channel = 16
        self.channel_ratio = 4
        self.embed_dim = 64
        self.gru_size = 64
        
        attn_mask = []
        depth = 8
        num_heads = 2
        num_layers = 2
        logging.info(f'depth: {depth}\tnum_heads: {num_heads}\tnum_layers: {num_layers}')
        
        for i in range(0,depth): #0~7
            attn_mask.append(None)
        if self.freq_mask:
                attn_mask[0] = make_cqt_attn_mask(352)
                attn_mask[1] = make_cqt_attn_mask(352)
                attn_mask[2] = make_cqt_attn_mask(352)
                attn_mask[3] = make_cqt_attn_mask(352)
                attn_mask[4] = make_cqt_attn_mask(176)
                attn_mask[5] = make_cqt_attn_mask(176)
                attn_mask[6] = make_cqt_attn_mask(88)
                attn_mask[7] = make_cqt_attn_mask(88)
                # self.freq_mask = make_mel_attn_mask(mel_config)
        logging.info(f'==> freq_mask = {["None" if mask is None else str(mask.shape) for mask in attn_mask]}')
        
        if self.cqt_or_mel:
            self.cqt = CQTSpectrogram(**config.cqt_config, log_scale=config.log_scale)
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
            bins = config.cqt_config['n_bins'] 
            logging.info(config.cqt_config)
        else:
            bins = config.mel_config['n_mels'] 
            logging.info(config.mel_config)
            if not self.pre_spec:
                # NOTE: pad_mode = reflect -> constant 
                self.spectrogram_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, n_fft=config.mel_config['n_fft'], win_length=config.mel_config['n_fft'], hop_length=config.hop_length, pad_mode='constant', n_mels=config.mel_config['n_mels'] , norm='slaney')

        self.hat = HAT(num_heads=num_heads, base_channel=self.conv_base_channel,  pos_freq=self.pos_freq,  # patch_size=16, 
            channel_ratio=self.channel_ratio, embed_dim=self.embed_dim, depth=depth, mlp_ratio=4, qkv_bias=True, attn_mask=attn_mask, has_token=self.has_token, n_bin=bins, bias=conformer_bias, incre_xt=self.incre_xt)
        
        gru_input = self.embed_dim
        self.reg_onset_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers)
        self.reg_offset_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=(self.easy_decoder or not self.offset_or_frame), 
                                             num_layers=num_layers)
        self.frame_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=(self.easy_decoder or self.offset_or_frame), 
                                        num_layers=num_layers)
        self.velocity_model = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=self.easy_decoder,  num_layers=num_layers, 
                                           is_velocity=(False if self.hard_decoder else self.is_velocity))

        if not self.easy_decoder:
            self.velocity_decoder = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers, 
                                           is_velocity=(False if self.hard_decoder else self.is_velocity))

            self.note_decoder = ConformerGru(input_size=gru_input, gru_size=self.gru_size, last_decoder=True,  num_layers=num_layers)
            
            if self.hard_decoder:
                self.velocity_gru = nn.GRU(input_size=classes_num * 2, hidden_size=256, num_layers=1,
                                            bias=True, batch_first=True, dropout=0., bidirectional=True)
                self.velocity_fc = nn.Linear(512, 88, bias=True)
                if self.is_velocity:
                    self.velocity_fc2 = nn.Linear(1, 128, bias=True)
            elif self.is_velocity:
                self.velocity_fc1 = nn.Linear(128, 1, bias=True)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length) # [*, 160000]

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        if self.pre_spec:
            # [10, 1, 352, 111]
            x = self.amplitude_to_db(input).transpose(2,3)
        elif self.cqt_or_mel:
            x = repeat(self.cqt(input), 'b f t -> b c t f', c=1)
        # if self.cqt_or_mel:
        #     # NOTICE:  hppt为  x = self.cqt(audio)[:, None]
        #     n_fft = cqt_config['sr'] / cqt_config['fmin'] / (2 ** (1 / cqt_config['bins_per_octave']) - 1)
        #     n_fft = 2 ** int(math.ceil(math.log2(n_fft)))
        #     input = torch.tensor(np.pad(input.cpu().numpy(), ((0, 0), (n_fft//2, n_fft//2)))).to('cuda')
        #     # [6, 352, 296] -> [6, 1, 296, 352]
        #     x = repeat(self.cqt(input), 'b f t -> b c t f', c=1)
        else:
            # (batch_size, 1, time_steps, freq_bins)
            x = spectrogram_extractor(input)  
            x = logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
            # x = (torch.log(x + 1e-8)).unsqueeze(1).transpose(2,3) # [2, 229, 101] -> [2, 1, 101, 229]

        # x = x.transpose(1, 3) 
        # x = self.bn0(x)  # BatchNorm2d 
        # x = x.transpose(1, 3)

        x = self.hat(x) # [4, 88, 128, 64], [4, 88, 128, 256]

        frame_output = self.frame_model(x)
        onset_output = self.reg_onset_model(x)
        offset_output = self.reg_offset_model(x)
        velocity_output = self.velocity_model(x)

        if not self.easy_decoder:
            if velocity_output.shape[-1] == config.velocity_scale:
                # vo = velocity_output.detach().argmax(-1) / config.velocity_scale
                vo = self.velocity_fc1(velocity_output.detach())
                vo = torch.sigmoid(vo).squeeze()

            velocity_output = (onset_output.detach() ** 0.5) * velocity_output
            velocity_output = self.velocity_decoder(velocity_output)

            if self.offset_or_frame:
                offset_output = (frame_output.detach() ** 0.5) * offset_output
                offset_output = self.note_decoder(offset_output)
            else:
                # 放大onset 抑制offset
                frame_output = (((onset_output.detach() ** 0.5 + offset_output.detach() ** 2) / 2)) * frame_output
                frame_output = self.note_decoder(frame_output)
            
            if self.hard_decoder:
                x = torch.cat((velocity_output, onset_output.detach()), dim=2)
                self.velocity_gru.flatten_parameters()
                (x, _) = self.velocity_gru(x)
                x = self.velocity_fc(x)
                # print(x.shape) # [2, 128, 1]
                if not self.is_velocity:
                    velocity_output = torch.sigmoid(x)

        output_dict = {
            'reg_onset_output': onset_output,
            'reg_offset_output': offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}

        return output_dict

class ConformerGru(nn.Module):
    def __init__(self, input_size=64, gru_size=64, norm='BatchNorm', is_velocity=False, invalid_rnn=False, last_decoder=True, num_layers=2):
        super(ConformerGru, self).__init__()

        self.is_velocity = is_velocity
        self.last_decoder = last_decoder
        self.invalid_rnn = invalid_rnn
        if not invalid_rnn:
            self.gru = nn.GRU(input_size=input_size, hidden_size=gru_size, num_layers=num_layers, bias=True, batch_first=True, dropout=0., bidirectional=True)
            fc_size = gru_size * 2
        if self.is_velocity:
            # NOTE: sony 无Sigmoid, loss的label为0~128, 此处会将label/128归一化
            #   loss_velocity = nn.CrossEntropyLoss(output_velocity, label_velocity)

            self.fc0 = nn.Linear(fc_size,config.velocity_scale, bias=True) 
            # self.fc1 = nn.Sequential(nn.Linear(frame_size, note_class, bias=False))
        elif self.last_decoder:
            self.fc0 = nn.Linear(fc_size,1, bias=True) 
            # (ConformerGru) TODO: 
            # self.fc1 = nn.Sequential(nn.Linear(frame_size, note_class, bias=False), nn.Sigmoid()) 
            self.fc1 = nn.Sequential(nn.Sigmoid()) 
        else:
            self.fc0 = nn.Linear(fc_size,input_size, bias=True) 
    
    def forward(self, x, return_x=False):
        b, f, t, c = x.shape # [1, 88, 401, 64]
        if not self.invalid_rnn:
            self.gru.flatten_parameters()
            (x, _) = self.gru(x.flatten(start_dim=0, end_dim=1)) # [88, 401, 64]
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        if self.is_velocity:
            output = self.fc0(x).reshape(b, f, t, -1).transpose(1, 2).contiguous() # [1, 401, 88, 128]
        elif self.last_decoder:
            # [88, 401, 128]
            output = self.fc0(x).reshape(b, f, t).transpose(1, 2).contiguous() # [88, 401, 1]
            output = self.fc1(output)  # [88, 401, 1]
        else:
            output = self.fc0(x).reshape(b, f, t, -1)
        if return_x:
            return output, x.reshape(b, f, t, -1)
        else:
            return output

class Regress_CRNN_Slim(nn.Module):
    def __init__(self, frames_per_second, classes_num, model_type='', norm='BatchNorm', decoder='hpp'):
        super(Regress_CRNN_Slim, self).__init__()

        assert norm in ['BatchNorm', 'InstanceNorm'], f'unsupported norm method: {norm}'
        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        cqt_bins = 352
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # NOTE: hppt_trans -> 3*3conv 压缩后给fc5
        momentum = 0.01
        self.cqt = CQTSpectrogram(**cqt_config)

        bins = cqt_bins 
        self.bn0 = nn.InstanceNorm2d(bins) if norm == 'InstanceNorm' else nn.BatchNorm2d(bins, momentum)
        bin_ratios = config.bin_ratios
        
        logging.info('='*50)
        logging.info(f'decoder : {decoder}\tbin_ratios:{bin_ratios}')
        
        # NOTE: no trans in offset and velocity model
        self.frame_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, norm=norm, decoder=decoder, bin_ratios=bin_ratios)
        self.reg_onset_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, norm=norm, decoder=decoder, bin_ratios=bin_ratios)
        self.reg_offset_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, norm=norm, decoder=decoder, bin_ratios=bin_ratios)
        self.velocity_model = AcousticModelCnnTrans(
            classes_num, momentum, model_type, norm=norm, decoder=decoder, bin_ratios=bin_ratios)

        self.reg_onset_gru = nn.GRU(input_size=classes_num * 2, hidden_size=256, num_layers=1,
                                    bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=classes_num * 3, hidden_size=256, num_layers=1,
                                bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input, extract_onset=False):
        """
        Args:
          input: (batch_size, data_length) # [*, 160000]

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        # NOTICE:  hppt为  x = self.cqt(audio)[:, None]
        n_fft = cqt_config['sr'] / cqt_config['fmin'] / \
            (2 ** (1 / cqt_config['bins_per_octave']) - 1)
        n_fft = 2 ** int(math.ceil(math.log2(n_fft)))
        input = torch.tensor(np.pad(input.cpu().numpy(), ((0, 0), (n_fft//2, n_fft//2)))).to('cuda')
        # [6, 352, 296] -> [6, 1, 296, 352]
        x = repeat(self.cqt(input), 'b f t -> b c t f', c=1)

        x = x.transpose(1, 3) 
        x = self.bn0(x)  # BatchNorm2d 
        x = x.transpose(1, 3)

        # (batch_size, 1, time_steps, mel_bins) =>
        # (batch_size, time_steps, classes_num)
        frame_output = self.frame_model(x)
        # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)
        # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)
        # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)

        # Use velocities to condition onset regression
        # [b, t, 88]
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5)
                      * velocity_output.detach()), dim=2)
        self.reg_onset_gru.flatten_parameters()
        (x, _) = self.reg_onset_gru(x)
        # extract features of onset
        if extract_onset:
            logging.info(f'ouput reg_onset_gru : {x.shape}')
            return {'onset_features': x}
        
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(),
                      reg_offset_output.detach()), dim=2)
        self.frame_gru.flatten_parameters()
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        # (batch_size, time_steps, classes_num)
        frame_output = torch.sigmoid(self.frame_fc(x))

        output_dict = {
            'reg_onset_output': reg_onset_output,
            'reg_offset_output': reg_offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}

        return output_dict
