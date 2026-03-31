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
import nnAudio
import torchaudio
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_utils import move_data_to_device, make_cqt_attn_mask

from hpptnet import Transformer, ConvTrans#, CQTSpectrogram
from einops import rearrange, repeat
import config


class CQTSpectrogram(torch.nn.Module):
    def __init__(self, sr, hop_length, fmin, n_bins, bins_per_octave, device='cuda', fmax=None, verbose=True, center=False):
        super(CQTSpectrogram, self).__init__()
        # NOTE: 此处center=True会自动pad
        self.spec_layer = nnAudio.features.cqt.CQT(sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax,  
                        n_bins=n_bins, bins_per_octave=bins_per_octave, verbose=verbose, center=center).to(device)   
        # self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.device = device
        # print('no amplitude_to_db')
        
    def forward(self, audio):
        """Computes cqt-spectrograms from a batch of waves
        RETURNS
        -------
        cqt_output: torch.FloatTensor of shape (B, T, n_bins)
        """
        cqt = self.spec_layer(audio.reshape(-1, audio.shape[-1])).transpose(-1, -2) 
        # print('with amplitude_to_db')
        # cqt = self.amplitude_to_db(cqt) 
        # cqt_output = self.spec_layer(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)  
        # cqt_output = torch.log(torch.clamp(cqt_output, min=1e-8))
        return cqt




class directOut(nn.Module):
    def __init__(self):
        super(directOut, self).__init__()

    def forward(self, x, pool_size=None, pool_type=None):
        return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class Gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                          bias=True, batch_first=True, dropout=0., bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]
    

class BiLSTM(nn.Module):
    # NOTE: hyper_para
    inference_chunk_length = 1000

    def __init__(self, input_features, recurrent_features, inference=False):
        super().__init__()
        #                       128                  64
        # self.inference = inference
        # print(self.inference, "*"*100)
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        # if self.training: #or self.inference:
            return self.rnn(x)[0]
        # else:
        #     # evaluation mode: support for longer sequences that do not fit in memory
        #     batch_size, sequence_length, input_features = x.shape
        #     hidden_size = self.rnn.hidden_size
        #     num_directions = 2 if self.rnn.bidirectional else 1

        #     h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        #     c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        #     output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

        #     # forward direction
        #     slices = range(0, sequence_length, self.inference_chunk_length)
        #     for start in slices:
        #         end = start + self.inference_chunk_length
        #         output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

        #     # reverse direction
        #     if self.rnn.bidirectional:
        #         h.zero_()
        #         c.zero_()

        #         for start in reversed(slices):
        #             end = start + self.inference_chunk_length
        #             result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
        #             output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

        #     return output
        

# hppnet.png  Hppnet-sp.png
class Regress_HPP(nn.Module):
    def __init__(self, frames_per_second, classes_num, model_type=''):
        super(Regress_HPP, self).__init__()

        self.model_type = model_type
        sample_rate = config.sample_rate # 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        cqt_bins = config.cqt_config['n_bins']
        fmin = config.cqt_config['fmin'] # 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        momentum = 0.01


        self.bn0 = nn.BatchNorm2d(cqt_bins, momentum)

        self.acoustis_model = HPPAcousticConvStack() # T * 88 * 128
        # TEST: PlanA: high_decode -- fc(88_128->768)+rnn(768->256*2)->fc(88) + onset_gru+frame_gru
        if model_type == "Regress_HPP_high_decode":
            self.onset_decode = nn.Sequential(
                nn.Linear(128*88, 768, bias=False),
                nn.BatchNorm1d(768, momentum=momentum),
                nn.GRU(input_size=768, hidden_size=256, 
                    num_layers=2,bias=True, batch_first=True,  dropout=0., bidirectional=True),
                nn.Linear(256*2, classes_num, bias=True)
            )
            self.offset_decode = nn.Sequential(
                nn.Linear(128*88, 768, bias=False),
                nn.BatchNorm1d(768, momentum=momentum),
                nn.GRU(input_size=768, hidden_size=256, 
                    num_layers=2, bias=True, batch_first=True,  dropout=0., bidirectional=True),
                nn.Linear(256*2, classes_num, bias=True)
            )
            self.frame_decode = nn.Sequential(
                nn.Linear(128*88, 768, bias=False),
                nn.BatchNorm1d(768, momentum=momentum),
                nn.GRU(input_size=768, hidden_size=256,     
                    num_layers=2, bias=True, batch_first=True,  dropout=0., bidirectional=True),
                nn.Linear(256*2, classes_num, bias=True)
            )
            self.velocity_decode = nn.Sequential(
                nn.Linear(128*88, 768, bias=False),
                nn.BatchNorm1d(768, momentum=momentum),
                nn.GRU(input_size=768, hidden_size=256, 
                    num_layers=2, bias=True, batch_first=True,  dropout=0., bidirectional=True),
                nn.Linear(256*2, classes_num, bias=True)
            )
            self.onset_gru = nn.GRU(input_size=classes_num * 2, 
                    hidden_size=256, num_layers=1,bias=True, batch_first=True, dropout=0., bidirectional=True),
            self.onset_fc = nn.Linear(512, classes_num, bias=True)
            
            self.frame_gru = nn.GRU(input_size=classes_num * 3, 
                    hidden_size=256, num_layers=1,bias=True, batch_first=True, dropout=0., bidirectional=True),
            self.frame_fc = nn.Linear(512, classes_num, bias=True)
            
        # TEST: PlanB: hpp_decode -- rnn(128->1)
        else:
            if model_type == "Regress_HPP_GRU":
                logging.info('use gru instead of lstm')
                self.onset_decode = nn.Sequential(
                    Gru(input_size=128, hidden_size=64),
                    nn.Linear(128, 1, bias=True)
                )
                self.offset_decode = nn.Sequential(
                    Gru(input_size=128, hidden_size=64),
                    nn.Linear(128, 1, bias=True)
                )
                self.frame_decode = nn.Sequential(
                    Gru(input_size=128, hidden_size=64),
                    nn.Linear(128, 1, bias=True)
                )
                self.velocity_decode = nn.Sequential(
                    Gru(input_size=128, hidden_size=64),
                    nn.Linear(128, 1, bias=True)
                )
            else:
                self.onset_decode = nn.Sequential(
                BiLSTM(128, 64, inference=False),
                nn.Linear(128, 1, bias=True)
                )
                self.offset_decode = nn.Sequential(
                    BiLSTM(128, 64, inference=False),
                    nn.Linear(128, 1, bias=True)
                )
                self.frame_decode = nn.Sequential(
                    BiLSTM(128, 64, inference=False),
                    nn.Linear(128, 1, bias=True)
                )
                self.velocity_decode = nn.Sequential(
                    BiLSTM(128, 64, inference=False),
                    nn.Linear(128, 1, bias=True)
                ) 
        self.init_weight()
        
        cqt_config = config.cqt_config
        num_heads = config.num_heads
        del cqt_config['n_fft']
        to_cqt = CQTSpectrogram(**cqt_config) # NOTE: 此处为本地CQT而非hpptnet

    def init_weight(self):
        init_bn(self.bn0)
        if self.model_type == "Regress_HPP_high_decode":
            init_gru(self.onset_gru)
            init_gru(self.frame_gru)
            init_layer(self.onset_fc)
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
        n_fft = cqt_config['sr'] / cqt_config['fmin'] / (2 ** (1 / cqt_config['bins_per_octave']) - 1)
        n_fft = 2 ** int(math.ceil(math.log2(n_fft)))
        input = torch.tensor(np.pad(input.cpu().numpy(), ((0, 0), (n_fft//2, n_fft//2)))).to('cuda')
        x = repeat(to_cqt(input), 'b f t -> b c f t', c=1)
        # x = x.transpose(2, 3)
        
        # x = x.transpose(1, 3)
        # x = self.bn0(x)  # BatchNorm2d
        # x = x.transpose(1, 3)

        x = self.acoustis_model(x) # -> [2, 128, 501, 88]
        b, c, t, n= x.shape
        x = x.transpose(1, 3).flatten(0,1) # -> [2*88, 501, 128]  
        onset = self.onset_decode(x).view(b, config.classes_num, t).transpose(1,2) # -> [176, 501, 1]
        offset = self.offset_decode(x).view(b, config.classes_num, t).transpose(1,2)
        frame = self.frame_decode(x).view(b, config.classes_num, t).transpose(1,2)
        velocity = self.velocity_decode(x).view(b, config.classes_num, t).transpose(1,2)

        # Use velocities to condition onset regression
        if self.model_type == "Regress_HPP_high_decode":
            x = torch.cat((onset, (onset ** 0.5) 
                           * velocity.detach()), dim=2)
            (x, _) = self.onset_gru(x)
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            if extract_onset:
                return x
            onset = torch.sigmoid(self.onset_fc(x))
            x = torch.cat((frame, onset.detach(), 
                offset.detach()), dim=2)
            (x, _) = self.frame_gru(x)
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            frame = torch.sigmoid(self.frame_fc(x))

        output_dict = {
            'reg_onset_output': onset,
            'reg_offset_output': offset,
            'frame_output': frame,
            'velocity_output': velocity}

        return output_dict
    
    
class Regress_HPP_SP(nn.Module):
    def __init__(self, frames_per_second, classes_num, model_type=''):
        super(Regress_HPP_SP, self).__init__()

        logging.info('='*20 + ' FuDan_SP ' + '='*20)
        self.model_type = model_type
        sample_rate = config.sample_rate # 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        cqt_bins = config.cqt_config['n_bins']
        fmin = config.cqt_config['fmin'] # 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        momentum = 0.01


        self.bn0 = nn.BatchNorm2d(cqt_bins, momentum)

        self.onset_acoustis_model = HPPAcousticConvStack() # T * 88 * 128
        self.frame_acoustis_model = HPPAcousticConvStack() # T * 88 * 128
        
        # NOTE: only hpp_decode -- rnn(128->1)
        if True:
            if True:
                self.onset_decode = nn.Sequential(
                BiLSTM(128, 64, inference=False),
                nn.Linear(128, 1, bias=True)
                )
                self.offset_decode = nn.Sequential(
                    BiLSTM(128, 64, inference=False),
                    nn.Linear(128, 1, bias=True)
                )
                self.frame_decode = nn.Sequential(
                    BiLSTM(128, 64, inference=False),
                    nn.Linear(128, 1, bias=True)
                )
                self.velocity_decode = nn.Sequential(
                    BiLSTM(128, 64, inference=False),
                    nn.Linear(128, 1, bias=True)
                ) 
        self.init_weight()
        
        cqt_config = config.cqt_config
        num_heads = config.num_heads
        del cqt_config['n_fft']
        to_cqt = CQTSpectrogram(**cqt_config) # NOTE: 此处为本地CQT而非hpptnet

    def init_weight(self):
        init_bn(self.bn0)
        if self.model_type == "Regress_HPP_high_decode":
            init_gru(self.onset_gru)
            init_gru(self.frame_gru)
            init_layer(self.onset_fc)
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
        n_fft = cqt_config['sr'] / cqt_config['fmin'] / (2 ** (1 / cqt_config['bins_per_octave']) - 1)
        n_fft = 2 ** int(math.ceil(math.log2(n_fft)))
        input = torch.tensor(np.pad(input.cpu().numpy(), ((0, 0), (n_fft//2, n_fft//2)))).to('cuda')
        cqt = repeat(to_cqt(input), 'b f t -> b c f t', c=1)
        # x = x.transpose(2, 3)
        
        # x = x.transpose(1, 3)
        # x = self.bn0(x)  # BatchNorm2d
        # x = x.transpose(1, 3)

        onset_feature = self.onset_acoustis_model(cqt) # -> [8, 128, 501, 88]
        b, c, t, n= onset_feature.shape
        onset_feature = onset_feature.transpose(1, 3).flatten(0,1) # -> [704, 501, 128]
        onset = self.onset_decode(onset_feature).view(b, config.classes_num, t).transpose(1,2) 
        
        src_size = list(cqt.size())
        src_size[-1] = 88
        # cqt = F.max_pool2d(cqt, [2,1])
        frame_feature = self.frame_acoustis_model(cqt) # -> [8, 128, 501, 88]
        frame_feature = F.interpolate(frame_feature, size=src_size[-2:], mode='bilinear') # [8, 128, 501, 88]
        frame_feature = frame_feature.transpose(1, 3).flatten(0,1)
        offset = self.offset_decode(frame_feature).view(b, config.classes_num, t).transpose(1,2)
        frame = self.frame_decode(frame_feature).view(b, config.classes_num, t).transpose(1,2)
        velocity = self.velocity_decode(frame_feature).view(b, config.classes_num, t).transpose(1,2)

        onset = torch.clip(onset, 1e-7, 1-1e-7)
        offset = torch.clip(offset, 1e-7, 1-1e-7)
        frame = torch.clip(frame, 1e-7, 1-1e-7)
        velocity = torch.clip(velocity, 1e-7, 1-1e-7)
        output_dict = {
            'reg_onset_output': onset,
            'reg_offset_output': offset,
            'frame_output': frame,
            'velocity_output': velocity}
    
        return output_dict
    

class HPPAcousticConvStack(nn.Module):  
    def __init__(self, conv_features=16, hd_conv_features=128, dilated_conv_feature=0):
        super().__init__()
        dilated_conv_feature = hd_conv_features if dilated_conv_feature == 0 else dilated_conv_feature

        # input is batch_size * 1 channel * 1000 frames * 352 input_features NOTE: considering InstanceNorm2d->BatchNorm2d
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, conv_features, (7, 7), padding=3),  
            nn.ReLU(),
            nn.InstanceNorm2d(conv_features),
            # layer 1
            nn.Conv2d(conv_features, conv_features, (7, 7), padding=3),
            nn.ReLU(),
            nn.InstanceNorm2d(conv_features),
            # layer 2
            nn.Conv2d(conv_features, conv_features, (7, 7), padding=3),
            nn.ReLU(),
            nn.InstanceNorm2d(conv_features)
        )

        # [48 76 96 111 124 135 144 152]
        self.dilation_list = self._make_freq_dialtion(config.cqt_config)
        for idx, dilation in enumerate(self.dilation_list):
            setattr(self, f"hd_conv_{idx}", nn.Conv2d(conv_features, hd_conv_features, (1, 3), padding=(0,dilation), dilation=(1, dilation)))

        dcnn_padding = (2, 12)
        """ self.d_cnn = nn.Sequential(  
            nn.MaxPool2d((1,4)),
            nn.Conv2d(hd_conv_features, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature)
        ) # T * 88 * 128 """
        # logging.info('='*20 + ' Reproduce ' + '='*20)
        # self.d_cnn = nn.Sequential(  
        #     nn.MaxPool2d((1,4)),
        #     nn.Conv2d(hd_conv_features, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(dilated_conv_feature),
        #     nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(dilated_conv_feature),
        #     nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(dilated_conv_feature),
        #     nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 3), padding=dcnn_padding, dilation=(1,12)),
        #     nn.ReLU(),
        #     nn.InstanceNorm2d(dilated_conv_feature)
        # ) # T * 88 * 128  
        
        logging.info('='*20 + ' FuDan ' + '='*20)
        self.d_cnn = nn.Sequential(  
            nn.Conv2d(hd_conv_features, dilated_conv_feature, (1, 3), padding='same', dilation=(1,48)),
            nn.ReLU(),
            nn.MaxPool2d((1,4), stride=[1, 4], padding=0, dilation=1, ceil_mode=False),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (1, 3), padding='same', dilation=(1,12)),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 1), padding='same'),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 1), padding='same'),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature),
            nn.Conv2d(dilated_conv_feature, dilated_conv_feature, (5, 1), padding='same'),
            nn.ReLU(),
            nn.InstanceNorm2d(dilated_conv_feature)
        ) # T * 88 * 128
        

    def _make_freq_dialtion(self, cqt_config, max_ratio=9, bin_ratio=1, bin_delta=0): 
        size = int(0.5 + cqt_config['n_bins'] * bin_ratio)

        dilation = []
        assert bin_delta >= 0, f'bin_delta: {bin_delta} should be not smaller than 0!'
        for ratio in range(2, max_ratio + 1):
            # "bins_per_octave": 48
            bin_distance = int(0.5 + cqt_config['bins_per_octave'] * bin_ratio * math.log2(ratio))
            for bin in range(bin_distance - bin_delta, bin_distance + bin_delta + 1):
                if bin < size and bin >= 0: #
                    dilation.append(bin)
        # [48 76 96 111 124 135 144 152]
        print(dilation)
        exit
        return dilation

    def forward(self, cqt):
        # x = cqt.view(cqt.size(0), 1, cqt.size(1), cqt.size(2)) 
        x = self.cnn(cqt) # [2, 1, 501, 352] -> [2, 16, 501, 352]
        y = None
        for i in range(0, len(self.dilation_list)):
            cur_hd_conv = getattr(self, f"hd_conv_{i}")
            t = cur_hd_conv(x) # -> [2, 128, 501, 352]
            if i == 0:
                y = t
            else:
                y += t
        z = self.d_cnn(y) # -> [2, 128, 501, 88])
        return z
    
    
class HPPTransAcousticConvStack(nn.Module):  
    def __init__(self, conv_features=16, hd_conv_features=128, dilated_conv_feature=0):
        super().__init__()
        dilated_conv_feature = hd_conv_features if dilated_conv_feature == 0 else dilated_conv_feature

        # input is batch_size * 1 channel * 1000 frames * 352 input_features NOTE: considering InstanceNorm2d->BatchNorm2d
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, conv_features, (7, 7), padding=3),  
            nn.ReLU(),
            nn.InstanceNorm2d(conv_features),
            Transformer(conv_features, num_heads=1),
            # layer 1
            nn.Conv2d(conv_features, conv_features, (7, 7), padding=3),
            nn.ReLU(),
            nn.InstanceNorm2d(conv_features),
            Transformer(conv_features, num_heads=1),
            # layer 2
            nn.Conv2d(conv_features, conv_features, (7, 7), padding=3),
            nn.ReLU(),
            nn.InstanceNorm2d(conv_features),
            Transformer(conv_features, num_heads=1)
        )

        # [48 76 96 111 124 135 144 152]
        self.dilation_list = self._make_freq_dialtion(config.cqt_config)
        for idx, dilation in enumerate(self.dilation_list):
            setattr(self, f"hd_conv_{idx}", nn.Conv2d(conv_features, hd_conv_features, (1, 3), padding=(0,dilation), dilation=(1, dilation)))
        
        freq_attn_mask_0 = make_cqt_attn_mask(cqt_config)
        self.trans_0 = Transformer(hd_conv_features, num_heads, freq_attn_mask_0)
        self.max_pool = self.get_conv2d_block(hd_conv_features, dilated_conv_feature, pool_size=[1, 4], dilation=[1, 48])
        freq_attn_mask_1 = make_cqt_attn_mask(cqt_config, bin_ratio=1/4)
        self.trans_1 = Transformer(dilated_conv_feature, num_heads, freq_attn_mask_1)
        self.dcnn_1 = self.get_conv2d_block(dilated_conv_feature, dilated_conv_feature, dilation=[1, 12])
        self.trans_2 = Transformer(dilated_conv_feature, num_heads, freq_attn_mask_1)
        self.dcnn_2 = self.get_conv2d_block(dilated_conv_feature, dilated_conv_feature, [5,1])
        self.trans_3 = Transformer(dilated_conv_feature, num_heads, freq_attn_mask_1)
        self.dcnn_3 = self.get_conv2d_block(dilated_conv_feature, dilated_conv_feature, [5,1])
        self.trans_4 = Transformer(dilated_conv_feature, num_heads, freq_attn_mask_1)
        self.dcnn_4 = self.get_conv2d_block(dilated_conv_feature, dilated_conv_feature, [5,1])
        self.trans_5 = Transformer(dilated_conv_feature, num_heads, freq_attn_mask_1)
        
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out)
            )

    def _make_freq_dialtion(self, cqt_config, max_ratio=9, bin_ratio=1, bin_delta=0): 
        size = int(0.5 + cqt_config['n_bins'] * bin_ratio)

        dilation = []
        assert bin_delta >= 0, f'bin_delta: {bin_delta} should be not smaller than 0!'
        for ratio in range(2, max_ratio + 1):
            # "bins_per_octave": 48
            bin_distance = int(0.5 + cqt_config['bins_per_octave'] * bin_ratio * math.log2(ratio))
            for bin in range(bin_distance - bin_delta, bin_distance + bin_delta + 1):
                if bin < size and bin >= 0: #
                    dilation.append(bin)
        # [48 76 96 111 124 135 144 152]
        return dilation

    def forward(self, cqt):
        # x = cqt.view(cqt.size(0), 1, cqt.size(1), cqt.size(2)) 
        x = self.cnn(cqt) # [2, 1, 501, 352] -> [2, 16, 501, 352]
        y = None
        for i in range(0, len(self.dilation_list)):
            cur_hd_conv = getattr(self, f"hd_conv_{i}")
            t = cur_hd_conv(x) # -> [2, 128, 501, 352]
            if i == 0:
                y = t
            else:
                y += t
        y = torch.relu(y)
        z = self.max_pool(y)
        z = self.dcnn_1(z)
        z = self.dcnn_2(z)
        z = self.dcnn_3(z)
        z = self.dcnn_4(z)
        # -> [2, 128, 501, 88])
        return z
    