import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import time
import librosa
import torch
import torch.nn as nn
import logging
from librosa.core.convert import mel_frequencies
import math

def make_cqt_attn_mask(n_bins=88, max_ratio=9, bin_ratio=1, bin_delta=0, density=1, sparse=True): # bin_ratio=1 1/2 1/4->352 176 88
    assert n_bins % 88 == 0
    if not sparse: # 非稀疏 返回全1矩阵(后面会False取反得True)
        return None
    # 计算mask矩阵，哪些地方不关注
    size = int(0.5 + n_bins * bin_ratio)

    fre_mask = torch.ones(size, size).bool()  # [88,88]
    assert bin_delta >= 0, f'bin_delta: {bin_delta} should be not smaller than 0!'
    for n_bin in range(size):
        for ratio in range(0, max_ratio + 1): # 1~8倍
            for d in range(1, density + 1): # density=2时, 计算 1/2 1 3/2 2 5/2 ... 即密度为1/2倍频
                # 默认 12*1*log2(0~8+1/1) = 0  12  19.02  24  27.86  31.02  33.69  36  38.04
                ratio_distance = n_bins // 88 * 12 * bin_ratio * math.log2(ratio + d / density)
                # print(f"{ratio_distance} =>")
                for bin_distance in range(int(ratio_distance + 0.25),int(ratio_distance + 1.75)):
                    # print(bin_distance)
                    for b in range(bin_distance - bin_delta, bin_distance + bin_delta + 1): # 倍频±bin_delta范围内所有频点
                        if n_bin + b < size and n_bin + b >= 0: 
                            fre_mask[n_bin][n_bin + b] = False
                        if n_bin - b < size and n_bin - b >= 0: 
                            fre_mask[n_bin][n_bin - b] = False

    mask = torch.zeros(size, size)
    mask.masked_fill_(fre_mask, float("-inf")) # true -> -inf
    # print(f'size:{size} bin_ratio:{bin_ratio} delta:{bin_delta} mask:{mask}')
    return mask

def make_mel_attn_mask(mel_config, max_ratio=9, bin_ratio=1, sparse=True): # bin_ratio=1 1/2 1/4->352 176 88
    if not sparse: # 非稀疏 返回全1矩阵(后面会False取反得True)
        return None
    # 计算mask矩阵，哪些地方不关注
    size = int(mel_config["n_bins"] * bin_ratio)
    scale = mel_frequencies(n_mels=mel_config["n_bins"], fmax=mel_config["fmax"], fmin=mel_config["fmin"])
    fre_mask = torch.ones(size, size).bool() 
    for bin in range(size):
        F = scale[bin]
        for ratio in range(1, max_ratio + 1):
            f = F * ratio
            for b in range(1, size):
                if scale[b-1] <= f and scale[b] >= f:
                    delta1 = f - scale[b-1]
                    delta2 = scale[b] - f
                    # print(f, scale[b-1], scale[b], delta1, delta2)
                    # 倍频在两mel频点之间, 两个都计算
                    if delta2 == 0 or delta1 / delta2 > 2.0:
                        fre_mask[bin][b] = False
                    elif delta1 / delta2 < 0.5:
                        fre_mask[bin][b-1] = False
                    else:
                        fre_mask[bin][b-1] = False
                        fre_mask[bin][b] = False

    # print(f'size:{size} bin_ratio:{bin_ratio} mask:{mask}')
    # torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    # print(fre_mask)
    mask = torch.zeros(size, size)
    mask.masked_fill_(fre_mask, float("-inf")) # true -> -inf, false -> 0
    return mask

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)



def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

 
def forward_dataloader(model, dataloader, batch_size, return_target=True):
    """Forward data generated from dataloader to model.

    Args:
      model: object
      dataloader: object, used to generate mini-batches for evaluation.
      batch_size: int
      return_target: bool

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        'frame_roll': (segments_num, frames_num, classes_num),
        'onset_roll': (segments_num, frames_num, classes_num),
        ...}
    """

    output_dict = {}
    device = next(model.parameters()).device

    for n, batch_data_dict in enumerate(dataloader):
        
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        # batch_waveform = move_data_to_device(batch_data_dict['mel_log'], device)

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)

        for key in batch_output_dict.keys():
            if '_list' not in key:
                append_to_dict(output_dict, key, 
                    batch_output_dict[key].data.cpu().numpy())

        if return_target:
            for target_type in batch_data_dict.keys():
                if 'roll' in target_type or 'reg_distance' in target_type or \
                    'reg_tail' in target_type:
                    append_to_dict(output_dict, target_type, 
                        batch_data_dict[target_type])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    
    return output_dict


def forward(model, x, batch_size):
    """Forward data to model in mini-batch. 
    
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    while True:
        if pointer >= len(x):
            break

        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)

        for key in batch_output_dict.keys():
            # if '_list' not in key:
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


if __name__ == '__main__':
    print(make_cqt_attn_mask())