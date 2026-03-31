import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt

from timm.models.layers import DropPath, trunc_normal_

import logging

DROP = 0.5
attn_show = 0.00
attn_global = [None,None,None,None,None,None,None,None,None,None,None,None]
attn_index = 0
attn_count = 0
date_time = datetime.datetime.now().strftime('%m%d_%H%M')
flops = 0

def count_flops(q, k, fold=False):
    global flops 
    print(f'q:{q.shape}\tk:{k.shape}')
    # attn = (q @ k.transpose(-2, -1))
    # flops += self.num_heads * N * (self.dim // self.num_heads) * N
    # x = (attn @ v)
    # flops += self.num_heads * N * N * (self.dim // self.num_heads)
    if fold:
        flops += 2 * q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3] * q.shape[4] * k.shape[3]
    else:
        flops += 2 * q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3] * k.shape[2]

def unfold_cqt_spec(cqt_spc): 
    N, H, F, C = cqt_spc.shape # [401, 352, 64]  [401, 2, 352, 32]
    assert F % 88 == 0
    n_bins = F
    
    bins_per_octave = n_bins // 88 * 12
    cqt_spc_pad = torch.cat((cqt_spc, torch.zeros(N, H, bins_per_octave * 8 - F, C).to('cuda')), 2)
    cqt_spc_unfolded = cqt_spc_pad.reshape(N, H, 8, bins_per_octave, C).transpose(2,3)
    return cqt_spc_unfolded

def fold_cqt_spec(cqt_spc_unfolded): 
    N, H, F, M, C = cqt_spc_unfolded.shape # [401, 352, 64]    [401, 2, 48, 8, 32]
    assert F % 12 == 0 and M == 8, f"F({F}) % 12 != 0 or M({M}) % 8 != 0"
    
    cqt_spc_fold = cqt_spc_unfolded.transpose(2,3).flatten(start_dim=2, end_dim=3)[:,:,:-8 * (F // 12),:]
    return cqt_spc_fold


def get_sinusoid_encoding_table(n_position, dim):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / dim)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(dim)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=DROP):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_mask=None, qkv_bias=False, qk_scale=None, attn_drop=DROP, proj_drop=DROP):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # self.qkv = nn.Linear(dim, dim * 1, bias=qkv_bias)

        self.fc_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.fc_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.fc_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if attn_mask is not None:
            # 在模型中注册一个不需要求梯度的张量
            self.register_buffer('attn_mask', attn_mask) 

    def forward(self, x, fold=False):
        B, N, C = x.shape # [1002, 230, 256]
        # qkv = self.qkv(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[0], qkv[0]  # make torchscript happy (cannot use tensor as tuple)
        # print(x.shape)
        q = self.fc_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.fc_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.fc_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print(f'{q.shape} {k.shape} {v.shape}')
        if fold:
            q = unfold_cqt_spec(q)
            k = unfold_cqt_spec(k)
            v = unfold_cqt_spec(v)
        
        # count_flops(q, k, fold)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [201, self.num_heads, 230, 230]
        if hasattr(self, 'attn_mask'):
        # if self.training and hasattr(self, 'attn_mask') and random.random() < 0.5:
            # attn[:,0,:,:] += self.attn_mask
            attn += self.attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # show attn matrix
        if not self.training and attn_show > 0:
            global attn_index
            global attn_global
            global attn_count
            
            if fold:
                attn_local = attn.mean(dim=(0, 1, 2))
            else:
                attn_local = attn.mean(dim=(0, 1))
            
            if attn_global[attn_index] is not None:
                attn_global[attn_index] += attn_local
            else:
                attn_global[attn_index] = attn_local
            
            if random.random() < attn_show or (attn_count > 0 and attn_count % 100 == 0):
                hotmap_dir = os.path.join('workspaces','hotmap',datetime.datetime.now().strftime('%y%m%d'))
                os.makedirs(hotmap_dir, exist_ok=True)
                
                hotmap_name = os.path.join(hotmap_dir, f"{date_time}_{attn_index+1}_{N}×{N}")
                plt.imshow(attn_global[attn_index].cpu(), cmap='viridis', interpolation='nearest', aspect='auto')
                plt.savefig(hotmap_name + '.png')
                print(f"hotmap saved at: {hotmap_name + '.png'}")
            
            attn_index = (attn_index + 1) % len(attn_global)
            if attn_index == 0:
                if attn_count == 1000:
                    exit()
                attn_count += 1
        
        attn = attn @ v
        
        # count_flops(attn, v, fold)
        # print(f'{flops / 1e9}GMac')
        
        if fold:
            attn = fold_cqt_spec(attn)

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransBlock(nn.Module):
    """ Transformer """
    def __init__(self, dim, num_heads, attn_mask=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=DROP, attn_drop=DROP,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads, attn_mask, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    """ 
    1*1 + 3*3 + 1*1 卷积块(cnn+bn+relu) 第二层3*3可单独输出给Transformer, 
    最后一层1*1输出 与 输入或输入的1*1卷积 残差相加作为最终的输出 
    """
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None, bias=True):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True) # nn.ReLU

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=bias)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True) # nn.ReLU

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True) # nn.ReLU

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=bias)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)


        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, act_layer=nn.GELU,# dw_stride,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), has_token=True):
        super(FCUDown, self).__init__()
        # self.dw_stride = dw_stride
        # if inplanes == outplanes: 
        #     self.conv_project = nn.Identity()
        # else:
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        # self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.has_token = has_token
        
    def forward(self, x, x_t=None):
        # print(f"FCUDown: x_in[{x.shape}]")
        x = self.conv_project(x)  # [N, C, H, W]   [2, 64, 501, 229] -> [2, 384, 501, 229])

        # x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = x.transpose(1, 2).flatten(start_dim=0, end_dim=1).transpose(1,2) # [1002, 229, 384]
        x = self.ln(x)
        x = self.act(x)

        if self.has_token and x_t:
            x = torch.cat([x_t[:, 0][:, None, :], x], dim=1) # [1002, 230, 384]
        # print(f"FCUDown: x_out[{x.shape}]")
        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, act_layer=nn.ReLU,# up_stride,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), has_token=True):
        super(FCUUp, self).__init__()

        # self.up_stride = up_stride
        # if inplanes == outplanes:
        #     self.conv_project = nn.Identity()
        # else:
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
            
        self.bn = norm_layer(outplanes)
        self.act = act_layer()
        self.has_token = has_token
        
    def forward(self, x, B, T):
        # print(f"FCUUp: x_in[{x.shape}]")
        _, F, C = x.shape # [1002, 230, 384]
        if self.has_token:
            x_r = x[:, 1:]
            F -= 1  
        else:
            x_r = x
        x_r = x_r.transpose(1, 2).reshape(B, T, C, F).transpose(1, 2) # [2, 384, 501, 229]
        x_r = self.act(self.bn(self.conv_project(x_r))) # [2, 64, 501, 229]
        return x_r #F.interpolate(x_r, size=(T * self.up_stride, F * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None, bias=True):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=bias)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, embed_dim, num_heads=12, mlp_ratio=4.,# dw_stride,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, attn_mask=None, has_token=True, bias=True):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups, bias=bias)

        if last_fusion: # 仅总模型最后一层
            # self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, res_conv=True, groups=groups, bias=bias)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups, bias=bias)

        if num_med_block > 0: # 一直为0
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups, bias=bias))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, has_token=has_token)#, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, has_token=has_token)#, up_stride=dw_stride)

        self.trans_block = TransBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_mask=attn_mask,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        # self.dw_stride = dw_stride
        # self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        # self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        N, C, T, F = x.shape # [2, 1, 501, 229]
        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, N, T)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


# no TransBlock
class ConvTransBlock_1(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, embed_dim, num_heads=12, mlp_ratio=4.,# dw_stride,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, attn_mask=None, has_token=True, bias=True):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups, bias=bias)

        if last_fusion: # 仅总模型最后一层
            # self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, res_conv=True, groups=groups, bias=bias)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups, bias=bias)

        if num_med_block > 0: # 一直为0
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups, bias=bias))
            self.med_block = nn.ModuleList(self.med_block)

        # self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, has_token=has_token)#, dw_stride=dw_stride)
        # self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, has_token=has_token)#, up_stride=dw_stride)

        # self.trans_block = TransBlock(
        #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_mask=attn_mask,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        # self.dw_stride = dw_stride
        # self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        # self.last_fusion = last_fusion

    def forward(self, x, x_t=None):
        x, x2 = self.cnn_block(x)
        N, C, T, F = x.shape # [2, 1, 501, 229]
        _, _, H, W = x2.shape

        # x_st = self.squeeze_block(x2, x_t)

        # x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        # x_t_r = self.expand_block(x_t, N, T)
        # x = self.fusion_block(x, x_t_r, return_x_2=False)
        x = self.fusion_block(x, return_x_2=False)

        return x, None

# no ConvBlock
class ConvTransBlock_2(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, embed_dim, num_heads=12, mlp_ratio=4.,# dw_stride,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, attn_mask=None, has_token=True, bias=True):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        # self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups, bias=bias)

        # if last_fusion: # 仅总模型最后一层
        #     # self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        #     self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, res_conv=True, groups=groups, bias=bias)
        # else:
        #     self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups, bias=bias)

        if num_med_block > 0: # 一直为0
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups, bias=bias))
            self.med_block = nn.ModuleList(self.med_block)

        # self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, has_token=has_token)#, dw_stride=dw_stride)
        # self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, has_token=has_token)#, up_stride=dw_stride)

        self.trans_block = TransBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_mask=attn_mask,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        # self.dw_stride = dw_stride
        # self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        # self.last_fusion = last_fusion

    def forward(self, x, x_t):
        # x, x2 = self.cnn_block(x)
        # N, C, T, F = x.shape # [2, 1, 501, 229]
        # _, _, H, W = x2.shape

        # x_st = self.squeeze_block(x2, x_t)

        # x_t = self.trans_block(x_st + x_t)
        x_t = self.trans_block(x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        # x_t_r = self.expand_block(x_t, N, T)
        # x = self.fusion_block(x, x_t_r, return_x_2=False)
        # x = self.fusion_block(x, return_x_2=False)

        return None, x_t

#DBCA
class Conformer(nn.Module):
    def __init__(self, in_chans=1, num_classes=88, base_channel=16, channel_ratio=4, num_med_block=0, incre_xt=False, 
                 embed_dim=768, depth=9, num_heads=12, mlp_ratio=2., qkv_bias=False, qk_scale=None, branch=1, # patch_size=16,
                 drop_rate=DROP, attn_drop_rate=DROP, drop_path_rate=DROP, attn_mask=None, has_token=True, n_bin=229, bias=True, cqt_or_mel=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        
        self.cnn_channel = base_channel # 16
        self.cnn_kernel = 3
        self.hid_dim = 256
        # self.n_frame = 128 # 
        self.n_bin = n_bin # 229
        self.incre_xt = incre_xt
        self.cqt_or_mel = cqt_or_mel
        
        # self.pos_embedding_freq = nn.Embedding(n_bin, self.cnn_channel)
        self.scale_freq = torch.sqrt(torch.FloatTensor([self.cnn_channel])).to('cuda')
        self.dropout = nn.Dropout(0.1)

        if has_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.cls_token = None
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        # self.trans_cls_head = nn.Linear(embed_dim, 1) if num_classes > 0 else nn.Identity()
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)
        # self.conv_cls_head = nn.Linear(int(256), 1)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv0 = nn.Conv2d(in_chans, self.cnn_channel, kernel_size=self.cnn_kernel, padding='same')
        # NOTE: 1D-Conv in Time axis
        # self.conv0 = nn.Conv2d(in_chans, self.cnn_channel, kernel_size=(5,1), padding='same')
        self.bn0 = nn.BatchNorm2d(self.cnn_channel)
        self.act0 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]
        # self.fc0 = nn.Linear(self.cnn_dim, self.hid_dim)

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio) # 16 * 4 = 64
        # trans_dw_stride = patch_size // 4 # (16 | 16 | 32) / 4 = 4 | 4 | 8
        self.conv_1 = ConvBlock(inplanes=self.cnn_channel, outplanes=stage_1_channel, res_conv=True, stride=1, bias=bias)
        # self.trans_patch_conv = nn.Conv2d(self.cnn_channel, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0) # 64 -> 384, (4,4)
        # TODO:  kernel_size=self.cnn_channel -> self.cnn_kernel
        if self.incre_xt:
            self.embed_dim = stage_1_channel
        self.trans_patch_conv = nn.Conv2d(self.cnn_channel, self.embed_dim, kernel_size=self.cnn_channel, stride=1, padding='same') # 16 -> 64
        self.trans_1 = TransBlock(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0], attn_mask=attn_mask[0])

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1 # 12 / 3 + 1 = 5

        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, embed_dim=self.embed_dim, #dw_stride=trans_dw_stride,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, attn_mask=attn_mask[i-1], has_token=has_token, bias=bias
                    )
            )
            # 1 TODO: forward 
            # if i == fin_stage - 1:
            #     self.conv_fre_down_fc = nn.Linear(self.n_bin + (1 if has_token else 0), num_classes + (1 if has_token else 0))
            #     self.trans_fre_down_fc = nn.Linear(self.n_bin + (1 if has_token else 0), num_classes + (1 if has_token else 0))
            #     if attn_mask != None:
            #         attn_mask = make_cqt_attn_mask()
            if self.cqt_or_mel and i == fin_stage - 1:
                self.conv_fre_down_fc_4 = nn.Linear(self.n_bin + (1 if has_token else 0), self.n_bin // 2 + (1 if has_token else 0))
                self.trans_fre_down_fc_4 = nn.Linear(self.n_bin + (1 if has_token else 0), self.n_bin // 2 + (1 if has_token else 0))

        stage_2_channel = int(base_channel * channel_ratio * 2) # 64 * 4 * 2 = 512
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        if self.incre_xt:
            self.embed_dim = stage_2_channel
            self.trans_patch_conv_2 = nn.Conv2d(stage_1_channel, stage_2_channel, kernel_size=self.cnn_kernel, stride=1, padding='same') # 64 -> 128
        for i in range(init_stage, fin_stage):
            # stride = 2 if i == init_stage else 1
            stride = 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                ConvTransBlock(
                    in_channel, stage_2_channel, res_conv, stride, embed_dim=self.embed_dim,# dw_stride=trans_dw_stride // 2, 
                    num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                    num_med_block=num_med_block, attn_mask=attn_mask[i-1], has_token=has_token, bias=bias
                )
            )
            # 2 TODO: forward 
            if i == fin_stage - 1:
                if self.cqt_or_mel:
                    self.conv_fre_down_fc_8 = nn.Linear(self.n_bin // 2 + (1 if has_token else 0), num_classes + (1 if has_token else 0))
                    self.trans_fre_down_fc_8 = nn.Linear(self.n_bin // 2 + (1 if has_token else 0), num_classes + (1 if has_token else 0))
                else:
                    self.conv_fre_down_fc = nn.Linear(self.n_bin + (1 if has_token else 0), num_classes + (1 if has_token else 0))
                    self.trans_fre_down_fc = nn.Linear(self.n_bin + (1 if has_token else 0), num_classes + (1 if has_token else 0))

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2) # 64 * 4 * 2 * 2 = 1024
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        if self.incre_xt:
            self.embed_dim = stage_3_channel
            self.trans_patch_conv_3 = nn.Conv2d(stage_2_channel, stage_3_channel, kernel_size=self.cnn_kernel, stride=1, padding='same') # 128 -> 256
        for i in range(init_stage, fin_stage): # 9~12
            # stride = 2 if i == init_stage else 1
            stride = 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                ConvTransBlock(
                    in_channel, stage_3_channel, res_conv, stride, embed_dim=self.embed_dim, #dw_stride=trans_dw_stride // 4, 
                    num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                    num_med_block=num_med_block, last_fusion=last_fusion, attn_mask=attn_mask[i-1], has_token=has_token, bias=bias
                )
            )
        if branch > 1:
            for n_branch in range(2, branch + 1):
                for i in range(init_stage, fin_stage):
                    stride = 1
                    in_channel = stage_2_channel if i == init_stage else stage_3_channel
                    res_conv = True if i == init_stage else False
                    last_fusion = True if i == depth else False
                    self.add_module('conv_trans_' + str(i) + f'.{n_branch}',
                        ConvTransBlock(
                            in_channel, stage_3_channel, res_conv, stride, embed_dim=self.embed_dim, #dw_stride=trans_dw_stride // 4, 
                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                            num_med_block=num_med_block, last_fusion=last_fusion, attn_mask=attn_mask[i-1], has_token=has_token, bias=bias
                        )
                    )
        self.init_stage = init_stage
        self.fin_stage = fin_stage
        self.branch = branch
        self.trans_norm = nn.LayerNorm(self.embed_dim)

        if self.cls_token != None:
            trunc_normal_(self.cls_token, std=.02) # 用于对张量中的值进行截断正态分布初始化。std参数指定了正态分布的标准差。默认情况下，mean为0

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, spec):
        N, _, T, F = spec.shape # [2, 1, 501, 229]
        if self.cls_token != None:
            cls_tokens = self.cls_token.expand(N * T, -1, -1) # [B, 1, 384]

        # pdb.set_trace()
        # stem stage [N, 1, T, 256] -> [N, 16, T, 256] 
        x_base = self.act0(self.bn0(self.conv0(spec))) # [2, 16, 501, 229]

        # pos_freq = self.pos_embedding_freq(torch.arange(0, self.n_bin).unsqueeze(0).repeat(N * T, 1).to('cuda')).reshape(N, T, self.n_bin, self.cnn_channel).permute(0, 3, 1, 2) # [2, 16, 129, 229]
        # x_base = self.dropout(x_base * self.scale_freq + pos_freq)
        # x_base = x_base + pos_freq

        # logging.info(f'x: {x.shape}     x_base: {x_base.shape}') #[2, 1, 501, 229] 
        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)
        # x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2) # [N, 384, T, 256] -> [N, 14*14=196, 384]
        x_t = self.trans_patch_conv(x_base).transpose(1, 2).flatten(start_dim=0, end_dim=1).transpose(1, 2).contiguous() # [N * T, 256, 384]   [1002, 229, 384]
        if self.cls_token != None:
            x_t = torch.cat([cls_tokens, x_t], dim=1) # [N, 257, 384]  [1002, 230, 384]
        x_t = self.trans_1(x_t) # [1002, 230, 384]
        # print('[1] x: {} | x_t: {}'.format(x.shape, x_t.shape))
        
        x_branch = []
        x_t_branch = []
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            # [2] x: torch.Size([1, 64, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [3] x: torch.Size([1, 64, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [4] x: torch.Size([1, 64, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [5] x: torch.Size([1, 128, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [6] x: torch.Size([1, 128, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [7] x: torch.Size([1, 128, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [8] x: torch.Size([1, 128, 128, 229]) | x_t: torch.Size([128*1, 229, 64])
            # [9] x: torch.Size([1, 256, 128, 88]) | x_t: torch.Size([128*1, 88, 64])
            # [10] x: torch.Size([1, 256, 128, 88]) | x_t: torch.Size([128*1, 88, 64])
            # [11] x: torch.Size([1, 256, 128, 88]) | x_t: torch.Size([128*1, 88, 64])
            # [12] x: torch.Size([1, 256, 128, 88]) | x_t: torch.Size([128*1, 88, 64]) 
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            # print("[{}] x: {} | x_t: {}".format(i, x.shape, x_t.shape), end = '\t')
            
            if i >= self.init_stage: # last conformer block
                for n_branch in range(2, self.branch + 1):
                    if i == self.init_stage:
                        tmp_x, tmp_x_t = eval('self.conv_trans_' + str(i) + f'.{n_branch}')(x, x_t)
                        x_branch.append(tmp_x)
                        x_t_branch.append(tmp_x_t)
                    else:
                        x_branch[n_branch-2], x_t_branch[n_branch-2] = \
                            eval('self.conv_trans_' + str(i) + f'.{n_branch}')(x_branch[n_branch-2], x_t_branch[n_branch-2])
            
            if i == (self.fin_stage - 1) // 3 * 1:
                if self.incre_xt:
                    # [4] => x_t: torch.Size([128, 229, 128])
                    x_t = self.trans_patch_conv_2(x_t.reshape(x.shape[0], x_t.shape[0]//x.shape[0], x_t.shape[1], x_t.shape[2]).transpose(1,3)).transpose(1,3).flatten(start_dim=0, end_dim=1).contiguous() # 
                # 频率压缩 352 -> 176
                if self.cqt_or_mel:
                    x = self.conv_fre_down_fc_4(x)
                    x_t = self.trans_fre_down_fc_4(x_t.transpose(1,2)).transpose(1,2).contiguous()
                
            # TODO: 触发 8 -> 4
            if i == (self.fin_stage - 1) // 3 * 2:
                if self.incre_xt:
                    # [8] => x_t: torch.Size([128, 229, 256])
                    x_t = self.trans_patch_conv_3(x_t.reshape(x.shape[0], x_t.shape[0]//x.shape[0], x_t.shape[1], x_t.shape[2]).transpose(1,3)).transpose(1,3).flatten(start_dim=0, end_dim=1).contiguous() # [N * T, 256, 384]   [1002, 229, 384]
                if self.cqt_or_mel:
                    # 频率压缩 176 -> 88
                    x = self.conv_fre_down_fc_8(x)
                    x_t = self.trans_fre_down_fc_8(x_t.transpose(1,2)).transpose(1,2).contiguous()
                else: 
                    # 频率压缩 229 -> 88
                    x = self.conv_fre_down_fc(x)
                    x_t = self.trans_fre_down_fc(x_t.transpose(1,2)).transpose(1,2).contiguous()

        conv_cls = self.get_conv_cls(x) 
        # logging.info(conv_cls.shape)

        tran_cls = self.get_tran_cls(x_t, N, T)
        # tran_cls = self.trans_cls_head(x_t[:, 0]).squeeze(1)
        if self.branch == 1:
            return [conv_cls, tran_cls]
        
        branch_cls = [[conv_cls],[tran_cls]]
        for x in x_branch:
            branch_cls[0].append(self.get_conv_cls(x))
        for x_t in x_t_branch:
            branch_cls[1].append(self.get_tran_cls(x_t, N, T))
        return branch_cls

    def get_conv_cls(self, x):
        # conv classification
        # x_p = self.pooling(x).flatten(1)
        # conv_cls = self.conv_cls_head(x_p)
        x_p = x.transpose(1,3).contiguous() # [2, 256, 501, 229] -> [2, 229, 501, 256]
        conv_cls = x_p #
        # conv_cls = self.conv_cls_head(x_p).squeeze() # [2, 229, 501]
        # logging.info(conv_cls.shape)
        return conv_cls
    
    def get_tran_cls(self, x_t, N, T):
        # trans classification
        # x_t = self.trans_norm(x_t)
        # tran_cls = self.trans_cls_head(x_t[:, 0])
        x_t = self.trans_norm(x_t) # [1002, 230, 64]
        # torch.Size([501, 229, 64])
        if self.cls_token != None:
            x_t = x_t[:, 1:, :]
        _, F, C = x_t.shape
        tran_cls = x_t.reshape(N, T, F, C).transpose(1,2).contiguous() # [2, 229, 501, 64]
        # tran_cls = self.trans_cls_head(x_t[:, 0]).squeeze(1)
        return tran_cls

class HAT(nn.Module):
    def __init__(self, in_chans=1, num_classes=88, base_channel=16, channel_ratio=2, num_med_block=0, incre_xt=False, pos_freq=False, 
                 embed_dim=64, depth=8, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, branch=1, # patch_size=16,
                 drop_rate=DROP, attn_drop_rate=DROP, drop_path_rate=DROP, attn_mask=None, has_token=True, n_bin=229, bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 2 == 0
        
        self.cnn_channel = base_channel # 16
        self.cnn_kernel = 7
        self.hid_dim = 64
        self.n_bin = n_bin # 256
        
        self.dropout = nn.Dropout(0.1)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv0 = nn.Conv2d(in_chans, self.cnn_channel, kernel_size=self.cnn_kernel, padding='same')
        self.bn0 = nn.BatchNorm2d(self.cnn_channel)
        self.act0 = nn.ReLU(inplace=True)

        # 0 stage
        self.trans_patch_conv = nn.Conv2d(self.cnn_channel, self.embed_dim, kernel_size=self.cnn_kernel, stride=1, padding='same') # 16 -> 64
        self.scale_freq = torch.sqrt(torch.FloatTensor([self.embed_dim])).to('cuda')
        self.pos_freq = None if not pos_freq else torch.arange(0, self.n_bin).unsqueeze(0).to('cuda') # 线性位置编码
        # self.pos_embedding_freq_learnable = nn.Embedding(self.n_bin, self.embed_dim)
        self.pos_embedding_freq = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.n_bin, self.embed_dim), freeze=True) # 正弦位置编码
        # self.pos_embedding_freq = nn.Embedding.from_pretrained(torch.sin(torch.arange(0, self.n_bin) / (n_bin / 88 * 12)).unsqueeze(0).repeat(self.embed_dim, 1).t(), freeze=True) # 谐波位置编码


        # 1~8 stage
        init_stage = 1
        self.fin_stage = depth + 1
        for i in range(init_stage, self.fin_stage):
            self.add_module('har_trans_' + str(i),
                TransBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                           attn_mask=attn_mask[i-1], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
            )
            if i == depth // 2:
                # self.fre_down_pool_4 = nn.AvgPool2d(kernel_size=(1,2))
                self.fre_down_fc_4 = nn.Linear(self.n_bin, self.n_bin // 2)
            elif i == depth // 4 * 3:
                # self.fre_down_pool_6 = nn.AvgPool2d(kernel_size=(1,2))
                self.fre_down_fc_6 = nn.Linear(self.n_bin//2, num_classes)

        # self.fre_down_fc = nn.Linear(self.n_bin + (1 if has_token else 0), num_classes + (1 if has_token else 0))
        # self.fre_down_conv = nn.Conv2d(self.n_bin + (1 if has_token else 0), num_classes + (1 if has_token else 0), kernel_size=1, stride=1, padding='same')
        # self.fre_down_pool = nn.AvgPool2d(kernel_size=(1,4))

        self.trans_norm = nn.LayerNorm(self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, spec):
        N, _, T, F = spec.shape # [1, 1, 401, 352] 8s * 352bins
    
        x = self.act0(self.bn0(self.conv0(spec))) # [2, 16, 501, 229]
        x = self.trans_patch_conv(x).transpose(1, 2).flatten(start_dim=0, end_dim=1).transpose(1, 2).contiguous() # [N * T, 256, 384]   [1002, 229, 384]

        # 位置编码
        if self.pos_freq is not None:
            # torch.Size([401, 352, 64])
            pos_freq = self.pos_embedding_freq(self.pos_freq.repeat(N * T, 1)).reshape(N * T, self.n_bin, self.embed_dim) 
            x = self.dropout(x * self.scale_freq + pos_freq)

        # 1 ~ 8
        for i in range(1, self.fin_stage):
            x = eval('self.har_trans_' + str(i))(x)
            # print("[{}] hat_x: {}".format(i, x.shape), end = '\t') # torch.Size([1452, 256, 64])

            # # 频率压缩 256 -> 88
            # if i == (self.fin_stage // 2):
            #     # torch.Size([401, 256, 64])
            #     x = self.fre_down_fc(x.transpose(1,2)).transpose(1,2).contiguous()
            #     # x = self.fre_down_pool(x.transpose(1,2)).transpose(1,2).contiguous()
            #     # x = self.fre_down_conv(x.transpose(1,0).unsqueeze(0)).squeeze(0).transpose(1,0).contiguous()
            if i == self.fin_stage // 2:
                # x = self.fre_down_pool_4(x.transpose(1,2)).transpose(1,2).contiguous()
                x = self.fre_down_fc_4(x.transpose(1,2)).transpose(1,2).contiguous()
            elif i == self.fin_stage // 4 * 3:
                # x = self.fre_down_pool_6(x.transpose(1,2)).transpose(1,2).contiguous()
                x = self.fre_down_fc_6(x.transpose(1,2)).transpose(1,2).contiguous()

        x = self.trans_norm(x)
        _, F, C = x.shape
        x = x.reshape(N, T, F, C).transpose(1,2).contiguous() 
        
        return x
    
