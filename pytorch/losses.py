import torch
import torch.nn.functional as F
import logging
import config

def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    if torch.sum(mask) == 0:
        # logging.info(f'WARING: torch.sum(mask) == 0, shape: {mask.shape}')
        print(f'WARING: torch.sum(mask) == 0, shape: {mask.shape}')
        return torch.sum(matrix)/torch.sum(matrix) if torch.sum(matrix) != 0 else torch.sum(mask)
    return torch.sum(matrix * mask) / torch.sum(mask)

def regress_nn_bce(output_dict, target_dict):
    nn_bce = torch.nn.BCELoss()
    onset_loss = nn_bce(output_dict['reg_onset_output'].contiguous().view(-1), 
                     target_dict['reg_onset_roll'].contiguous().view(-1))
    offset_loss = nn_bce(output_dict['reg_offset_output'].contiguous().view(-1), 
                      target_dict['reg_offset_roll'].contiguous().view(-1))
    frame_loss = nn_bce(output_dict['frame_output'].contiguous().view(-1), 
                     target_dict['frame_roll'].contiguous().view(-1))
    if len(output_dict['velocity_output'].shape) == len(target_dict['velocity_roll'].shape): 
        velocity_loss = nn_bce(output_dict['velocity_output'].contiguous().view(-1), 
                            target_dict['velocity_roll'].contiguous().view(-1) / config.velocity_scale)
    else:
        velocity_loss_func = torch.nn.CrossEntropyLoss()
        velocity_loss = velocity_loss_func(output_dict['velocity_output'].contiguous().view(-1, config.velocity_scale), 
                                           target_dict['velocity_roll'].long().contiguous())
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    print('[nn_bce]  onset: {:.3f} | offset: {:.3f} | frame: {:.3f} | velocity: {:.3f}'.format(onset_loss, offset_loss, frame_loss, velocity_loss), end='\r')
    return total_loss

############ High-resolution regression loss ############
def regress_onset_offset_frame_velocity_bce(output_dict, target_dict, last_loss=torch.ones(1,4)):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'].contiguous().view(-1), 
                     target_dict['reg_onset_roll'].contiguous().view(-1), 
                     target_dict['mask_roll'].contiguous().view(-1))
    offset_loss = bce(output_dict['reg_offset_output'].contiguous().view(-1), 
                      target_dict['reg_offset_roll'].contiguous().view(-1), 
                      target_dict['mask_roll'].contiguous().view(-1))
    frame_loss = bce(output_dict['frame_output'].contiguous().view(-1), 
                     target_dict['frame_roll'].contiguous().view(-1), 
                     target_dict['mask_roll'].contiguous().view(-1))

    if len(output_dict['velocity_output'].shape) == len(target_dict['velocity_roll'].shape): 
        # velocity_loss = bce(output_dict['velocity_output'].contiguous().view(-1), 
        #                     target_dict['velocity_roll'].contiguous().view(-1) / config.velocity_scale, 
        #                     torch.where(target_dict['reg_onset_roll'] > 0, 1, 0).view(-1))
        #                     # torch.where(target_dict['reg_onset_roll'] > 0.5, 1, 0).view(-1))
        #                     # target_dict['onset_roll'].contiguous().view(-1))
        velocity_loss = bce(output_dict['velocity_output'].contiguous().view(-1), target_dict['velocity_roll'].contiguous().view(-1) / config.velocity_scale, target_dict['onset_roll'].contiguous().view(-1))

    else:
        velocity_loss_func = torch.nn.CrossEntropyLoss()
        velocity_loss = velocity_loss_func(output_dict['velocity_output'].contiguous().view(-1, config.velocity_scale), 
                                           target_dict['velocity_roll'].long().contiguous().view(-1))
        
    
    print('[bce]  onset: {:.3f} | offset: {:.3f} | frame: {:.3f} | velocity: {:.3f}'.format(onset_loss, offset_loss, frame_loss, velocity_loss), end='\r')
    if torch.isnan(onset_loss):
        logging.info('isnan')
        print(output_dict['reg_onset_output'])
        print(target_dict['reg_onset_roll'])
        print(target_dict['mask_roll'])
        onset_loss = last_loss[0]
    elif torch.isnan(offset_loss):
        logging.info('isnan')
        print(output_dict['reg_offset_output'])
        print(target_dict['reg_offset_roll'])
        print(target_dict['mask_roll'])
        offset_loss = last_loss[1]
    elif torch.isnan(frame_loss):
        logging.info('isnan')
        print(output_dict['frame_output'])
        print(target_dict['frame_roll'])
        print(target_dict['mask_roll'])
        frame_loss = last_loss[2]
    elif torch.isnan(velocity_loss):
        logging.info('isnan')
        print(output_dict['velocity_output'])
        print(target_dict['velocity_roll'])
        print(target_dict['onset_roll'])
        velocity_loss = last_loss[3]
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def regress_pedal_bce(output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['reg_pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['reg_pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss

############ Google's onsets and frames system loss ############
def google_onset_offset_frame_velocity_bce(output_dict, target_dict):
    """Google's onsets and frames system piano note loss. Only used for comparison.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / config.velocity_scale, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def google_pedal_bce(output_dict, target_dict):
    """Google's onsets and frames system piano pedal loss. Only used for comparison.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce

    elif loss_type == 'regress_pedal_bce':
        return regress_pedal_bce

    elif loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    elif loss_type == 'google_pedal_bce':
        return google_pedal_bce
    
    elif loss_type == 'regress_nn_bce':
        return regress_nn_bce

    else:
        raise Exception('Incorrect loss_type!')