import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import argparse
import time
import datetime
import logging

import torch
import torch.optim as optim
import torch.utils.data
import multiprocessing

from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor) 
from data_generator import MaestroDataset, OMAP_Dataset, Augmentor, Sampler, TestSampler, collate_fn
from models_highTrans import Regress_CRNN, Regress_CRNN_Slim, Regress_Conformer, Regress_HAT
from models_hpp import Regress_HPP, Regress_HPP_SP
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator
import config
import atexit

from functools import reduce

def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = torch.nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    from ptflops import get_model_complexity_info
    input = (128000,)
    flops, params = get_model_complexity_info(model, input, print_per_layer_stat=True)
    string += f'\n{input} => FLOPs = ' + flops + '\tParams = ' + params
    if file is not None:
        # if isinstance(file, str):
        #     file = open(file, 'w')
        # print(string, file=file)
        # file.flush()
        logging.info(string)
    # exit()
    return string, count


def del_log(log_path, text='提前终止, 清除日志'):
    print('{}: {}'.format(text, log_path))
    os.remove(log_path)

def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    resume_checkpoint = args.resume_checkpoint
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename
    pre_spec = args.pre_spec

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = multiprocessing.cpu_count() // 4 * 3
    norm = args.norm

    # Loss function
    loss_func = get_loss_func(loss_type)

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        '{}'.format(args.dataset),
        'batch_size={}'.format(batch_size),
        '{}'.format(segment_seconds))
    create_folder(logs_dir)
    _, log_path = create_logging(logs_dir, filemode='w')
    atexit.register(del_log, log_path)

    # Model
    if model_type == 'Regress': # default high-resolution
        Model = eval("Regress_CRNN")
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num, norm=norm)
    elif model_type == 'Regress_CRNN_Slim': 
        Model = eval('Regress_CRNN_Slim')
        model_type = 'Four_blocks'
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num, model_type=model_type)
    elif model_type in ["Conv_trans", "Three_blocks", "Four_blocks", "ConvBlock", "Conv_quarter_trans", "ConvBlock_Transblock"]: 
        Model = eval("Regress_CRNN")
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num, 
                model_type=model_type, norm=norm)
    elif model_type in ["Regress_HPP", "Regress_HPP_GRU", "Regress_HPP_SP"]:
        if  model_type == "Regress_HPP_SP":
            Model = eval(model_type)
        else:
            Model = eval("Regress_HPP")
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num, 
                model_type=model_type)
    elif model_type == "Regress_onset_offset_frame_velocity_CRNN" :
        Model = eval(model_type)
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num)
    elif "Conformer" in model_type:
        Model = eval("Regress_Conformer")    
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num, 
                model_type=model_type)
    elif "HAT" in model_type:
        Model = eval("Regress_HAT")    
        model = Model(frames_per_second=frames_per_second, classes_num=classes_num, 
                model_type=model_type, pre_spec=pre_spec)
    else:
        assert False, f'Wrong model_type: {model_type}'
        
    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', args.dataset)

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        '{}'.format(args.dataset),
        'batch_size={}'.format(batch_size),
        '{}'.format(segment_seconds),
        datetime.datetime.now().strftime('%m%d_%H%M'))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(checkpoints_dir, 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))


    logging.info(args)
    logging.info(f'batch_size: {args.batch_size:<10}, sr: {sample_rate:<10}, segment_seconds: {segment_seconds:<10}, hop_seconds: {hop_seconds:<10},\tsegment_samples: {segment_samples:<10}, fps: {frames_per_second:<10}, classes_num: {classes_num:<10}, num_workers: {num_workers:<10}, norm: {norm:<10}')
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
        model.to(device)
    else:
    
        logging.info('Using CPU.')
        device = 'cpu'
    summary(model, file='log')
    
    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')
    
    # Dataset
    logging.info(f"loading {args.dataset}")
    if 'maestro' in args.dataset:
        train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
            segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
            max_note_shift=max_note_shift, augmentor=augmentor, pre_spec=pre_spec)

        evaluate_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
            segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
            max_note_shift=0)
    elif args.dataset == 'omap':
        train_dataset = OMAP_Dataset(hdf5s_dir=hdf5s_dir, 
            segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
            max_note_shift=max_note_shift, augmentor=augmentor, pre_spec=pre_spec)
        
        evaluate_dataset = OMAP_Dataset(hdf5s_dir=hdf5s_dir, 
            segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
            max_note_shift=0)
    elif args.dataset == 'maps':
        print('dataset not supported')
        exit(1)
    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data, dataset=args.dataset)

    # Sampler for evaluation
    evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data, dataset=args.dataset)

    evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='valid' if args.dataset == 'omap' else 'validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data, dataset=args.dataset)

    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data, dataset=args.dataset)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Resume training
    if resume_iteration > 0 or resume_checkpoint:
        resume_checkpoint_path = resume_checkpoint 
        logging.info(f"load {resume_checkpoint_path}")
        statistics_container = StatisticsContainer(statistics_path, resume_checkpoint_path.replace(".pth", ".pkl"))
        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        
        checkpoint = torch.load(resume_checkpoint_path)
        model_dict = model.state_dict()
        checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if (k in model_dict) and (v.shape == model_dict[k].shape)}
        model.load_state_dict(checkpoint['model'], strict=False)
        train_sampler.load_state_dict(checkpoint['sampler'])
        if checkpoint.get('optimizer'):
            try:
                optimizer.load_state_dict(checkpoint['optimizer']) 
                logging.warn('loaded optimizer state_dict.')
            except Exception as e:
                logging.warn('load optimizer state_dict fail:')
                logging.warn(e.with_traceback)
        resume_iteration = checkpoint['iteration']
        iteration = resume_iteration
        statistics_container.load_state_dict(resume_iteration)
    else:
        iteration = 0

    if 'cuda' in str(device):
        model.to(device)
    
    
    # Parallel
    logging.info('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    print(log_path)

    train_bgn_time = time.time()

    best_checkpoint = {
        'iteration': 0,
        'sum': 0
    }
    iteration_checkpoint = 1250
    for batch_data_dict in train_loader:     

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        model.train()
        batch_output_dict = model(batch_data_dict['waveform'])
        loss = loss_func(batch_output_dict, batch_data_dict)

        # Backward
        loss.backward()
        print('Iteration: {}  loss: {:.3f}    '.format(iteration - resume_iteration, loss), end='')
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Stop learning
        if iteration - resume_iteration >= early_stop:
            print(f'{iteration} >= early_stop({early_stop})')
            break

        iteration += 1
        # Evaluation 
        # try:
        if iteration % iteration_checkpoint == 0 and iteration > 0:
            logging.info('------------------------------------------------------------')
            logging.info('Iteration: {}\t  loss: {}\tlr: {}'.format(iteration, loss, learning_rate))

            train_fin_time = time.time()

            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            validate_statistics = evaluator.evaluate(validate_loader) 
            test_statistics = evaluator.evaluate(test_loader)

            logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
            logging.info('    Valid statistics: {}'.format(validate_statistics))
            logging.info('    Test statistics: {}'.format(test_statistics))

            statistics_container.append(iteration, evaluate_train_statistics, data_type='train')
            statistics_container.append(iteration, validate_statistics, data_type='validation')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))
            if validate_time > train_time * 0.05:
                iteration_checkpoint = iteration_checkpoint * 2
                logging.warn(f'Double iteration_checkpoint to {iteration_checkpoint}')
            train_bgn_time = time.time()
        
            # Save model              
            if iteration >= 100000 and iteration - resume_iteration >= 50000: 
                atexit.unregister(del_log)
                checkpoint = {
                    'iteration': iteration, 
                    'model': model.module.state_dict(), 
                    'sampler': train_sampler.state_dict(),
                    'optimizer': optimizer.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                    
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to \n{}'.format(checkpoint_path))
                
                if test_statistics['sum'] >= best_checkpoint['sum']:
                    best_checkpoint['sum'] = test_statistics['sum']
                    best_checkpoint['iteration'] = iteration
                    torch.save(checkpoint, os.path.join(checkpoints_dir, 'best.pth'.format(iteration)))
                else:
                    best_checkpoint['sum'] = best_checkpoint['sum'] - 0.0001
                    
                if best_checkpoint['iteration'] > 0:
                    logging.info(f"best_checkpoint: {best_checkpoint['iteration']}")
                print(log_path)
        if iteration % reduce_iteration == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
                learning_rate = param_group['lr']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, default=0, required=False)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--resume_checkpoint', type=str, required=False)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--pre_spec', action='store_true', default=False)
    parser_train.add_argument('--dataset', type=str, default='omap', choices=['maestro', 'maestro_slim', 'maestro_v3_mel', 'maestro_v3_cqt', 'maps', 'omap'])
    parser_train.add_argument('--norm', type=str, default='BatchNorm', choices=['BatchNorm', 'InstanceNorm'])
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')