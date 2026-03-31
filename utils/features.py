import numpy as np
import argparse
import csv
import os
import time
import logging
import h5py
import librosa
import logging

import torch
import torchaudio
import nnAudio.features

from utilities import (create_folder, float32_to_int16, create_logging, 
    get_filename, read_metadata, read_midi, read_maps_midi)
import config


def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro_v3_cqt')

    # logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    # create_logging(logs_dir, filemode='w')
    # logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))
    feature_time = time.time()

    # mel_process = torchaudio.transforms.MelSpectrogram(n_mels=config.mel_config['n_mels'], sample_rate=config.sample_rate, n_fft=config.mel_config['n_fft'], hop_length=config.hop_length, win_length=config.mel_config['n_fft'], center=True, pad_mode='constant', norm='slaney')
    spec_process = nnAudio.features.cqt.CQT(sr=config.sample_rate, hop_length=config.hop_length, fmin=config.cqt_config['fmin'], 
                            n_bins=config.cqt_config['n_bins'], bins_per_octave=config.cqt_config['bins_per_octave'])
    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]))
        if os.path.exists(packed_hdf5_path):
            print(f'{packed_hdf5_path} already exists!')
            continue
        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        pre_spec = spec_process(torch.tensor(audio))
        # mel_log = (torch.log(mel + 1e-08)).T
        logging.info('{} {} {} {} {}'.format(n, meta_dict['midi_filename'][n], meta_dict['duration'][n], audio.shape, pre_spec.shape))
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            # hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            hf.create_dataset(name='pre_spec', data=pre_spec, dtype=np.float32)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def pack_omap_dataset_to_hdf5(args):
    """
    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    label_dir = args.label_dir
    
    sample_rate = config.sample_rate
    # pianos = ['ENSTDkCl', 'ENSTDkAm']
    splits = ['train', 'valid', 'test']

    # Paths
    # waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'omap')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a hdf5 file
    for split in splits:
        sub_dir = os.path.join(dataset_dir, split, 'wav')

        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) 
            if os.path.splitext(name)[-1] == '.wav']
        
        for audio_name in audio_names:
            print('{} {}'.format(count, audio_name))
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            # midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))
            # midi_path = '{}.mid'.format(os.path.join(label_dir, audio_name))
            midi_path = '{}.txt'.format(os.path.join(label_dir, audio_name))

            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            duration = len(audio) / sample_rate
            print(audio_path)
            print(duration)
            # midi_dict = read_maps_midi(midi_path)
            midi_text = np.loadtxt(midi_path, dtype=np.float32)
            midi_dict = midi_text
            # midi_dict = []
            # for note in midi_text:
            #     midi_dict.append({
            #         'midi_note': int(note[2]), 
            #         'onset_time': note[0], 
            #         'offset_time': note[1], 
            #         'velocity': int(note[3])})
            
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(audio_name))
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, 'w') as hf:
                # hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('split', data=split.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('duration', data=duration, dtype=np.float32)
                # hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event', data=midi_dict, dtype=np.float32)
                # hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            
            count += 1

    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))

def pack_maps_dataset_to_hdf5(args):
    """MAPS is a piano dataset only used for evaluating our piano transcription
    system (optional). Ref:

    [1] Emiya, Valentin. "MAPS Database A piano database for multipitch 
    estimation and automatic transcription of music. 2016

    Load & resample MAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    pianos = ['ENSTDkCl', 'ENSTDkAm']
    pianos = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps_train')

    # logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    # create_logging(logs_dir, filemode='w')
    # logging.info(args)

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a hdf5 file
    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')

        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) 
            if os.path.splitext(name)[-1] == '.mid']
        
        for audio_name in audio_names:
            print('{} {}'.format(count, audio_name))
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))

            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            midi_dict = read_maps_midi(midi_path)
            
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(audio_name))
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name).encode(), dtype='S100')
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            
            count += 1

    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_maestro = subparsers.add_parser('pack_maestro_dataset_to_hdf5')
    parser_pack_maestro.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maestro.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_maps = subparsers.add_parser('pack_maps_dataset_to_hdf5')
    parser_pack_maps.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maps.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    
    parser_pack_omap = subparsers.add_parser('pack_omap_dataset_to_hdf5')
    parser_pack_omap.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_omap.add_argument('--label_dir', type=str, required=True, help='Directory of label.')
    parser_pack_omap.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_maestro_dataset_to_hdf5':
        pack_maestro_dataset_to_hdf5(args)
        
    elif args.mode == 'pack_maps_dataset_to_hdf5':
        pack_maps_dataset_to_hdf5(args)

    elif args.mode == 'pack_omap_dataset_to_hdf5':
        pack_omap_dataset_to_hdf5(args)
            
    else:
        raise Exception('Incorrect arguments!')