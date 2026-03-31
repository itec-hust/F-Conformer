import argparse
import os
import logging
import h5py
from matplotlib import pyplot as plt
import soundfile
import librosa
import audioread
import numpy as np
import pandas as pd
import csv
import datetime
import collections
import pickle
from mido import MidiFile

from piano_vad import (note_detection_with_onset_offset_regress, note_detection_without_frame, note_detection_without_offset, 
    pedal_detection_with_onset_offset_regress, onsets_frames_note_detection, onsets_frames_pedal_detection)
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def traverse_folder(folder):
    paths = []
    names = []
    
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith('.h5'):
                filepath = os.path.join(root, name)
                names.append(name)
                paths.append(filepath)
            
    return names, paths


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440

    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    log_name = '' + datetime.datetime.now().strftime('%y%m%d-%H%M%S') + '.log'
    # i1 = 0

    # while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
    #     i1 += 1
        
    # log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    log_path = os.path.join(log_dir, log_name)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    print('log saving at ' + log_path)
    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s\t%(filename)s:[line:%(lineno)d] %(levelname)s\t%(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging, log_path


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def read_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...], 
        'canonical_title': ['Sonata Op. 1', ...], 
        'split': ['train', ...], 
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...], 
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'split': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['split'].append(lines[n][0])
        meta_dict['midi_filename'].append(lines[n][1])
        meta_dict['audio_filename'].append(lines[n][2])
        meta_dict['duration'].append(float(lines[n][3]))

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    return meta_dict


def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


def read_maps_midi(midi_path):
    """Parse MIDI file of MAPS dataset. Not used anymore.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            '<meta message set_tempo tempo=439440 time=0>',
            'control_change channel=0 control=64 value=0 time=0',
            'control_change channel=0 control=64 value=0 time=7531',
            ...],
        'midi_event_time': [0., 0.53200309, 0.53200309, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 1

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[0]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num=88):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds # 10S
        self.frames_per_second = frames_per_second # 100
        self.begin_note = begin_note # 21
        self.classes_num = classes_num # 88
        self.max_piano_note = self.classes_num - 1 # 87

    def process(self, start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=0, onset_tolerance=0.05, offset_tolerance=0.05):
        """Process MIDI events of an audio segment to target for training, 
        将midi标签转为帧级标签, 注意pedal会影响note的时间范围
        与下面process0()相比简化了note帧级标签的计算, 输出结果不包含pedal的标签
        includes: 
        1. Parse MIDI events
        2. Prepare note targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 二值化
            'offset_roll': (frames_num, classes_num), 二值化
            'reg_onset_roll': (frames_num, classes_num), onset_tolerance时间范围内线性回归
            'reg_offset_roll': (frames_num, classes_num), offset_tolerance时间范围内线性回归
            'frame_roll': (frames_num, classes_num), 二值化
            'velocity_roll': (frames_num, classes_num), onset线性回归>0.5部分赋值
            'mask_roll':  (frames_num, classes_num), 被截断的音符置零

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index and the end index of a segment
        # """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break

        note_events = []
        """E.g. [{'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, ...]"""

        pedal_events = []
        """E.g. [{'onset_time': 696.46875, 'offset_time': 696.62604}, ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
            
        return self.prepare_targets(start_time, note_events, buffer_dict, note_shift, onset_tolerance, offset_tolerance)
        
    def prepare_targets(self, start_time, note_events, buffer_dict, note_shift=0, onset_tolerance=0.05, offset_tolerance=0.05):
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_tolerant_num = int(onset_tolerance * self.frames_per_second + 0.5)
        offset_tolerant_num = int(offset_tolerance * self.frames_per_second + 0.5)
        
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num)) # 存储每个时间帧与最近的起音事件的相对时间距离
        reg_offset_roll = np.ones((frames_num, self.classes_num)) # 存储每个时间帧与最近的离音事件的相对时间距离
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num)) # 用于标记哪些音符是跨越当前音频段边界的。0 表示该帧上的音符应该被忽略，1 表示应该被用于训练。
        """mask_roll is used for masking out cross segment notes"""

        # ------ 2. Get note targets ------
        # Process note events to target
        for i in range(len(note_events)):
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""
            note_event = note_events[i]
            
            # 21~108 => 0~87
            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int((note_event['onset_time'] - start_time) * self.frames_per_second + 0.5)
                fin_frame = int((note_event['offset_time'] - start_time) * self.frames_per_second + 0.5)

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    # Vector from the center of a frame to ground truth offset
                    for d in range(-offset_tolerant_num, offset_tolerant_num+1):
                        _idx = fin_frame + d 
                        if _idx >= 0 and _idx < frames_num: 
                            # 时间差值在offset_tolerance范围内, 即±50ms内计算相对距离
                            vector_offset = min(1, abs(note_event['offset_time'] - start_time - _idx / self.frames_per_second) / offset_tolerance)
                            reg_offset_roll[_idx, piano_note] = vector_offset

                    # Vector from the center of a frame to ground truth onset
                    for d in range(-onset_tolerant_num, onset_tolerant_num+1):
                        _idx = bgn_frame + d 
                        if _idx >= 0 and _idx < frames_num: 
                            # 时间差值在onset_tolerance范围内, 即±50ms内计算相对距离, 差值在50%即±25ms内时标注velocity
                            vector_onset = min(1, abs(note_event['onset_time'] - start_time - _idx / self.frames_per_second) / onset_tolerance)
                            reg_onset_roll[_idx, piano_note] = vector_onset
                            if vector_onset < 0.5:
                                velocity_roll[_idx, piano_note] = max(velocity_roll[_idx, piano_note], note_event['velocity'])

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            # reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            # reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])
            reg_onset_roll[:, k] = 1. - reg_onset_roll[:, k]
            reg_offset_roll[:, k] = 1. - reg_offset_roll[:, k]

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     

        target_dict = {
            'onset_roll': onset_roll, 
            'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 
            'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 
            'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 
            }

        return target_dict, note_events

    def process0(self, start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35}, 
            {'onset_time': 150.54, 'offset_time': 152.06}, 
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [{'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, ...]"""

        pedal_events = []
        """E.g. [{'onset_time': 696.46875, 'offset_time': 696.62604}, ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            # 21~108 => 0~87
            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll, 
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, note_events

    def process_note_events(self, start_time, midi_events, note_shift=0, onset_tolerance=0.05, offset_tolerance=0.05):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Prepare note targets

        Args:
          start_time: float, start time of a segment
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
        }

        note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index and the end index of a segment 
        # "E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"
        for bgn_idx in range(len(midi_events)):
            if midi_events[bgn_idx][0] > start_time:
                break
        for fin_idx in range(len(midi_events)):
            if midi_events[fin_idx][0] > start_time + self.segment_seconds:
                break

        note_events = []
        # "E.g. [
        #     {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        #     {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        #     ...]"

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            if midi_events[i][1] > start_time + self.segment_seconds: # offset超界
                buffer_dict[int(midi_events[i][2])] = {
                            'onset_time': midi_events[i][0]}
            note_events.append({
                            'onset_time': midi_events[i][0], 
                            'offset_time': min(midi_events[i][1], start_time + self.segment_seconds), 
                            'midi_note': int(midi_events[i][2])})
            """ 
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                "E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]
            # # Add unpaired onsets to events
                # for midi_note in buffer_dict.keys():
                #     note_events.append({
                #         'midi_note': midi_note, 
                #         'onset_time': buffer_dict[midi_note]['onset_time'], 
                #         'offset_time': start_time + self.segment_seconds, 
                #         'velocity': buffer_dict[midi_note]['velocity']})
            """
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        # velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""
            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            
            """There are 88 keys on a piano"""
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1  # onset与offset之间
                    offset_roll[fin_frame, piano_note] = 1
                    # velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     


        target_dict = {
            'onset_roll': onset_roll, 
            'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 
            'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 
            # 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 
            }

        return target_dict, note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0]  # input记录 帧与标注准确值之间的相对时间差(相邻最近onset帧的时间差, 单位s) 忽略时差在0.5s以上
        if len(locts) > 0:
            for t in range(0, locts[0]):  # output记录 所有帧与最近标注的相对时间差(input的拓展版)
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20 # 时间差绝对值在0.05s以内, 并归一化
        output = (1. - output)

        return output

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events
    

def write_events_to_midi(start_time, note_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    
    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1
    track1 = MidiTrack()
    
    # Message rolls of MIDI
    message_roll = []

    for note_event in note_events:
        # Onset
        message_roll.append({
            'time': note_event['onset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': note_event['velocity']})

        # Offset
        message_roll.append({
            'time': note_event['offset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': 0})


    # Sort MIDI messages by time
    message_roll.sort(key=lambda note_event: note_event['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message.keys():
                track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message.keys():
                track1.append(Message('control_change', channel=0, control=message['control_change'], value=message['value'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)

    midi_file.save(midi_path)


def plot_waveform_midi_targets(data_dict, start_time, note_events):
    """For debugging. Write out waveform, MIDI and plot targets for an 
    audio segment.

    Args:
      data_dict: {
        'waveform': (samples_num,),
        'onset_roll': (frames_num, classes_num), 
        'offset_roll': (frames_num, classes_num), 
        'reg_onset_roll': (frames_num, classes_num), 
        'reg_offset_roll': (frames_num, classes_num), 
        'frame_roll': (frames_num, classes_num), 
        'velocity_roll': (frames_num, classes_num), 
        'mask_roll':  (frames_num, classes_num), 
        'reg_pedal_onset_roll': (frames_num,),
        'reg_pedal_offset_roll': (frames_num,),
        'pedal_frame_roll': (frames_num,)}
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
    """
    import matplotlib.pyplot as plt

    create_folder('debug')
    audio_path = 'debug/debug.wav'
    midi_path = 'debug/debug.mid'
    fig_path = 'debug/debug.pdf'

    librosa.output.write_wav(audio_path, data_dict['waveform'], sr=config.sample_rate)
    write_events_to_midi(start_time, note_events, midi_path)
    x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=160, window='hann', center=True)
    x = np.abs(x) ** 2

    fig, axs = plt.subplots(11, 1, sharex=True, figsize=(30, 30))
    fontsize = 20
    axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[2].matshow(data_dict['offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[3].matshow(data_dict['reg_onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[4].matshow(data_dict['reg_offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[5].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[6].matshow(data_dict['velocity_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[7].matshow(data_dict['mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
    axs[8].matshow(data_dict['reg_pedal_onset_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
    axs[9].matshow(data_dict['reg_pedal_offset_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
    axs[10].matshow(data_dict['pedal_frame_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
    axs[0].set_title('Log spectrogram', fontsize=fontsize)
    axs[1].set_title('onset_roll', fontsize=fontsize)
    axs[2].set_title('offset_roll', fontsize=fontsize)
    axs[3].set_title('reg_onset_roll', fontsize=fontsize)
    axs[4].set_title('reg_offset_roll', fontsize=fontsize)
    axs[5].set_title('frame_roll', fontsize=fontsize)
    axs[6].set_title('velocity_roll', fontsize=fontsize)
    axs[7].set_title('mask_roll', fontsize=fontsize)
    axs[8].set_title('reg_pedal_onset_roll', fontsize=fontsize)
    axs[9].set_title('reg_pedal_offset_roll', fontsize=fontsize)
    axs[10].set_title('pedal_frame_roll', fontsize=fontsize)
    axs[10].set_xlabel('frames')
    axs[10].xaxis.set_label_position('bottom')
    axs[10].xaxis.set_ticks_position('bottom')
    plt.tight_layout(1, 1, 1)
    plt.savefig(fig_path)

    print('Write out to {}, {}, {}!'.format(audio_path, midi_path, fig_path))


class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold, 
        offset_threshold, frame_threshold, onset_tolerance=0.05, offset_tolerance=0.05, onset_tolerant_num=2, offset_tolerant_num=2):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale
        self.onset_tolerant_num = onset_tolerant_num#int(onset_tolerance * self.frames_per_second + 0.5)
        self.offset_tolerant_num = offset_tolerant_num#int(offset_tolerance * self.frames_per_second + 0.5)
        
        print(f'\n\nthreshold:\t[onset = {onset_threshold},\toffset = {offset_threshold},\t frame = {frame_threshold}]')
        print(f'tolerant_num:\t[onset = {self.onset_tolerant_num},\toffset = {self.offset_tolerant_num}]')
        
    def output_dict_to_midi_events(self, output_dict): # 帧级结果 => mid
        """Main function. Post process model outputs to MIDI events.

        Args:
            output_dict: {
                'reg_onset_output': (segment_frames, classes_num), 
                'reg_offset_output': (segment_frames, classes_num), 
                'frame_output': (segment_frames, classes_num), 
                'velocity_output': (segment_frames, classes_num), 
            }

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]
        """
        # # Post process piano note outputs to piano note and pedal events information
        est_on_off_note_vels = self.output_dict_to_note_arrays(output_dict) # 帧级结果 => mid数组
        # est_on_off_note_vels = self.output_dict_to_note_arrays_SONY(output_dict)
        # # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels) # 数组包装成dict NOTE: 'velocity': vels * 128
        return est_note_events
    
    def output_dict_to_note_arrays_SONY(self, output_dict, onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.1, mode_velocity='ignore_zero', mode_offset='shorter'): 
        """Args:
            output_dict: {
                'reg_onset_output': (segment_frames, classes_num), 
                'reg_offset_output': (segment_frames, classes_num), 
                'frame_output': (segment_frames, classes_num), 
                'velocity_output': (segment_frames, classes_num), 
            }
        Returns:
            est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
                offset_time, piano_note and velocity. E.g. [
                [39.74, 39.87, 27, 0.65], 
                [11.98, 12.11, 33, 0.69], 
                ...]
        """
        print("SONY post_processor", end='\r')
        onset_output = output_dict['reg_onset_output']
        offset_output = output_dict['reg_offset_output']
        frame_output = output_dict['frame_output']
        velocity_output = output_dict['velocity_output']
        
        notes = []
        hop_length = 1 / self.frames_per_second

        for j in range(self.classes_num):
            # find local maximum
            onset_detect = []
            for i in range(len(onset_output)):
                if onset_output[i][j] >= onset_threshold:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if onset_output[i][j] > onset_output[ii][j]:
                            left_flag = True
                            break
                        elif onset_output[i][j] < onset_output[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(onset_output)):
                        if onset_output[i][j] > onset_output[ii][j]:
                            right_flag = True
                            break
                        elif onset_output[i][j] < onset_output[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):  # 左右两边出现的第一个不相同的值都更低(寻找峰值)
                        if (i == 0) or (i == len(onset_output) - 1):
                            onset_time = i * hop_length
                        else:
                            if onset_output[i-1][j] == onset_output[i+1][j]:   # 位于首尾帧或者相邻帧相等, 则i直接对应onset_time
                                onset_time = i * hop_length
                            elif onset_output[i-1][j] > onset_output[i+1][j]:   # 相邻±1帧作线性回归拟合
                                onset_time = (i * hop_length - (hop_length * 0.5 * (onset_output[i-1][j] - onset_output[i+1][j]) / (onset_output[i][j] - onset_output[i+1][j])))
                            else:
                                onset_time = (i * hop_length + (hop_length * 0.5 * (onset_output[i+1][j] - onset_output[i-1][j]) / (onset_output[i][j] - onset_output[i-1][j])))
                        onset_detect.append({'loc': i, 'onset_time': onset_time})
            offset_detect = []
            for i in range(len(offset_output)):
                if offset_output[i][j] >= offset_threshold:
                    left_flag = True
                    for ii in range(i-1, -1, -1):
                        if offset_output[i][j] > offset_output[ii][j]:
                            left_flag = True
                            break
                        elif offset_output[i][j] < offset_output[ii][j]:
                            left_flag = False
                            break
                    right_flag = True
                    for ii in range(i+1, len(offset_output)):
                        if offset_output[i][j] > offset_output[ii][j]:
                            right_flag = True
                            break
                        elif offset_output[i][j] < offset_output[ii][j]:
                            right_flag = False
                            break
                    if (left_flag is True) and (right_flag is True):
                        if (i == 0) or (i == len(offset_output) - 1):
                            offset_time = i * hop_length
                        else:
                            if offset_output[i-1][j] == offset_output[i+1][j]:
                                offset_time = i * hop_length
                            elif offset_output[i-1][j] > offset_output[i+1][j]:
                                offset_time = (i * hop_length - (hop_length * 0.5 * (offset_output[i-1][j] - offset_output[i+1][j]) / (offset_output[i][j] - offset_output[i+1][j])))
                            else:
                                offset_time = (i * hop_length + (hop_length * 0.5 * (offset_output[i+1][j] - offset_output[i-1][j]) / (offset_output[i][j] - offset_output[i-1][j])))
                        offset_detect.append({'loc': i, 'offset_time': offset_time})

            time_next = 0.0
            time_offset = 0.0
            time_frame = 0.0
            for idx_on in range(len(onset_detect)):
                # onset
                loc_onset = onset_detect[idx_on]['loc']
                time_onset = onset_detect[idx_on]['onset_time']

                if idx_on + 1 < len(onset_detect):
                    loc_next = onset_detect[idx_on+1]['loc']
                    #time_next = loc_next * hop_sec
                    time_next = onset_detect[idx_on+1]['onset_time']
                else:
                    loc_next = len(frame_output)
                    time_next = (loc_next-1) * hop_length

                # offset
                loc_offset = loc_onset+1
                flag_offset = False
                #time_offset = 0###
                # 找到第一个大于loc_onset的loc_offset
                for idx_off in range(len(offset_detect)):
                    if loc_onset < offset_detect[idx_off]['loc']: 
                        loc_offset = offset_detect[idx_off]['loc']
                        time_offset = offset_detect[idx_off]['offset_time']
                        flag_offset = True
                        break
                # loc_offset应当小于等于loc_next
                if loc_offset > loc_next:
                    loc_offset = loc_next
                    time_offset = time_next

                # offset by frame
                # (1frame longer)
                loc_frame = loc_onset+1
                flag_frame = False
                #time_frame = 0###
                # 找到第一个无效frame(offset的第二个参考值)
                for ii_frame in range(loc_onset+1, loc_next):
                    if frame_output[ii_frame][j] < frame_threshold:
                        loc_frame = ii_frame
                        flag_frame = True
                        time_frame = loc_frame * hop_length
                        break
                '''
                # (right algorighm)
                loc_frame = loc_onset
                flag_frame = False
                for ii_frame in range(loc_onset+1, loc_next+1):
                    if a_frame[ii_frame][j] < thred_frame:
                        loc_frame = ii_frame-1
                        flag_frame = True
                        time_frame = loc_frame * hop_sec
                        break
                '''
                pitch_value = j + self.begin_note
                velocity_value = velocity_output[loc_onset][j]

                # offset和frame都无效采用下一个onset作offset
                # 一方无效直接采用另一方, 
                # 都有效时 默认选用更小值
                # 默认忽略velocity为0的结果
                if (flag_offset is False) and (flag_frame is False):
                    offset_value = float(time_next)
                elif (flag_offset is True) and (flag_frame is False):
                    offset_value = float(time_offset)
                elif (flag_offset is False) and (flag_frame is True):
                    offset_value = float(time_frame)
                else:
                    if mode_offset == 'offset':
                        ## (a) offset
                        offset_value = float(time_offset)
                    elif mode_offset == 'longer':
                        ## (b) longer
                        if loc_offset >= loc_frame:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_frame)
                    else:
                        ## (c) shorter
                        if loc_offset <= loc_frame:
                            offset_value = float(time_offset)
                        else:
                            offset_value = float(time_frame)
                if mode_velocity != 'ignore_zero':
                    # notes.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})
                    notes.append([time_onset, offset_value, pitch_value, velocity_value])
                else:
                    if velocity_value > 0:
                        # notes.append({'pitch': pitch_value, 'onset': float(time_onset), 'offset': offset_value, 'velocity': velocity_value})
                        notes.append([time_onset, offset_value, pitch_value, velocity_value])

                if (len(notes) > 1) and \
                   (notes[len(notes)-1][2] == notes[len(notes)-2][2]) and \
                   (notes[len(notes)-1][0] < notes[len(notes)-2][1]):
                    notes[len(notes)-2][1] = notes[len(notes)-1][0]

        notes = sorted(sorted(notes, key=lambda x: x[2]), key=lambda x: x[0])
        return np.array(notes)

    def output_dict_to_note_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output 峰值提取二值化onset的帧级下标 线性回归计算对应偏移量
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=self.onset_threshold, neighbour=self.onset_tolerant_num)

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output  

        # Calculate binarized offset output from regression output 峰值提取二值化offset的帧级下标 线性回归计算对应偏移量
        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=self.offset_threshold, neighbour=self.offset_tolerant_num)

        output_dict['offset_output'] = offset_output  # Values are 0 or 1
        output_dict['offset_shift_output'] = offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict) 

        return est_on_off_note_vels

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape
        
        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour): # 大于阈值且处于峰值
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]: # 线性回归计算偏移值
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2 
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]: # 左边非递增
                monotonic = False
            if x[n + i] < x[n + i + 1]: # 右边非递减
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]

        counts = [0,0,0,0,0]
        for piano_note in range(classes_num):
            """Detect piano notes"""
            # est_tuples_per_note = note_detection_without_frame(
            # est_tuples_per_note = note_detection_without_offset(
            est_tuples_per_note, count = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
                frame_threshold=self.frame_threshold) #, onset_tolerant_num=self.onset_tolerant_num)
            
            counts = [a + b for a, b in zip(counts, count)]
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        print(f'连音onset{counts[0]}, 连音offset{counts[4]}, 单音offset{counts[1]},  单音frame{counts[2]} 12s未检{counts[3]}')
        
        """[onset, offset, onset_shift, offset_shift, normalized_velocity]"""
        est_tuples = np.array(est_tuples)   # (notes, 5)

        est_midi_notes = np.array(est_midi_notes) # (notes,)

        if est_tuples.size == 0 or est_tuples.ndim < 2:
            # 返回一个形状为 (0, 3) 的空数组，确保后续逻辑不会崩溃
            return np.zeros((0, 3), dtype=np.float32)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        
        
        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes), axis=-1)
        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """
        frames_num = output_dict['pedal_frame_output'].shape[0]
        
        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict['pedal_frame_output'][:, 0], 
            offset_output=output_dict['pedal_offset_output'][:, 0], 
            offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0], 
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""
        
        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g., 
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0], 
                'offset_time': pedal_on_offs[i, 1]})
        
        return pedal_events


class OnsetsFramesPostProcessor(object):
    def __init__(self, frames_per_second, classes_num):
        """Postprocess the Googl's onsets and frames system output. Only used
        for comparison.

        Args:
          frames_per_second: int
          classes_num: int
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale
        
        self.frame_threshold = 0.5
        self.onset_threshold = 0.1
        self.offset_threshold = 0.3

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num), 
            'reg_pedal_onset_output': (segment_frames, 1), 
            'reg_pedal_offset_output': (segment_frames, 1), 
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96}, 
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""
        
        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
            and offset_time. E.g. [
             [0.17, 0.96], 
             [1.17, 2.65], 
             ...]
        """

        # Sharp onsets and offsets
        output_dict = self.sharp_output_dict(
            output_dict, onset_threshold=self.onset_threshold, 
            offset_threshold=self.offset_threshold)

        # Post process output_dict to piano notes
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict, 
            frame_threshold=self.frame_threshold)

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
 
        else:
            est_pedal_on_offs = None    

        return est_on_off_note_vels, est_pedal_on_offs

    def sharp_output_dict(self, output_dict, onset_threshold, offset_threshold):
        """Sharp onsets and offsets. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]
        [0., 0., 1., 0., 0., 0.]

        Args:
          output_dict: {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            ...}
          onset_threshold: float
          offset_threshold: float

        Returns:
          output_dict: {
            'onset_output': (frames_num, classes_num), 
            'offset_output': (frames_num, classes_num)}
        """
        if 'reg_onset_output' in output_dict.keys():
            output_dict['onset_output'] = self.sharp_output(
                output_dict['reg_onset_output'], 
                threshold=onset_threshold)

        if 'reg_offset_output' in output_dict.keys():
            output_dict['offset_output'] = self.sharp_output(
                output_dict['reg_offset_output'], 
                threshold=offset_threshold)

        return output_dict

    def sharp_output(self, input, threshold=0.3):
        """Used for sharping onset or offset. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]

        Args:
          input: (frames_num, classes_num)

        Returns:
          output: (frames_num, classes_num)
        """
        (frames_num, classes_num) = input.shape
        output = np.zeros_like(input)

        for piano_note in range(classes_num):
            loct = None
            for i in range(1, frames_num - 1):
                if input[i, piano_note] > threshold and input[i, piano_note] > input[i - 1, piano_note] and input[i, piano_note] > input[i + 1, piano_note]:
                    loct = i
                else:
                    if loct is not None:
                        output[loct, piano_note] = 1
                        loct = None

        return output

    def output_dict_to_detected_notes(self, output_dict, frame_threshold):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """

        est_tuples = []
        est_midi_notes = []

        for piano_note in range(self.classes_num):
            
            est_tuples_per_note = onsets_frames_note_detection(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                threshold=frame_threshold)

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 3)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)
        
        if len(est_midi_notes) == 0:
            return []
        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            velocities = est_tuples[:, 2]
        
            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            """(notes, 3), the three columns are onset_times, offset_times and velocity."""

            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

            return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """

        frames_num = output_dict['pedal_frame_output'].shape[0]
        
        est_tuples = onsets_frames_pedal_detection(
            frame_output=output_dict['pedal_frame_output'][:, 0], 
            offset_output=output_dict['reg_pedal_offset_output'][:, 0], 
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""
        
        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g., 
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0], 
                'offset_time': pedal_on_offs[i, 1]})
        
        return pedal_events


class StatisticsContainer(object):
    def __init__(self, statistics_path, load_statistics_path=None):
        """Contain statistics of different training iterations.
        """
        self.statistics_path = statistics_path
        self.load_statistics_path = load_statistics_path

        self.statistics_dict = {'train': [], 'validation': [], 'test': []}

        logging.info('    Statistics path: {}'.format(self.statistics_path))

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        
    def dump_file(self, pth_path):
        pickle.dump(self.statistics_dict, open(pth_path.replace(".pth", ".pkl"), 'wb'))
        
    def load_state_dict(self, resume_iteration):
        if os.path.exists(self.load_statistics_path):
            self.statistics_dict = pickle.load(open(self.load_statistics_path, 'rb'))

            resume_statistics_dict = {'train': [], 'validation': [], 'test': []}
            
            for key in self.statistics_dict.keys():
                for statistics in self.statistics_dict[key]:
                    if statistics['iteration'] <= resume_iteration:
                        resume_statistics_dict[key].append(statistics)
                    
            self.statistics_dict = resume_statistics_dict


def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None,
    dtype=np.float32, res_type='kaiser_best', 
    backends=[audioread.ffdec.FFmpegAudioFile]):
    """Load audio. Copied from librosa.core.load() except that ffmpeg backend is 
    always used in this function."""

    y = []
    with audioread.audio_open(os.path.realpath(path), backends=backends) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = librosa.util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = librosa.to_mono(y)

        if sr is not None:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)

def save_pianoroll(notes, img_path=None, min_frames=None, max_frames=None, fps=50, zoom=4, to_print=False):
    from PIL import Image

    assert notes.shape[1] in {3,4}, f'notes.shape: {notes.shape} is not supported'
    
    if img_path == None:
        img_path = f"piano_roll_{datetime.datetime.now().strftime('%m%d-%H%M')}.png"
    
    max_time = notes[-1,1]
    if min_frames == None:
        min_frames = 0 
    if max_frames == None:
        max_frames = fps * max_time

    assert min_frames < max_frames, print(f'min_frames({min_frames}) >= max_frames({max_frames})')
    
    image = np.zeros((88, max_frames-min_frames))
    for note in notes: # [onset, offset, pitch, velocity]
        onset = int(0.5 + note[0]*fps)
        offset = int(0.5 + note[1]*fps) - min_frames
        if onset > max_frames:
            break
        if onset < min_frames:
            continue
        onset -= min_frames
        pitch = int(0.5 + note[2] - 21)
        velocity = note[3] if note.shape[0] == 4 else 1
        image[pitch, max(onset,0) : min(offset,max_frames)] = velocity
        if onset > 335 and onset < 350:
            print(f'[{pitch+21}, {(onset+min_frames)/fps}s : {(offset+min_frames)/fps}s]')
    print()
    plt.imshow(image, cmap='viridis', interpolation='nearest', aspect='auto')
    plt.axis('on')
    plt.gca().invert_yaxis()
    # plt.yticks(np.arange(0, 89, 88),['A0','C8'])
    plt.yticks(np.arange(0, 89, 24),['A0','A2','A4','A6'])
    seconds = 1
    plt.xticks(np.arange(0, max_frames-min_frames, fps*seconds),range(0, (max_frames-min_frames) // fps * seconds, seconds))
    plt.gca().set_aspect(1, adjustable='box', anchor='SW')
    plt.yticks(fontproperties = 'Times New Roman', size = 9)
    plt.xticks(fontproperties = 'Times New Roman', size = 9)    
    if to_print:
        plt.savefig(img_path, dpi=426, bbox_inches='tight') # dpi 213
        print(f"Piano roll saved at: {img_path}")
    return image

def get_delta(label_path, result_path, workspace='workspaces', dataset='maestro', segment_index=0, segment_frames=250, to_print=False): 
    img_dir = os.path.join(os.path.dirname(result_path), 'img')
    os.makedirs(img_dir, exist_ok=True)
    filename = os.path.basename(result_path)
    if label_path == None:
        label_dir = os.path.join(workspace, 'hdf5s', dataset + '_labels')
        label_path = os.path.join(label_dir, filename.replace('_pred','').replace('.pred','').replace('.wav\'',''))
    
    from comment_notes import get_notes
    label = get_notes(label_path)
    result = get_notes(result_path)
    
    min_frames = int(segment_frames * segment_index)
    max_frames = min_frames + segment_frames
    # print(f'min_frames({min_frames}) : max_frames({max_frames})')
    label_image = np.where(save_pianoroll(label, min_frames=min_frames, max_frames=max_frames, img_path=os.path.join(img_dir, filename + '_label.png'), to_print=to_print) != 0, 1, 0)
    result_image = np.where(save_pianoroll(result, min_frames=min_frames, max_frames=max_frames, img_path=os.path.join(img_dir, filename + '_result.png'), to_print=to_print) != 0, 1, 0)
    delta = np.sum(np.abs(label_image - result_image), axis=0)
    delta = np.sum(delta.reshape(-1, 50), axis=1)
    np.set_printoptions(threshold=np.inf) 
    if to_print:
        print(f'sum: {np.sum(delta)}\t=\t{delta}')
    
    return int(np.sum(delta)), delta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--result_path', '-r', type=str)
    parser.add_argument('--onset-win', type=float, default=0.05)
    parser.add_argument('--offset-win', type=float, default=0.05)
    parser.add_argument('--workspace', type=str, default='workspaces')
    parser.add_argument('--dataset', type=str, default='maestro')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    label_path = args.label_path
    result_path = args.result_path
    workspace = args.workspace
    dataset = args.dataset
    bgn = 27
    for i in range(bgn, bgn + 1):
        sums = []
        deltas = []
        for result in result_path.split(','):
            s, delta = get_delta(label_path, result, workspace, dataset, segment_index=i, segment_frames=500, 
                                   to_print=True)
            sums.append(s)
            deltas.append(delta)
        # if sums[0] < 50:
        #     sums = [a - b for a, b in zip(sums[1:], sums[:-1])]
        #     if sums[0] > 0 and sums[1] > 0:
        #         print(f'{i}:\t{sums}', end='\t')
        #         for j in range(len(deltas[0])):
        #             if deltas[0][j] <= deltas[1][j] and deltas[1][j] <= deltas[2][j]:
        #                 print(j, end='\t')
        #         print()
        
        sums = [a - b for a, b in zip(sums[1:], sums[:-1])]
        if all(x >= 0 for x in sums) and sum(sums) > 0:
            print(f'{i}:\t{sums}')

# 17:     [10, 3] 0       1       2       3
# 35:     [5, 5]
# 36:     [1, 1]  0       1       2       3       4
# 59:     [4, 7]  0       1       2
# 61:     [1, 21] 0       1       2       4
# 63:     [1, 96] 0       1       2       3
# 65:     [5, 54] 0       1       2       3       4
# 70:     [2, 9]  0       1       2       3       4
# 75:     [4, 14] 0       4