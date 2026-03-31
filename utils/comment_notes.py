import argparse
import os
import pickle
import librosa
import numpy as np
import mir_eval
import time
from sklearn import metrics
from tqdm import tqdm
from scipy.stats import hmean
import sys
import warnings
import logging
import datetime
import config
from utilities import RegressionPostProcessor
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt

eps = sys.float_info.epsilon
mir_eval.multipitch.MAX_TIME = 300000.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--onset-win', type=float, default=0.05)
    parser.add_argument('--offset-win', type=float, default=0.05)
    parser.add_argument('--with_offset', action='store_true', default=True)
    parser.add_argument('--with_velocity', action='store_true', default=True)
    return parser.parse_args()

def get_notes(filepath):
    notes = []
    filepath = filepath.replace('_slim_','_')
    with open(filepath) as f:
        for line in f:
            if "OnsetTime" in line:
                continue
            #print(line)
            #有四列，onset\offset\pitch\velocity
            onset, offset, pitch, velocity = line.strip().split()[:4]
            notes.append([float(onset), float(offset), float(pitch),float(velocity)])
            # onset, offset, pitch = line.strip().split()[:3]
            # notes.append([float(onset), float(offset), float(pitch)])
    # f = np.loadtxt(filepath, delimiter='\t', skiprows=1)
    # for line in f:
    #         # if "OnsetTime" in line:
    #         #     continue
    #         #print(line)
    #         onset, offset, pitch = line[0:3]
    #         notes.append([float(onset), float(offset), float(pitch)])
    #print(notes)
    notes.sort(key=lambda x:x[0])
    return np.array(notes)

def notes_to_frames(pitches, intervals, shape, fps=None):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        if fps != None:
            onset = int(0.5 + fps * onset)
            offset = int(0.5 + fps * offset)
        roll[onset:offset, int(pitch)] = 1

    time = np.arange(roll.shape[0])  # 有效时间序列
    freqs = [roll[t, :].nonzero()[0] for t in time]  # 帧级音高
    return time, freqs


def evaluate_single(labelpath, resultpath, log_file, onset_win=0.05, offset_win=0.05, frame_output=None):
    fp = [0., 0., 0.]
    nfp = [0., 0., 0.]
    op = [0., 0., 0.]
    oop = [0., 0., 0.]
    oopv = [0., 0., 0.]

    fps = config.frames_per_second
    
    label = get_notes(labelpath)
    result = get_notes(resultpath)
    
    label_shape = [int(0.5 + max(label[:,1]) * fps), 128]
    result_shape = [int(0.5 + max(result[:,1]) * fps), 128]
    t_ref, f_ref = notes_to_frames(np.ascontiguousarray(label[:, 2]), np.ascontiguousarray(label[:, :2]), label_shape, fps=fps)
    t_est, f_est = notes_to_frames(np.ascontiguousarray(result[:, 2]), np.ascontiguousarray(result[:, :2]), result_shape, fps=fps)
    
    start_time = time.time()
    
    if frame_output:
        y_pred = (np.sign(frame_output[1] - 0.3) + 1) / 2
        y_pred[np.where(y_pred==0.5)] = 0
        y_true = frame_output[0]
        y_pred = y_pred[0 : y_true.shape[0], :]
        y_true = y_true[0 : y_pred.shape[0], :]
        tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
        fp = [tmp[0][1], tmp[1][1], tmp[2][1]]
        fp_time = time.time()
        # frame
        str = '  fp\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(tmp[0][1]*100, tmp[1][1]*100, tmp[2][1]*100, fp_time - start_time)
        log_file.write(str)
        print(str, end='')
    
    frame_metrics = mir_eval.multipitch.evaluate(t_ref, f_ref, t_est, f_est)
    fp_time = time.time()
    r = frame_metrics['Recall']
    p = frame_metrics['Precision']
    f = hmean([p + eps, r + eps]) - eps
    nfp = [p, r, f]
    # frame
    str = '  nfp\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(p*100, r*100, f*100, fp_time - start_time)
    log_file.write(str)
    print(str, end='')
    
    label_intervals = np.ascontiguousarray(label[:, :2])
    label_pitches = librosa.midi_to_hz(np.ascontiguousarray(label[:, 2]))
    label_velocity = np.ascontiguousarray(label[:, 3])
    result_intervals = np.ascontiguousarray(result[:, :2])
    result_pitches = librosa.midi_to_hz(np.ascontiguousarray(result[:, 2]))
    result_velocity = np.ascontiguousarray(result[:, 3])
    
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        label_intervals, label_pitches, result_intervals, result_pitches,
        offset_ratio =None, #onset_tolerance = onset_win, 
    )
    op_time = time.time()
    op = [p, r, f]
    # onset AKA note
    str = '  op\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(p*100, r*100, f*100, op_time - fp_time)
    log_file.write(str)
    print(str, end='')
    
    
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        label_intervals, label_pitches, result_intervals, result_pitches,
        #onset_tolerance=onset_win, offset_min_tolerance = offset_win
    )
    oop_time = time.time()
    oop = [p, r, f1]
    # onset with offset
    str = ' oop\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\t  %4.2f\n'%(p*100, r*100, f1*100, oop_time - op_time, f1*100 - f*100)
    log_file.write(str)
    print(str, end='')

    p, r, f2, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        label_intervals, label_pitches, label_velocity,
        result_intervals, result_pitches, result_velocity,
    )
    oopv_time = time.time()
    oopv = [p, r, f2]
    # offset，velocity
    str = 'oopv\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\t  %4.2f\n'%(p*100, r*100, f2*100, oopv_time - oop_time, f2*100 - f1*100)
    log_file.write(str)
    print(str)

    return [nfp, op, oop, oopv, fp]

def evaluate_folder(label_path, result_path, log_path=None, onset_win=0.05, offset_win=0.05, err_img=False):
    if log_path == None:
        log_name = '' + datetime.datetime.now().strftime('%y%m%d') + '.log'
        log_path = os.path.join(result_path, log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_file = open(log_path, 'w')
    #filenames=sorted([filename for filename in os.listdir(result_path) if (filename.endswith("_2nd.txt") and ("mpe_16ms" not in filename))])
    filenames=sorted([filename for filename in os.listdir(result_path) if filename.endswith(".txt")])

    fp = [[], [], []]
    op = [[], [], []]
    oop = [[], [], []]
    oopv = [[], [], []]
    n = 0
    total = len(filenames)
    assert total > 0, 'no txt in {}'.format(result_path)
    # fps = 1 / (onset_win / 5)
    fps = config.frames_per_second
    
    if err_img:
        err_img_dir = os.path.join(result_path, 'piano_roll')
        os.makedirs(err_img_dir, exist_ok=True)
    
    for filename in filenames:
        labelpath = os.path.join(label_path, filename.replace('_pred','').replace('.pred','').replace('.wav\'',''))
        #.replace('.txt','.flac.txt')
        resultpath = os.path.join(result_path, filename)
        label = get_notes(labelpath)
        result = get_notes(resultpath)
        str = f'{n:3} | {total}\t{filename}\t{label.shape[0]} | {result.shape[0]}\n'
        log_file.write(str)
        print(str, end='')
        n = n + 1
        
        label_shape = [int(0.5 + max(label[:,1]) * fps), 128]
        result_shape = [int(0.5 + max(result[:,1]) * fps), 128]
        t_ref, f_ref = notes_to_frames(np.ascontiguousarray(label[:, 2]), np.ascontiguousarray(label[:, :2]), label_shape, fps=fps)
        t_est, f_est = notes_to_frames(np.ascontiguousarray(result[:, 2]), np.ascontiguousarray(result[:, :2]), result_shape, fps=fps)
        
        start_time = time.time()
        frame_metrics = mir_eval.multipitch.evaluate(t_ref, f_ref, t_est, f_est)
        fp_time = time.time()
        r = frame_metrics['Recall']
        p = frame_metrics['Precision']
        f = hmean([p + eps, r + eps]) - eps
        fp[0].append(p)
        fp[1].append(r)
        fp[2].append(f)
        # frame
        str = '  fp\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(p*100, r*100, f*100, fp_time - start_time)
        log_file.write(str)
        print(str, end='')
        
        label_intervals = np.ascontiguousarray(label[:, :2])
        label_pitches = librosa.midi_to_hz(np.ascontiguousarray(label[:, 2]))
        label_velocity = np.ascontiguousarray(label[:, 3])
        result_intervals = np.ascontiguousarray(result[:, :2])
        result_pitches = librosa.midi_to_hz(np.ascontiguousarray(result[:, 2]))
        result_velocity = np.ascontiguousarray(result[:, 3])
        
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            label_intervals, label_pitches, result_intervals, result_pitches,
            offset_ratio =None, #onset_tolerance = onset_win, 
        )
        op_time = time.time()

        op[0].append(p)
        op[1].append(r)
        op[2].append(f)
        # onset AKA note
        str = '  op\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(p*100, r*100, f*100, op_time - fp_time)
        log_file.write(str)
        print(str, end='')
        
        if err_img:
            # ``matching[i] == (i, j)`` where reference note ``i`` matches estimated note ``j``.
            matching = mir_eval.transcription.match_notes(
                np.ascontiguousarray(label[:, :2]), 
                librosa.midi_to_hz(np.ascontiguousarray(label[:, 2])), 
                np.ascontiguousarray(result[:, :2]), 
                librosa.midi_to_hz(np.ascontiguousarray(result[:, 2])), 
                offset_ratio =None
            )
            shape = [max(label_shape[0], result_shape[0]), 128]
            roll = np.zeros(tuple(shape))
            for index, (onset, offset, pitch, velocity) in enumerate(label):
                if index in [t[0] for t in matching]:
                    roll[int(0.5 + fps * onset):int(0.5 + fps * offset), int(pitch)] = 1
                else:
                    roll[int(0.5 + fps * onset):int(0.5 + fps * offset), int(pitch)] = 2
            for index, (onset, offset, pitch, velocity) in enumerate(result):
                if index in [t[1] for t in matching]:
                    roll[int(0.5 + fps * onset):int(0.5 + fps * offset), int(pitch)] = 1
                else:
                    roll[int(0.5 + fps * onset):int(0.5 + fps * offset), int(pitch)] = 3

            colors = ['white', 'black', 'yellow', 'red']
            plt.figure(figsize=(64.0, 128.0))
            plt.imshow(roll.T, cmap=plt.cm.colors.ListedColormap(colors))
            # plt.colorbar()  # 添加颜色条
            png_path = os.path.join(err_img_dir, filename.replace('.txt', '.png'))
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
            print(f'Piano roll png is saved at: {png_path}')

        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            label_intervals, label_pitches, result_intervals, result_pitches,
            #onset_tolerance=onset_win, offset_min_tolerance = offset_win
        )
        oop_time = time.time()
        oop[0].append(p)
        oop[1].append(r)
        oop[2].append(f1)
        # onset with offset
        str = ' oop\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\t%4.2f\n'%(p*100, r*100, f*100, oop_time - op_time, f1 - f)
        log_file.write(str)
        print(str, end='')

        p, r, f2, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
            label_intervals, label_pitches, label_velocity,
            result_intervals, result_pitches, result_velocity,
        )
        oopv_time = time.time()
        oopv[0].append(p)
        oopv[1].append(r)
        oopv[2].append(f2)
        # offset，velocity
        str = 'oopv\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\t%4.2f\n'%(p*100, r*100, f*100, oopv_time - oop_time, f2 - f1)
        log_file.write(str)
        print(str, end='')

    mean_fp = [round(np.mean(x)*100, 2) for x in fp]
    mean_op = [round(np.mean(x)*100, 2) for x in op]
    mean_oop = [round(np.mean(x)*100, 2) for x in oop]
    mean_oopv = [round(np.mean(x)*100, 2) for x in oopv]
    
    str = '\n\nmean frame\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_fp) + \
        'mean note\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_op) + \
        'mean note/offset\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_oop) + \
        'mean note/offset&velocity\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_oopv) + \
        '\nframe F1: %4.2f%%'%(mean_fp[2]) + '   note F1: %4.2f%%'%(mean_op[2]) + \
        '   offset F1: %4.2f%%'%(mean_oop[2]) +  '  %4.2f'%(mean_oop[2] - mean_op[2]) + \
        '   vel F1: %4.2f%%'%(mean_oopv[2]) +  '  %4.2f'%(mean_oopv[2] - mean_oop[2]) + \
        '      avg[%4.2f%%]\n'%((mean_fp[2] + mean_op[2] + mean_oop[2] + mean_oopv[2])/4) 

    log_file.write(str)
    print(str, end=f'\n{result_path}')
    
    log_file.close()

def evaluate_folder_grid_search(label_path, pkl_path):
    thresholds = {'onset_thresholds': np.arange(40,46,10)/100, 
                 'offset_thresholds': np.arange(30,36,10)/100, 
                 'frame_thresholds': np.arange(30,36,10)/100}

    tolerant_nums= {
        'onset_tolerant_num': np.arange(2,3),
        'offset_tolerant_num': np.arange(2,3)
    }

    
    total_log_name = os.path.join(pkl_path, 'grid_' + datetime.datetime.now().strftime('%y%m%d') + '.log')
    total_log_file = open(total_log_name, 'w')
    total_grid = [] # thresholds,,,tolerant,,note,,,avg
    
    os.makedirs(os.path.join(pkl_path, 'log',), exist_ok=True)
    
    filenames=sorted([filename for filename in os.listdir(pkl_path) if filename.endswith(".pkl")])
    total = len(filenames)
    assert total > 0, 'no txt in {}'.format(pkl_path)
    
    for onset_threshold in thresholds['onset_thresholds']: 
        for offset_threshold in thresholds['offset_thresholds']: 
            for frame_threshold in thresholds['frame_thresholds']:
                for onset_tolerant_num in tolerant_nums['onset_tolerant_num']:
                    for offset_tolerant_num in tolerant_nums['offset_tolerant_num']:
                        log_name = f'{onset_threshold},{offset_threshold},{frame_threshold}_{onset_tolerant_num},{offset_tolerant_num}_' + datetime.datetime.now().strftime('%y%m%d') + '.log'
                        log_path = os.path.join(pkl_path, 'log', log_name)
                        log_file = open(log_path, 'w')

                        post_processor = RegressionPostProcessor(config.frames_per_second, 
                                classes_num=config.classes_num, onset_threshold=onset_threshold, 
                                offset_threshold=offset_threshold, 
                                frame_threshold=frame_threshold,
                                onset_tolerant_num=onset_tolerant_num,
                                offset_tolerant_num=offset_tolerant_num)

                        fp = [[], [], []]
                        nfp = [[], [], []]
                        op = [[], [], []]
                        oop = [[], [], []]
                        oopv = [[], [], []]
                        n = 0

                        for filename in filenames:
                            labelpath = os.path.join(label_path, filename.replace('_pred','').replace('.pred','').replace('.wav\'','').replace('.pkl','.txt'))

                            filename = os.path.join(pkl_path, filename)
                            total_dict = pickle.load(open(filename, 'rb'))
                            result = post_processor.output_dict_to_note_arrays(total_dict)
                            result_intervals = result[:, 0 : 2]
                            result_pitches = librosa.midi_to_hz(result[:, 2])
                            result_velocity = result[:, 3] * config.velocity_scale
                            
                            # label_intervals = total_dict['ref_on_off_pairs']
                            # label_pitches = total_dict['ref_midi_notes']
                            # label_velocity= total_dict['ref_velocity']
                            label = get_notes(labelpath)
                            label_intervals = np.ascontiguousarray(label[:, :2])
                            label_pitches = librosa.midi_to_hz(np.ascontiguousarray(label[:, 2]))
                            label_velocity = np.ascontiguousarray(label[:, 3])
                            # print(label_velocity)
                            
                            str = f'{n:3} | {total}\t{filename}\t{result_intervals.shape[0]} | {result.shape[0]}\n'
                            log_file.write(str)
                            print(str, end='')
                            print(log_path)
                            n = n + 1

                            start_time = time.time()
                            y_pred = (np.sign(total_dict['frame_output'] - frame_threshold) + 1) / 2
                            y_pred[np.where(y_pred==0.5)] = 0
                            y_true = total_dict['frame_roll']
                            y_pred = y_pred[0 : y_true.shape[0], :]
                            y_true = y_true[0 : y_pred.shape[0], :]
                            tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
                            fp[0].append(tmp[0][1])
                            fp[1].append(tmp[1][1])
                            fp[2].append(tmp[2][1])                    
                            fp_time = time.time()
                            # frame
                            str = '  fp\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(tmp[0][1]*100, tmp[1][1]*100, tmp[2][1]*100, fp_time - start_time)
                            log_file.write(str)
                            print(str, end='')
                            
                            fps = config.frames_per_second
                            label_shape = [int(0.5 + max(label[:,1]) * fps), 128]
                            result_shape = [int(0.5 + max(result[:,1]) * fps), 128]
                            t_ref, f_ref = notes_to_frames(np.ascontiguousarray(label[:, 2]), np.ascontiguousarray(label[:, :2]), label_shape, fps=fps)
                            t_est, f_est = notes_to_frames(np.ascontiguousarray(result[:, 2]), np.ascontiguousarray(result[:, :2]), result_shape, fps=fps)
                            frame_metrics = mir_eval.multipitch.evaluate(t_ref, f_ref, t_est, f_est)
                            nfp_time = time.time()
                            r = frame_metrics['Recall']
                            p = frame_metrics['Precision']
                            f = hmean([p + eps, r + eps]) - eps
                            nfp[0].append(p)
                            nfp[1].append(r)
                            nfp[2].append(f)
                            # frame
                            str = ' nfp\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(p*100, r*100, f*100, nfp_time - fp_time)
                            log_file.write(str)
                            print(str, end='')
                            
                            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                                label_intervals, label_pitches, result_intervals, result_pitches,
                                offset_ratio =None, #onset_tolerance = onset_win, 
                            )
                            op_time = time.time()

                            op[0].append(p)
                            op[1].append(r)
                            op[2].append(f)
                            # onset AKA note
                            str = '  op\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\n'%(p*100, r*100, f*100, op_time - nfp_time)
                            log_file.write(str)
                            print(str, end='')
                            
                            p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                                label_intervals, label_pitches, result_intervals, result_pitches,
                                #onset_tolerance=onset_win, offset_min_tolerance = offset_win
                            )
                            oop_time = time.time()
                            oop[0].append(p)
                            oop[1].append(r)
                            oop[2].append(f1)
                            # onset with offset
                            str = ' oop\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\t%4.2f\n'%(p*100, r*100, f1*100, oop_time - op_time, f1*100 - f*100)
                            log_file.write(str)
                            print(str, end='')

                            p, r, f2, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
                                label_intervals, label_pitches, label_velocity,
                                result_intervals, result_pitches, result_velocity,
                            )
                            oopv_time = time.time()
                            oopv[0].append(p)
                            oopv[1].append(r)
                            oopv[2].append(f2)
                            # offset，velocity
                            str = 'oopv\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\t[%.2fs]\t%4.2f\n'%(p*100, r*100, f2*100, oopv_time - oop_time, f2*100 - f1*100)
                            log_file.write(str)
                            print(str, end='')
                            log_file.flush()

                        mean_fp = [round(np.mean(x)*100, 2) for x in fp]
                        mean_nfp = [round(np.mean(x)*100, 2) for x in nfp]
                        mean_op = [round(np.mean(x)*100, 2) for x in op]
                        mean_oop = [round(np.mean(x)*100, 2) for x in oop]
                        mean_oopv = [round(np.mean(x)*100, 2) for x in oopv]
                        avg = (max(mean_fp[2], mean_nfp[2]) + mean_op[2] + mean_oop[2] + mean_oopv[2])/4
                        str0 = '\nmean frame\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_fp) + \
                            'mean note2frame\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_nfp) + \
                            'mean note\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_op) + \
                            'mean note/offset\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n'%tuple(mean_oop) + \
                            'mean note/offset&velocity\tP: %4.2f%%\tR:  %4.2f%%\tF1: %4.2f%%\n\n'%tuple(mean_oopv)
                        str1 = 'frame F1: %4.2f%%'%(mean_fp[2]) + '   note2frame F1: %4.2f%%'%(mean_nfp[2]) + \
                            '   note F1: %4.2f%%'%(mean_op[2]) + '   offset F1: %4.2f%%'%(mean_oop[2]) +  '  %4.2f'%(mean_oop[2] - mean_op[2]) + \
                            '   vel F1: %4.2f%%'%(mean_oopv[2]) +  '  %4.2f'%(mean_oopv[2] - mean_oop[2]) + '      avg[%4.2f%%]'%(avg) + \
                            f'\t\tthreshold=[{onset_threshold},{offset_threshold},{frame_threshold}]\ttolerant_num=[{onset_tolerant_num},{offset_tolerant_num}]\n'
                        log_file.write(str0 + str1)
                        print(str0 + str1, end='')
                        
                        log_file.close()
                        
                        total_grid.append([onset_threshold, offset_threshold, frame_threshold, onset_tolerant_num, offset_tolerant_num, mean_fp[2], mean_nfp[2], mean_op[2], mean_oop[2], mean_oopv[2], avg]) # thresholds,,,tolerant,,note,,,avg
                        total_log_file.write(str1)
    total_grid = sorted(total_grid, key=lambda x: x[-1], reverse=True)
    for i, grid in enumerate(total_grid):
        total_log_file.write(f'{i:3} | threshold=[{grid[0]},{grid[1]},{grid[2]}]\ttolerant_num=[{grid[3]},{grid[4]}]\tavg[{grid[10]:4.2f}%]\t\tframe F1: {grid[5]:4.2f}%   note2frame F1: {grid[6]:4.2f}%   note F1: {grid[7]:4.2f}%   offset F1: {grid[8]:4.2f}%  {(grid[8] - grid[7]):4.2f}  vel F1: {grid[9]:4.2f}%\n  {(grid[9] - grid[8]):4.2f}')
    total_log_file.close()
    print("Saved at " + total_log_name)

# python /home/data/wrm_data/temp_task/piano_transcription-master/utils/comment_notes.py --label_path /home/data/wrm_data/temp_task/piano_transcription-master/workspaces/hdf5s/v3_test_labels --result_path /home/data/wrm_data/temp_task/piano_transcription-master/workspaces/probs/Four_blocks/none/dataset=maestro/test/batch_size=2/532500_iterations/532500_iterations_midi/txt
if __name__=='__main__':
    args = parse_args()
    label_path = args.label_path
    result_path = args.result_path
    log_path = args.log_path
    onset_win = args.onset_win
    offset_win = args.offset_win
    evaluate_folder(label_path, result_path, log_path, onset_win, offset_win)