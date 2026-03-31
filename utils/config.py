sample_rate = 16000
classes_num = 88    # Number of notes of piano
begin_note = 21     # MIDI note of A0, the lowest note of a piano.
segment_seconds = 2.2	# Training segment duration(second) default: 10.
hop_seconds = 1.
frames_per_second = 50 # 100  # frames_per_segment = segment_seconds * frames_per_second = 200+1
velocity_scale = 128
hop_length = sample_rate // frames_per_second

mel_config = {
    'sr' : sample_rate,
    'n_fft' : 2048,
    'n_mels' : 256,
    'fmin' : 30, # 27.5
    'fmax' : sample_rate // 2
}


cal_cqt = True
conv_trans_block = True
convtrans_channel = 128
output_channel = 256
# dilations = [1, 1, 2, 2]
bin_ratios = [1,1,0.5,0.25] # [1,1,1,1] # [1,1,1,0.25] # 
log_scale = False # True # 
bin_deltas = [0,0,0,0] # [5,5,5,5] # [2,2,2,2] # [1,1,1,1] # [1,1,0,0] # [10,10,10,10] # 
num_heads = 1 # 1
cqt_config = {
    "sr": sample_rate,
    "hop_length": hop_length,
    "n_fft": 2048,
    "fmin": 27.5,
    "n_bins": 352,
    "bins_per_octave": 48
}
