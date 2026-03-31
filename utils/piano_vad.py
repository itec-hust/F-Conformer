import numpy as np


def note_detection_with_onset_offset_regress(frame_output, onset_output, 
    onset_shift_output, offset_output, offset_shift_output, 
    frame_threshold, onset_tolerant_num = 0):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.
    由frame下降沿_offset 共同判断offset
    后面两个函数仅根据其中一个判断
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445], 
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None
    count = [0,0,0,0,0] # 连音onset, 连音offset, 单音offset,  单音frame, 12s未检

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two 
                consecutive notes being played. 
                同一音高连续两个不同onset被检测到 但没有offset, 强制截断上一个onset, 即上一个的offset=当前onset-1"""
                # fin = offset_occur if offset_occur else i - 1
                # print(f'onset_next:{i-bgn}\tframe: {frame_disappear - bgn if frame_disappear else None}\toffset:{offset_occur-bgn if offset_occur else None}')
                for j in range(i - 1, (i + bgn) // 2, -1):
                    """Offset detected"""
                    """bgn --------- offset_occur --- frame_disappear"""
                    if offset_output[j] == 1:
                        offset_occur = j
                        count[4] += 1
                        break
                fin = offset_occur if offset_occur else i - 1
                count[0] += 1
                # if fin != bgn:
                #     if velocity_output[bgn] <= 1e-7:
                #         print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                #     output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                #         0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            # 检测到frame消失或超时 拼装note_event
            if frame_disappear:
                for j in range(frame_disappear, (frame_disappear + bgn) // 2, -1):
                    """Offset detected"""
                    """bgn --------- offset_occur --- frame_disappear"""
                    if offset_output[j] == 1:
                        offset_occur = j
                        break
                    
                if offset_occur:
                    count[1] += 1
                    fin = offset_occur
                else:
                    """bgn -------------------------- frame_disappear"""
                    """bgn --- offset_occur --------- frame_disappear"""
                    """bgn -------------------------- frame_disappear --- offset_occur"""
                    count[2] += 1
                    fin = frame_disappear
                # if velocity_output[bgn] <= 1e-7:
                #     print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                count[3] += 1
                # if velocity_output[bgn] <= 1e-7:
                #     print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])
    for o in output_tuples:
        assert o[1] > o[0], f'{o[0]} should be bigger than {o[1]}'
    return output_tuples, count

def note_detection_with_onset_offset_regress1(frame_output, onset_output, 
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold, onset_tolerant_num = 0):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.
    由frame的下降沿和offset 两者的最小值作为音符offset
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445], 
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two 
                consecutive notes being played. 
                同一音高连续两个不同onset被检测到"""
                fin = frame_disappear if frame_disappear else (offset_occur if offset_occur else i - 1)
                if fin != bgn:
                    if velocity_output[bgn] <= 1e-7:
                        print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                        0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            # 检测到frame下降沿和offset
            if frame_disappear and offset_occur:
                fin = min(frame_disappear, offset_occur)
                if velocity_output[bgn] <= 1e-7:
                    print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], max(velocity_output[max(0,bgn-onset_tolerant_num):min(1+bgn+onset_tolerant_num,onset_output.shape[0])])])
                bgn, frame_disappear, offset_occur = None, None, None

            # 检测到超时
            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = frame_disappear if frame_disappear else (offset_occur if offset_occur else i)
                if velocity_output[bgn] <= 1e-7:
                    print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin],max(velocity_output[max(0,bgn-onset_tolerant_num):min(1+bgn+onset_tolerant_num,onset_output.shape[0])])])
                bgn, frame_disappear, offset_occur = None, None, None
    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])
    for o in output_tuples:
        assert o[1] > o[0], f'{o[0]} should be bigger than {o[1]}'
    return output_tuples

def note_detection_without_offset(frame_output, onset_output, 
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445], 
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two 
                consecutive notes being played."""
                fin = i - 1
                if fin != bgn:
                    if velocity_output[bgn] <= 1e-7:
                        print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                        0, velocity_output[bgn]])
                frame_disappear = None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if frame_disappear:
                fin = frame_disappear
                if velocity_output[bgn] <= 1e-7:
                    print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear = None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                if velocity_output[bgn] <= 1e-7:
                    print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear = None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])
    for o in output_tuples:
        assert o[1] > o[0], f'{o[0]} should be bigger than {o[1]}'
    return output_tuples

def note_detection_without_frame(frame_output, onset_output, 
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445], 
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
        ...]
    """
    output_tuples = []
    bgn = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two 
                consecutive notes being played."""
                fin = i - 1
                if fin != bgn:
                    if velocity_output[bgn] <= 1e-7:
                        print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                        0, velocity_output[bgn]])
                offset_occur = None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

                if offset_occur :
                    """bgn --------- offset_occur """
                    fin = offset_occur
                if velocity_output[bgn] <= 1e-7:
                    print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, offset_occur = None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                if velocity_output[bgn] <= 1e-7:
                    print(f'velocity_output[bgn] == {velocity_output[bgn]}')
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, offset_occur = None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])
    for o in output_tuples:
        assert o[1] > o[0], f'{o[0]} should be bigger than {o[1]}'
    return output_tuples


def pedal_detection_with_onset_offset_regress(frame_output, offset_output, 
    offset_shift_output, frame_threshold):
    """Process prediction array to pedal events information.
    
    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift], 
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533], 
        [1909, 1947, 0.30730522, -0.45764327], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            """Pedal onset detected"""
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


###### Google's onsets and frames post processing. Only used for comparison ######
def onsets_frames_note_detection(frame_output, onset_output, offset_output, 
    velocity_output, threshold):
    """Process pedal prediction matrices to note events information. onset_ouput 
    is used to detect the presence of notes. frame_output is used to detect the 
    offset of notes.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      threshold: float
    
    Returns: 
      bgn_fin_pairs: list of [bgn, fin, velocity]. E.g. 
        [[1821, 1909, 0.47498, 0.72119445], 
         [1909, 1947, 0.30730522, 0.64200014], 
         ...]
    """
    output_tuples = []

    loct = None
    for i in range(onset_output.shape[0]):
        # Use onset_output is used to detect the presence of notes
        if onset_output[i] > threshold:
            if loct:
                output_tuples.append([loct, i, velocity_output[loct]])
            loct = i
        if loct and i > loct:
            # Use frame_output is used to detect the offset of notes
            if frame_output[i] <= threshold:
                output_tuples.append([loct, i, velocity_output[loct]])
                loct = None

    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def onsets_frames_pedal_detection(frame_output, offset_output, frame_threshold):
    """Process pedal prediction matrices to pedal events information.
    
    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin], 
      e.g., [
        [1821, 1909], 
        [1909, 1947], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples