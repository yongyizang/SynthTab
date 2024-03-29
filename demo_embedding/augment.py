import random
from pathlib import Path
from typing import List
import librosa
import numpy as np
import soundfile as sf
from pedalboard import *
from pedalboard.io import AudioFile

def generate_random_board() -> Pedalboard:
    board = Pedalboard()
    board.append(HighpassFilter(cutoff_frequency_hz=float(random.randint(100, 1000))))
    board.append(LowpassFilter(cutoff_frequency_hz=float(random.randint(16000, 20000))))
    board.append(Compressor(threshold_db=float(-random.randint(10, 60)), ratio=float(random.randint(1,10)), attack_ms=float(random.randint(1,1000)), release_ms=float(random.randint(1,1000))))
    tipping_point = random.randint(100, 5000)
    board.append(LowShelfFilter(cutoff_frequency_hz=random.randint(100, tipping_point), gain_db=random.random()*20 - 10, q=random.random()))
    board.append(HighShelfFilter(cutoff_frequency_hz=random.randint(tipping_point, 15000), gain_db=random.random()*20 - 10, q=random.random()))
    board.append(Reverb(room_size=random.random(), damping=random.random(), wet_level=random.random(), dry_level=random.random(), width=random.random(), freeze_mode=random.randint(0,1)))
    return board

def process_audio_signals(audio_signals: List[np.ndarray], target_audio_length = None):
    """ process (assuming different microphone signals) audio signals

    Args:
        audio_signals (List[np.ndarray]): an array of audio signals.
        target_audio_length (int, optional): target audio length of returned audio signal, in samples. Defaults to None. When this is set to None, the returned audio signal will be the length of the longest audio signal in the audio_signals array; otherwise, the returned audio signal will be the target_audio_length.

    Returns:
        augmented audio signal.
    """
    length_array = []
    for i in range(len(audio_signals)):
        if np.max(np.abs(audio_signals[i])) > 0:
            audio_signals[i] = 0.99 * audio_signals[i] / np.max(np.abs(audio_signals[i]))
            length_array.append(len(audio_signals[i]))
    if len(length_array) == 0 or np.max(np.abs(length_array)) == 0:
        return np.zeros(target_audio_length)
    max_audio_length = max(length_array)

    output_audio = []
    for i in range(len(audio_signals)):
        # sf.write('processed_audio.wav', audio_signals[i], 22050)
        # exit()
        # board = generate_random_board()
        # audio_signals[i] = np.array(audio_signals[i])
        # processed_audio = board.process(audio_signals[i], 22050)
        # print(processed_audio)
        # sf.write('processed_audio.wav', processed_audio, 22050)
        # exit()
        output_audio.append(audio_signals[i] * random.random())
    final_mix = np.zeros(max_audio_length)
    # print(final_mix.shape)
    for audio in output_audio:
        # add the audio to the final mix
        # np.add(final_mix, audio, out=final_mix, casting="unsafe")
        final_mix[:len(audio)] += audio
        
    # if target_audio_length is not None:
    #     if max_audio_length > target_audio_length:
    #         final_mix = final_mix[:target_audio_length]
    #     else:
    #         final_mix = np.pad(final_mix, (0, target_audio_length - max_audio_length), 'constant')
    # else:
    #     final_mix = final_mix[:max_audio_length]
        
    # # if NaN then clip as zero
    # if np.isnan(final_mix).any():
    #     np.nan_to_num(final_mix, copy=False, nan=0)
        
    # # if it clips then normalize
    # if np.max(np.abs(final_mix)) > 1:
    #     final_mix = final_mix / np.max(np.abs(final_mix))
    return final_mix
    

def process_audio_files(audio_files: List[str]):
    output_audio = []

    for audio_file in audio_files:
        audio_data, sample_rate = sf.read(audio_file)
        audio_data = 0.99 * audio_data / np.max(np.abs(audio_data))
        board = generate_random_board()
        processed_audio = board.process(audio_data, sample_rate)
        processed_audio = np.mean(processed_audio, axis=1)
        processed_audio = processed_audio * random.random()
        output_audio.append(processed_audio)
        
    maxLength = 0
    for audio in output_audio:
        if len(audio) > maxLength:
            maxLength = len(audio)
            
    final_mix = np.zeros(maxLength)
    final_mix = final_mix.astype(np.float32)
    for audio in output_audio:
        final_mix[:len(audio)] += audio
        
    final_mix = final_mix / np.max(np.abs(final_mix))
    return final_mix

# Example usage:
#audio_files = ['/Volumes/ext/taylor/Zwan - The Number Of The Beast - gp4-2 - Acoustic Steel Guitar- midi/taylor_finger_body.flac', '/Volumes/ext/taylor/Zwan - The Number Of The Beast - gp4-2 - Acoustic Steel Guitar- midi/taylor_finger_amb.flac']
#processed_audio = process_audio_files(audio_files)
#sf.write('processed_audio.wav', processed_audio, 44100)
