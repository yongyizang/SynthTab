import os,sys
import numpy as np
import time as time
import audio_dspy as adsp
import soundfile
import scipy.signal as signal
import matplotlib.pyplot as plt
from audiomentations import *
import scipy.interpolate as interpolate

def augment_mic_signals(mic_signals, sample_length, sample_rate, noise_signal=None, microphone_ir_signals=None, room_ir_signal=None):
    # start_time = time.time()
    sample_length += 1
    # pad mic_signals with one 1e-9 sample
    fs = sample_rate
    highest_freq = fs // 2 # Nyquist frequency
    output_audio = np.zeros(sample_length)
    for i in range(mic_signals.shape[0]):
        mic_signal = mic_signals[i] # get mic signal
        mic_signal = np.pad(mic_signal, (0, 1), 'constant', constant_values=(0, 1e-9))
        mic_signal = np.where(mic_signal == 0, 1e-9, mic_signal)
        # print('mic_signal.shape', mic_signal.shape)
        # add white noise
        mic_signal = mic_signal + np.random.normal(0, 0.01, len(mic_signal))
        if microphone_ir_signals is not None:
            # convolution with microphone ir
            mic_signal = signal.fftconvolve(mic_signal, microphone_ir_signals[[mic_signals.index(mic_signal)][0]], mode='same')
        # normalize, with random volume balancing
        mic_signal = mic_signal * np.random.uniform(0.5, 1.5) / np.max(np.abs(mic_signal))
        # add to output audio
        output_audio = output_audio + mic_signal
    # normalize output_audio
    output_audio = output_audio / np.max(np.abs(output_audio))
    
    augment = Compose([
        AirAbsorption(p=0.9, min_distance=0.1, max_distance=2.0),
        LowPassFilter(p=0.5, max_cutoff_freq=1000),
        HighPassFilter(p=0.5, min_cutoff_freq=8000, max_cutoff_freq=highest_freq),
        LowShelfFilter(p=0.5, min_gain_db=-2, max_gain_db=2, min_center_freq=300, max_center_freq=1000),
        HighShelfFilter(p=0.5, min_gain_db=-2, max_gain_db=2, min_center_freq=5000, max_center_freq=highest_freq),
        SevenBandParametricEQ(p=0.6, min_gain_db=-3.0, max_gain_db=3.0),
        RoomSimulator(p=0.5, min_size_x = 1, max_size_x = 5, min_size_y = 1, max_size_y = 5, max_order = 3),
        Limiter(p=0.5),
    ])

    output_audio_ = augment(samples=output_audio, sample_rate=fs)
    output_audio_ = output_audio_[:sample_length]
    # print('output_audio_.shape', output_audio_.shape)
    
    if room_ir_signal is not None and noise_signal is not None:
        # convolution with room ir
        output_audio_ir = signal.fftconvolve(output_audio_, room_ir_signal[0], mode='same')
        output_audio_ = output_audio_ir * np.random.uniform(0.5, 1.0) + output_audio_ + noise_signal
    
    # add random volume fluctuations
    # generate a series of random numbers between 0.5 and 1.5 every 0.05s, and interpolate between them
    random_volumes = np.random.uniform(0.5, 1.5, int(sample_length / fs * 20))
    random_interpolated_volumes = interpolate.interp1d(np.linspace(0, sample_length, len(random_volumes)), random_volumes, kind='linear')(np.arange(sample_length))
    output_audio_ = output_audio_ * random_interpolated_volumes
    output_audio_ = output_audio_ / np.max(np.abs(output_audio_))
    
    # print("Time Elapsed: ", time.time() - start_time, "s")
    return output_audio_[:sample_length-1]

# run on 10s dummy audio

if __name__ == "__main__":
    # dummy_audio = np.random.normal(0, 0.01, 220500)
    audio_1 = soundfile.read('/home/frank/SynthTab/demo_data/val/Zuly - Fable 31/gibson/gibson_body_only.flac')[0]
    audio_2 = soundfile.read('/home/frank/SynthTab/demo_data/val/Zuly - Fable 31/gibson/gibson_neck_only.flac')[0]
    audio_1 = audio_1[:220500, 0]
    # replace all 0s with 1e-9
    audio_1 = np.where(audio_1 == 0, 1e-9, audio_1)
    audio_2 = audio_2[:220500, 0]
    audio_2 = np.where(audio_2 == 0, 1e-9, audio_2)
    dummy_audio = [audio_1, audio_2]
    output_audio = augment_mic_signals(dummy_audio, 220500, 44100, None, None, None)
    print(output_audio)
    soundfile.write("output.wav", output_audio, 44100)