# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.inference import run_offline
from amt_tools.features import HCQT

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedNoteTranscriber, \
                                 StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, \
                               MultipitchEvaluator, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy

import amt_tools.tools as tools

# Regular imports
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import jams
import os


track = '00_Jazz2-187-F#_solo'

# Define path to audio and ground-truth
audio_path = os.path.join(tools.DEFAULT_DATASETS_DIR, 'GuitarSet', 'audio_mono-mic', f'{track}_mic.wav')
#audio_path = f'path/to/WAV.wav'
jams_path = os.path.join(tools.DEFAULT_DATASETS_DIR, 'GuitarSet', 'annotation', f'{track}.jams')
#jams_path = f'path/to/JAM.jams'

# Define time boundaries to plot [x_min, x_max]
x_bounds = [-0.5, 20.5]

# Construct the path to the model to evaluate
model_path = os.path.join('.', 'generated', 'experiments', 'FretNet_SynthTab_HCQT_07-30-2023@16:11', 'models', 'model-100.pt')
#model_path = os.path.join('.', 'generated', 'experiments', '<EXPERIMENT_DIR>', 'models', 'model-<CHECKPOINT>.pt')

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512
# Choose the GPU on which to perform evaluation
gpu_id = 0

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load the model onto the specified device
model = torch.load(model_path, map_location=device)
model.change_device(device)
model.eval()

# Initialize the estimation pipeline (Tablature -> Stacked Multi Pitch -> Notes | Multi Pitch)
estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                            StackedNoteTranscriber(profile=model.profile),
                            StackedMultiPitchCollapser(profile=model.profile)])

# Initialize the evaluation pipeline (Multi Pitch | Tablature)
evaluator = ComboEvaluator([MultipitchEvaluator(),
                            TablatureEvaluator(profile=model.profile),
                            SoftmaxAccuracy()])

# Create an HCQT feature extraction module comprising
# the first five harmonics and a sub-harmonic, where each
# harmonic transform spans 4 octaves w/ 3 bins per semitone
data_proc = HCQT(sample_rate=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.note_to_hz('E2'),
                 harmonics=[0.5, 1, 2, 3, 4, 5],
                 n_bins=144, bins_per_octave=36)

# Load and normalize the audio along with the sampling rate
audio, fs = tools.load_normalize_audio(audio_path, fs=sample_rate, norm=np.inf)

# Compute the features
features = {tools.KEY_FEATS : data_proc.process_audio(audio),
            tools.KEY_TIMES : data_proc.get_times(audio)}

# Perform inference offline
predictions = run_offline(features, model, estimator)

# Extract the ground-truth and predicted datasets
stacked_notes_est = predictions[tools.KEY_NOTES]

##############################
# Ground-Truth               #
##############################

# Open up the JAMS data
jam = jams.load(jams_path)

# Get the total duration of the file
duration = tools.extract_duration_jams(jam)

# Get the times for the start of each frame
times = tools.get_frame_times(duration, sample_rate, hop_length)

# Load the ground-truth notes
stacked_notes_ref = tools.load_stacked_notes_jams(jams_path)

# Obtain the multipitch predictions
multipitch_ref = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes_ref, times, model.profile)
# Determine the ground-truth datasets
tablature_ref = tools.stacked_multi_pitch_to_tablature(multipitch_ref, model.profile)
# Collapse the multipitch array
multipitch_ref = tools.stacked_multi_pitch_to_multi_pitch(multipitch_ref)

# Construct the ground-truth dictionary
ground_truth = {tools.KEY_MULTIPITCH : multipitch_ref,
                tools.KEY_TABLATURE : tablature_ref,
                tools.KEY_NOTES : stacked_notes_ref}

##############################
# Results                    #
##############################

# Evaluate the predictions and track the results
results = evaluator.process_track(predictions, ground_truth)

# Print results to the console
print(results)

##############################
# Plotting                   #
##############################

# Convert the ground-truth notes to frets
stacked_frets_ref = tools.stacked_notes_to_frets(stacked_notes_ref)

fig_ref = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_ref = tools.plot_guitar_tablature(stacked_frets_ref, x_bounds=x_bounds, fig=fig_ref)
fig_ref.suptitle('Reference')

# Convert the predicted notes to frets
stacked_frets_est = tools.stacked_notes_to_frets(stacked_notes_est)

# Plot both sets of notes and add an appropriate title
fig_est = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_est = tools.plot_guitar_tablature(stacked_frets_est, x_bounds=x_bounds, fig=fig_est)
fig_est.suptitle('Estimated')

# Extract multi-pitch estimates
multipitch_est = predictions[tools.KEY_MULTIPITCH]

# Determine which indices correspond to the selected range
valid_idcs = (times >= x_bounds[0]) & (times <= x_bounds[-1])

fig_mpe = tools.initialize_figure(interactive=False, figsize=(20, 5))
fig_mpe = tools.plot_pianoroll(multipitch_ref[..., valid_idcs], times=times[valid_idcs],
                               profile=model.profile, alpha=0.5, fig=fig_mpe)
fig_mpe = tools.plot_pianoroll(multipitch_est[..., valid_idcs], times=times[valid_idcs],
                               profile=model.profile, overlay=True, color='orange', alpha=0.5, fig=fig_mpe)
fig_mpe.suptitle('Multi-Pitch Estimates and Ground-Truth')

# Display the plots
plt.show(block=True)
