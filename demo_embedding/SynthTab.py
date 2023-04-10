# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import TranscriptionDataset
from augment import process_audio_signals
import time
import amt_tools.tools as tools

# Regular imports
import numpy as np
import guitarpro
import librosa
import jams
import torch
import torchaudio
import random
import os
import soundfile as sf

jams.schema.add_namespace(os.path.join('..', 'gp_to_JAMS', 'note_tab.json'))


def load_stacked_notes_jams(jams_path):
    """
    TODO - copying function here for now to address tiny changes
    """

    # Load the metadata from the jams file
    jam = jams.load(jams_path)

    # Extract all of the midi note annotations
    note_data_slices = jam.annotations['note_tab']

    # Extract the tempo from the annotation
    # TODO - dealing with changing tempos
    tempo = jam.annotations['tempo'].pop().data[0].value

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    # Loop through the slices of the stack
    for slice_notes in note_data_slices:
        # Initialize lists to hold the pitches and intervals
        pitches, intervals = list(), list()

        # Loop through the notes pertaining to this slice
        for note in slice_notes:
            # Append the note pitch and interval to the respective list
            pitches.append(note.value['fret'])
            intervals.append([note.time, note.time + note.duration])

        # Convert the pitch and interval lists to arrays
        pitches, intervals = np.array(pitches), np.array(intervals)

        # Add fret to open-string tuning to obtain pitch
        pitches += slice_notes.sandbox['open_tuning']

        # Convert interval times from ticks to seconds
        intervals = (60 / tempo) * intervals / guitarpro.Duration.quarterTime

        # Extract the string label for this slice
        #string = slice_notes.sandbox['string_index']
        string = slice_notes.sandbox['open_tuning']

        # Add the pitch-interval pairs to the stacked notes dictionary under the string entry as key
        stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, string))

    # Re-order keys starting from lowest string
    stacked_notes = {librosa.midi_to_note(i) : stacked_notes[i] for i in sorted(stacked_notes.keys())}

    return stacked_notes


class SynthTab(TranscriptionDataset):
    """
    Implements a wrapper for SynthTab (TODO - url).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                       profile=None, num_frames=None, audio_norm=-1, reset_data=False, store_data=True,
                       save_data=True, save_loc=None, seed=0):
        """
        Initialize an instance of SynthTab.

        Parameters
        ----------
        See TranscriptionDataset class...
        """

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames,
                         audio_norm, False, reset_data, store_data, save_data, save_loc, seed)

    def get_tracks(self, split):
        """
        Get the tracks associated with a partition of the dataset.

        Parameters
        ----------
        split : string
          Name of the partition from which to fetch tracks

        Returns
        ----------
        tracks : list of strings
          Names of tracks within the given partition
        """

        # Initialize a list to hold track names
        tracks = list()

        # Construct a path to the directory containing songs in the split
        split_dir = os.path.join(self.base_dir, split)

        # Parse the split directory to obtain tacks as guitars for each song
        for root, dirs, files in os.walk(split_dir):
            if 'ground_truth.jams' in files:
                # Obtain the song as parent directory
                song = os.path.basename(root)
                # Construct paths to each guitar from the top-level directory
                tracks += [os.path.join(split, song, guitar) for guitar in dirs]

        return tracks

    def get_track_data(self, track, seq_length=None):
        """
        TODO
        """
        # start log time
        # timestart = time.time()
        # Check if a specific sequence length was given
        if seq_length is None:
            if self.seq_length is not None:
                # Use the global sequence lengh
                seq_length = self.seq_length

        # Default data to None (not existing)
        data = dict()

        # Add the track ID to the dictionary
        data[tools.KEY_TRACK] = track

        # Determine the expected path to the track's data
        gt_path = self.get_gt_dir(track)

        # Check if an entry for the data exists
        if self.save_data and os.path.exists(gt_path):
            # Load and unpack the data
            audio = torch.load(gt_path)
        else:
            # Construct the paths to the track's audio
            audio_paths = self.get_audio_paths(track)

            audio = list()

            for path in audio_paths:
                # Load and normalize the audio
                audio_, fs_ = torchaudio.load(path)
                # Average channels to obtain mono-channel
                audio_ = torch.mean(audio_, dim=0, keepdim=True)
                # Resample audio to appropriate sampling rate
                audio += [torchaudio.functional.resample(audio_, fs_, self.sample_rate)]

            audio = torch.cat(audio)

            if self.save_data:
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                torch.save(audio, gt_path)

        if seq_length is not None:
            audio_length = audio[0].shape[-1]

            if audio_length >= seq_length:
                # Sample a random starting index for the trim
                sample_start = self.rng.randint(0, audio_length - seq_length + 1)
                times = self.data_proc.get_times(audio[0])
                # Trim audio to the sequence length
                audio = torch.cat([a[..., sample_start: sample_start + seq_length].unsqueeze(0) for a in audio])
                # Determine the frames contained in this slice
                frame_start = sample_start // self.hop_length
                frame_end = frame_start + self.num_frames
                times = times[..., frame_start:frame_end]
            else:
                # Determine how much padding is required
                pad_total = seq_length - audio_length
                # Pad the audio with zeros
                audio = torch.cat([torch.nn.functional.pad(a, (0, pad_total)).unsqueeze(0) for a in audio])
                times = self.data_proc.get_times(audio[0])

        # audio = process_audio_signals(audio, seq_length)
        # Mix microphone signals into mono-channel audio
        audio = torch.mean(audio, dim=0, keepdim=True)
        
        if self.audio_norm == -1:
            # Calculate the square root of the squared mean
            rms = torch.sqrt(torch.mean(audio ** 2))

            # If root-mean-square is zero (audio is all zeros), do nothing
            if rms > 0:
                # Divide the audio by the root-mean-square
                audio = audio / rms
        else:
            return NotImplementedError

        # Calculate the features
        features = self.data_proc.process_audio(audio.squeeze().numpy())

        # Construct the path to the track's JAMS data
        jams_path = self.get_jams_path(track)

        # Load the notes by string from the JAMS file
        stacked_notes = load_stacked_notes_jams(jams_path)

        # Represent the string-wise notes as a stacked multi pitch array
        stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

        # Convert the stacked multi pitch array into tablature
        tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

        # Convert the stacked multi pitch array into a single representation
        multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

        # Add all relevant ground-truth to the dictionary
        data.update({tools.KEY_FS: self.sample_rate,
                     tools.KEY_AUDIO: audio,
                     tools.KEY_FEATS: features,
                     tools.KEY_TABLATURE: tablature,
                     tools.KEY_MULTIPITCH: multi_pitch})

        # print('Time to load track: ', time.time() - timestart)
        return data

    def get_audio_paths(self, track):
        """
        Get the paths to the audio of a track.

        Parameters
        ----------
        track : string
          SynthTab track name

        Returns
        ----------
        audio_paths : list of string
          Paths to the various mic recordings for the track
        """

        # Construct a path to the track's audio
        track_dir = os.path.join(self.base_dir, track)

        # Get paths to the different mic recordings for the track
        audio_paths = [os.path.join(track_dir, p) for p in os.listdir(track_dir)]

        return audio_paths

    def get_jams_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          SynthTab track name

        Returns
        ----------
        jams_path : string
          Path to the JAMS file of the specified track
        """

        # Get the path to the annotations
        jams_path = os.path.join(self.base_dir, os.path.dirname(track), 'ground_truth.jams')

        return jams_path

    @staticmethod
    def available_splits():
        """
        TODO.

        Returns
        ----------
        splits : list of strings
          TODO
        """

        splits = ['train', 'val']

        return splits

    @staticmethod
    def download(save_dir):
        """
        TODO
        """

        return NotImplementedError
