# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import TranscriptionDataset
#from augment_new import augment_mic_signals
from augment import process_audio_signals

import amt_tools.tools as tools

# Regular imports
import numpy as np
import torchaudio
import guitarpro
import librosa
import torch
import jams
import os


# Include the namespace for our tablature note-events
jams.schema.add_namespace(os.path.join('..', 'gp_to_JAMS', 'note_tab.json'))


def load_stacked_notes_jams(jams_path):
    """
    Extract MIDI notes spread across sources (e.g. guitar strings) into a dictionary from a JAMS file.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Load the data from the JAMS file
    jam = jams.load(jams_path)

    # Extract all midi note annotations
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
            # Append the note's fret
            pitches.append(note.value['fret'])
            # Append the note's onset and offset (in ticks)
            intervals.append([note.time, note.time + note.duration])

        # Convert the pitch and interval lists to arrays
        pitches, intervals = np.array(pitches), np.array(intervals)

        # Extract the open string pitch for the string
        string_pitch = slice_notes.sandbox['open_tuning']

        # Add open-string tuning to obtain pitches
        pitches += string_pitch

        # Convert onset and offset times from ticks to seconds
        intervals = (60 / tempo) * intervals / guitarpro.Duration.quarterTime

        # Add the pitch-interval pairs to the stacked notes dictionary under the string entry as key
        stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, string_pitch))

    # Re-order keys starting from lowest string and switch to the corresponding note label
    stacked_notes = {librosa.midi_to_note(i) : stacked_notes[i] for i in sorted(stacked_notes.keys())}

    return stacked_notes


class SynthTab(TranscriptionDataset):
    """
    Implements a wrapper for SynthTab (https://synthtab.dev).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                       profile=None, num_frames=None, audio_norm=np.inf, reset_data=False, store_data=True,
                       save_data=True, save_loc=None, guitars=None, sample_attempts=1, augment_audio=False,
                       include_onsets=False, seed=0):
        """
        Initialize an instance of the SynthTab dataset.

        Parameters
        ----------
        See TranscriptionDataset class for others...

        guitars : list of string or None (Optional)
          Names of guitars to include in this instance
        sample_attempts : int (>= 1)
          Number of attempts to sample non-silence
        augment_audio : bool
          Whether to combine and augment the separate microphone signals
        include_onsets : bool
          Whether to include onset activations within the ground-truth
        """

        self.guitars = guitars
        self.sample_attempts = max(1, sample_attempts)
        self.augment_audio = augment_audio
        self.include_onsets = include_onsets

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
                tracks += [os.path.join(split, song, guitar) for guitar in dirs
                           if self.guitars is None or guitar in self.guitars]

        return tracks

    def get_track_data(self, track, seq_length=None):
        """
        Get the features and ground truth for a track.

        Parameters
        ----------
        track : string
          SynthTab track name
        seq_length : int
          Number of samples to take for the slice

        Returns
        ----------
        data : dict
          Dictionary containing for the track
        """

        # Check if a specific sequence length was given
        if seq_length is None:
            if self.seq_length is not None:
                # Use the global sequence length
                seq_length = self.seq_length

        # Determine the expected path to the track's audio
        audio_path = self.get_feats_dir(track)

        # Empty audio variable to populate
        audio = None

        try:
            # Check if an entry for the audio exists
            if self.save_data and os.path.exists(audio_path):
                # Load and unpack the audio
                audio = torch.load(audio_path)
        except Exception as e:
            # Print offending track to console and regenerate audio
            print(f'Error loading audio for track \'{track}\': {repr(e)}')

        if audio is None:
            # Construct the paths to the track's audio
            audio_paths = self.get_audio_paths(track)

            # Initialize a list to hold all microphone signals
            audio = list()

            for path in audio_paths:
                # Load and normalize the audio
                audio_, fs_ = torchaudio.load(path)
                # Extract the first channel
                audio_ = audio_[0].unsqueeze(0)
                # Resample audio to appropriate sampling rate
                audio_ = torchaudio.functional.resample(audio_, fs_, self.sample_rate)

                if self.audio_norm == np.inf or self.audio_norm == torch.inf:
                    # Normalize the audio to the range [-1, 1]
                    audio_ /= audio_.max()
                else:
                    # TODO
                    return NotImplementedError

                # Add the microphone signal to the list
                audio += [audio_]

            # Concatenate microphone signals
            audio = torch.cat(audio)

            if self.save_data:
                # Make sure the top-level pre-processed audio directory exists
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                # Save the pre-processed audio
                torch.save(audio, audio_path)

        # Determine the expected path to the track's ground-truth
        gt_path = self.get_gt_dir(track)

        # Empty ground-truth variable to populate
        stacked_multi_pitch = None

        try:
            # Check if an entry for the ground-truth exists
            if self.save_data and os.path.exists(gt_path):
                # Load and unpack the ground-truth
                ground_truth = tools.load_dict_npz(gt_path)
                # Extract the string-level multi-pitch activations
                stacked_multi_pitch = ground_truth[tools.KEY_MULTIPITCH]

                if self.include_onsets:
                    # Extract the string-level onset activations
                    stacked_onsets = ground_truth[tools.KEY_ONSETS]
        except Exception as e:
            # Print offending track to console and regenerate ground-truth
            print(f'Error loading ground-truth for track \'{track}\': {repr(e)}')

        if stacked_multi_pitch is None:
            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the notes by string from the JAMS file
            stacked_notes = load_stacked_notes_jams(jams_path)

            # Determine the times associated with each frame of audio
            times = self.data_proc.get_times(audio[0])

            # Represent the string-wise notes as a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            if self.include_onsets:
                # Obtain onset activations at the string-level for all of the notes
                stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile)

            if self.save_data:
                # Make sure the top-level ground-truth directory exists
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                # Add the ground-truth to a dictionary that will be saved
                save_data = {tools.KEY_MULTIPITCH : stacked_multi_pitch}

                if self.include_onsets:
                    # Add the onsets to the dictionary
                    save_data.update({tools.KEY_ONSETS : stacked_onsets})

                # Save the pre-computed ground-truth
                tools.save_dict_npz(gt_path, save_data)

        if seq_length is not None:
            # Determine how many audio samples are available
            audio_length = audio[0].shape[-1]

            if audio_length >= seq_length:
                # Initialize a counter for attempts
                attempts_left = self.sample_attempts

                while attempts_left > 0:
                    # Sample a random starting index for the trim
                    sample_start = self.rng.randint(0, audio_length - seq_length + 1)
                    # Determine the frames contained in this slice
                    frame_start = sample_start // self.hop_length
                    frame_end = frame_start + self.num_frames

                    if np.max(stacked_multi_pitch[..., frame_start : frame_end]) > 0:
                        # Non-silence was sampled
                        attempts_left = 0
                    else:
                        # Make another attempt to sample non-silence
                        attempts_left -= 1

                # Trim all microphone signals to the appropriate sequence length
                audio = torch.cat([a[sample_start: sample_start + seq_length].unsqueeze(0) for a in audio])
                # Trim the ground-truth multi-pitch activations to the corresponding frames
                stacked_multi_pitch = stacked_multi_pitch[..., frame_start: frame_end]

                if self.include_onsets:
                    # Trim the ground-truth onset activations to the corresponding frames
                    stacked_onsets = stacked_onsets[..., frame_start: frame_end]
            else:
                # Determine how much padding is required
                pad_total = seq_length - audio_length
                # Pad all microphones with zeros to meet appropriate sequence length
                audio = torch.cat([torch.nn.functional.pad(a, (0, pad_total)).unsqueeze(0) for a in audio])
                # Determine how many frames are missing from the ground-truth
                frames_missing = self.num_frames - stacked_multi_pitch.shape[-1]
                # Pad the ground-truth multi-pitch activations to the corresponding number of frames
                stacked_multi_pitch = np.pad(stacked_multi_pitch, ((0, 0), (0, 0), (0, frames_missing)))

                if self.include_onsets:
                    # Pad the ground-truth onset activations to the corresponding number of frames
                    stacked_onsets = np.pad(stacked_onsets, ((0, 0), (0, 0), (0, frames_missing)))

        if self.augment_audio:
            # Augment and mix the separate microphone signals
            audio = process_audio_signals(audio, seq_length)
        else:
            # Randomly sample one of the microphone signals
            audio = audio[self.rng.randint(0, audio.shape[0])].unsqueeze(0)

        # Compute features for the sampled audio snippet
        features = self.data_proc.process_audio(audio.squeeze().numpy())

        # Convert the stacked multi pitch array into tablature
        tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

        # Convert the stacked multi pitch array into a single representation
        multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

        # Add all relevant ground-truth to the dictionary
        data = {tools.KEY_TRACK: track,
                tools.KEY_FS: self.sample_rate,
                tools.KEY_AUDIO: audio,
                tools.KEY_FEATS: features,
                tools.KEY_TABLATURE: tablature,
                tools.KEY_MULTIPITCH: multi_pitch}

        if self.include_onsets:
            # Add the onsets to the ground-truth dictionary
            data.update({tools.KEY_ONSETS : stacked_onsets})

        return data

    def get_audio_paths(self, track):
        """
        Get the paths to the microphone signals of a track.

        Parameters
        ----------
        track : string
          SynthTab track name

        Returns
        ----------
        audio_paths : list of string
          Paths to the various microphone recordings for the track
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

    def get_feats_dir(self, track=None):
        """
        Get the path for the features directory or a track's features.

        Parameters
        ----------
        track : string or None
          Append a track to the directory for the track's features path

        Returns
        ----------
        path : string
          Path to the features directory or a specific track's features
        """

        # Get the path to the directory holding the pre-processed audio
        path = os.path.join(self.save_loc, self.dataset_name(), 'audio')

        if track is not None:
            # Append track name to path if provided
            path = os.path.join(path, f'{track}.npz')

        return path

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset for training/validation stage
        """

        splits = ['train', 'val']

        return splits

    @staticmethod
    def download(save_dir):
        """
        TODO
        """

        return NotImplementedError
