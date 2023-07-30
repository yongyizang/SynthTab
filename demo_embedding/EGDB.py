# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import TranscriptionDataset

import amt_tools.tools as tools

# Regular imports
import numpy as np
import pretty_midi
import warnings
import librosa
import os


def load_stacked_notes_midi(midi_path):
    """
    Extract MIDI notes spread across strings into a dictionary
    from a MIDI file following the EGDB format.

    Parameters
    ----------
    midi_path : string
      Path to MIDI file to read

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Standard tuning is assumed for all tracks in EGDB
    open_tuning = list(librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING))

    # Initialize a dictionary to hold the notes for each string
    stacked_notes = [tools.notes_to_stacked_notes([], [], p) for p in open_tuning]
    stacked_notes = {k : v for d in stacked_notes for k, v in d.items()}

    # Extract the notes from the MIDI file
    annotations = pretty_midi.PrettyMIDI(midi_path)

    # Loop through all strings with annotations
    for i, string in enumerate(annotations.instruments):
        if string.name.isdigit():
            # Determine the appropriate index
            string_idx = 6 - int(string.name)
        else:
            print(midi_path)
            # Address "6i" bug
            string_idx = 5 - i

        # Determine relevant attributes of each note
        onsets, offsets, pitches = np.array([(n.start, n.end, n.pitch) for n in string.notes]).transpose()

        # Combine onsets and offsets to obtain note intervals
        intervals = np.concatenate(([onsets], [offsets])).T

        # Re-insert the pitch-interval pairs into the stacked notes dictionary under the appropriate key
        stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, open_tuning[string_idx]))

    # Re-order keys starting from lowest string and switch to the corresponding note label
    stacked_notes = {librosa.midi_to_note(i) : stacked_notes[i] for i in sorted(stacked_notes.keys())}

    return stacked_notes


class EGDB(TranscriptionDataset):
    """
    A wrapper for the EGDB dataset (https://ss12f32v.github.io/Guitar-Transcription/).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                       profile=None, num_frames=None, audio_norm=-1, reset_data=False, store_data=True,
                       save_data=True, save_loc=None, seed=0):
        """
        Initialize an instance of the EGDB dataset.

        Parameters
        ----------
        See TranscriptionDataset class...
        """

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames,
                         audio_norm, False, reset_data, store_data, save_data, save_loc, seed)

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Name of dataset split

        Returns
        ----------
        tracks : list of strings
          Tracks pertaining to specified dataset split
        """

        # Naming scheme specifies amplifier with tracks numbered 1-240
        tracks = [os.path.join(split, str(t + 1)) for t in range(240)]

        return tracks

    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          EGDB track name

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = super().load(track)

        # If the track data is being instantiated, it will not have the 'audio' key
        if not tools.query_dict(data, tools.KEY_AUDIO):
            # Obtain the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            audio, fs = tools.load_normalize_audio(wav_path,
                                                   fs=self.sample_rate,
                                                   norm=self.audio_norm)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(audio)

            # Obtain the path to the track's annotations
            midi_path = self.get_midi_path(track)

            # Load the notes by string from the MIDI file
            stacked_notes = load_stacked_notes_midi(midi_path)

            # Represent the string-wise notes as a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Convert the stacked multi pitch array into tablature
            tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Convert the stacked multi pitch array into a single representation
            multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

            # Add all relevant ground-truth to the dictionary
            data.update({tools.KEY_FS : fs,
                         tools.KEY_AUDIO : audio,
                         tools.KEY_TABLATURE : tablature,
                         tools.KEY_MULTIPITCH : multi_pitch})

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)
                # Create the (sub-directory) path if it doesn't exist
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                # Save the data as a NumPy zip file
                tools.save_dict_npz(gt_path, data)

        return data

    def get_wav_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          EGDB track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Break apart the track in order to reconstruct the path
        amp_dir, n = os.path.dirname(track), os.path.basename(track)

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, f'audio_{amp_dir}', f'{n}.wav')

        return wav_path

    def get_midi_path(self, track):
        """
        Get the path for a track's annotations.

        Parameters
        ----------
        track : string
          EGDB track name

        Returns
        ----------
        midi_path : string
          Path to the specified track's annotations
        """

        # Get the path to the MIDI annotations
        midi_path = os.path.join(self.base_dir, 'audio_label', f'{os.path.basename(track)}.midi')

        return midi_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits.

        Returns
        ----------
        splits : list of strings
          Different amplifiers of dataset
        """

        splits = ['DI', 'Ftwin', 'JCjazz', 'Marshall', 'Mesa', 'Plexi']

        return splits

    @staticmethod
    def download(save_dir):
        """
        Download the EGDB dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of EGDB
        """

        # Create top-level directory
        TranscriptionDataset.download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        #url = 'https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_?usp=share_link'

        # Construct a path for saving the file
        #zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        #tools.stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        #tools.unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        #tools.change_base_dir(save_dir, os.path.join(save_dir, 'IDMT-SMT-GUITAR_V2'))

        return NotImplementedError
