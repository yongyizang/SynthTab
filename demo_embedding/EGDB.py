from amt_tools.datasets import TranscriptionDataset

import amt_tools.tools as tools

# Regular imports
import numpy as np
import librosa
import gdown
import mido
import os
import random


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

    # Open the MIDI file
    midi = mido.MidiFile(midi_path)

    # Initialize a counter for the time
    time = 0

    # Initialize an empty list to store MIDI events
    events = []

    # Parse all MIDI messages
    for message in midi:
        # Increment the time
        time += message.time

        # Check if message is a note event (NOTE_ON or NOTE_OFF)
        if 'note' in message.type:
            # Determine corresponding string index
            string_idx = 5 - message.channel
            # MIDI offsets can be either NOTE_OFF events or NOTE_ON with zero velocity
            onset = message.velocity > 0 if message.type == 'note_on' else False

            # Create a new event detailing the note
            event = dict(time=time,
                         pitch=message.note,
                         onset=onset,
                         string=string_idx)
            # Add note event to MIDI event list
            events.append(event)

    # Loop through all tracked MIDI events
    for i, event in enumerate(events):
        # Ignore note offset events
        if not event['onset']:
            continue

        # Extract note attributes
        pitch = event['pitch']
        onset = event['time']
        string_idx = event['string']

        # Determine where the corresponding offset occurs by finding the next note event
        # with the same string, clipping at the final frame if no correspondence is found
        offset = next(n for n in events[i + 1:] if n['string'] == event['string'] or n is events[-1])['time']

        # Obtain the current collection of pitches and intervals
        pitches, intervals = stacked_notes.pop(open_tuning[string_idx])

        # Append the (nominal) note pitch
        pitches = np.append(pitches, pitch)
        # Append the note interval
        intervals = np.append(intervals, [[onset, offset]], axis=0)

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

    def get_tracks(self, _split):
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
        if len(_split.split('_')) == 2:
            type, set = _split.split('_')
            tracks = [os.path.join(type, str(t + 1)) for t in range(240)]
            rng1 = random.Random(505)
            rng1.shuffle(tracks)

            if set == 'train':
                tracks = tracks[:-48]
                tracks.remove('DI/165') # skip the only one shorter than 500 frames

            elif set == 'val':
                tracks = tracks[-48:-24]
            elif set == 'test':
                tracks = tracks[-24:]
                tracks.remove('DI/104')
                tracks.remove('DI/105')
                tracks.remove('DI/166')
        else:
            tracks = [os.path.join(_split, str(t + 1)) for t in range(240)]

        print(tracks)

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

        splits = ['DI', 'Ftwin', 'JCjazz', 'Marshall', 'Mesa', 'Plexi', 'DI_train', 'DI_val', 'DI_test']

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

        # URL pointing to the top-level Google Drive folder
        url = 'https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_'

        # Download the entire Google Drive folder
        gdown.download_folder(url, output=save_dir, quiet=True, use_cookies=False, remaining_ok=True)

        # Move contents of downloaded folder to the base directory
        tools.change_base_dir(save_dir, os.path.join(save_dir, 'EGDB'))
