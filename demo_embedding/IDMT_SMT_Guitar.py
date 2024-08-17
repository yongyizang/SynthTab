
from amt_tools.datasets import TranscriptionDataset

import amt_tools.tools as tools

# Regular imports
import numpy as np
import xmltodict
import warnings
import librosa
import os


def load_stacked_notes_xml(xml_path, absolute=False):
    """
    Extract MIDI notes spread across strings into a dictionary
    from an XML file following the IDMT-SMT-Guitar format.

    Parameters
    ----------
    xml_path : string
      Path to XML file to read
    absolute : bool
      Whether to use absolute pitch annotation instead of pitch implied by string and fret

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    with open(xml_path) as xml_file:
        # Load the data from the XML file
        xml = xmltodict.parse(xml_file.read())['instrumentRecording']

    if 'instrumentTuning' in xml['globalParameter']:
        # Obtain the open string tuning for the recording
        open_tuning = [int(p) for p in xml['globalParameter']['instrumentTuning'].replace(',', ' ').split()]
    else:
        # Assume standard tuning by default
        open_tuning = list(librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING))

    # Initialize a dictionary to hold the notes for each string
    stacked_notes = [tools.notes_to_stacked_notes([], [], p) for p in open_tuning]
    stacked_notes = {k : v for d in stacked_notes for k, v in d.items()}

    # Extract the notes from the XML
    notes = xml['transcription']['event']

    if isinstance(notes, dict):
        # Add single note to list
        notes = [notes]

    # Loop through notes
    for n in notes:
        # Extract relevant note attributes
        string_idx = int(n['stringNumber']) - 1
        fret = int(n['fretNumber'])
        pitch = int(n['pitch'])
        onset = float(n['onsetSec'])
        offset = float(n['offsetSec'])

        if pitch != open_tuning[string_idx] + fret:
            # Print an appropriate warning to console
            warnings.warn('Note pitch not equal to nominal pitch of '
                          'corresponding string and fret.', RuntimeWarning)
            # Print offending file name
            print(f'File: {os.path.basename(xml_path)} | note {n}')

        # Obtain the current collection of pitches and intervals
        pitches, intervals = stacked_notes.pop(open_tuning[string_idx])

        if absolute:
            # Append the absolute note pitch
            pitches = np.append(pitches, pitch)
        else:
            # Append the nominal note pitch of the string and fret
            pitches = np.append(pitches, open_tuning[string_idx] + fret)

        # Append the note interval
        intervals = np.append(intervals, [[onset, offset]], axis=0)

        # Re-insert the pitch-interval pairs into the stacked notes dictionary under the appropriate key
        stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, open_tuning[string_idx]))

    # Re-order keys starting from lowest string and switch to the corresponding note label
    stacked_notes = {librosa.midi_to_note(i) : stacked_notes[i] for i in sorted(stacked_notes.keys())}

    return stacked_notes


class IDMT_SMT_Guitar(TranscriptionDataset):
    """
    A wrapper for the IDMT-SMT-Guitar dataset (https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                       profile=None, num_frames=None, audio_norm=-1, reset_data=False, store_data=True,
                       save_data=True, save_loc=None, seed=0):
        """
        Initialize an instance of the IDMT_SMT_Guitar dataset.

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

        # Define paths to all dataset section directories
        isol_dir = os.path.join(self.base_dir, 'dataset1')
        lick_dir = os.path.join(self.base_dir, 'dataset2')
        piece_dir = os.path.join(self.base_dir, 'dataset3')


        if split == 'isolated_clean':
            # Start with all tracks under the lick directory
            tracks = os.listdir(os.path.join(lick_dir, 'audio'))
            # Keep only the clean note recordings
            tracks = [os.path.join('dataset2', t) for t in tracks if 'fret_0-20' in t]
            # Loop through all guitars
            for dir in os.listdir(isol_dir):
                # Construct a path to the audio for the guitar
                audio_dir = os.path.join(isol_dir, dir, 'audio')
                # Add all tracks under the audio directory
                tracks += [os.path.join('dataset1', dir, t)
                           for t in os.listdir(audio_dir)]
        elif split == 'isolated_techniques':
            # Start with all tracks under the lick directory
            tracks = os.listdir(os.path.join(lick_dir, 'audio'))
            # Remove licks and recordings of clean notes
            tracks = [os.path.join('dataset2', t) for t in tracks
                      if 'fret_0-20' not in t and 'Lick' not in t]

        # add splits for 'licks_train', 'licks_val', 'licks_test'
        elif split == 'licks_train':
            # Start with all tracks under the lick directory
            tracks = os.listdir(os.path.join(lick_dir, 'audio'))
            # Keep only the licks
            tracks = [os.path.join('dataset2', t) for t in tracks if (('Lick' in t) and ('Lick8' not in t) and ('Lick10' not in t))]
        elif split == 'licks_val':
            tracks = os.listdir(os.path.join(lick_dir, 'audio'))
            # Keep only the licks
            tracks = [os.path.join('dataset2', t) for t in tracks if
                      'Lick10' in t]
        elif split == 'licks_test':
            tracks = os.listdir(os.path.join(lick_dir, 'audio'))
            # Keep only the licks
            tracks = [os.path.join('dataset2', t) for t in tracks if
                      'Lick8' in t]

        elif split == 'pieces':
            # Obtain all tracks under the piece directory
            tracks = [os.path.join('dataset3', t)
                      for t in os.listdir(os.path.join(piece_dir, 'audio'))]
            tracks.remove('dataset3/nocturneNr2.wav')


        # Remove the .wav extension and sort the track names
        tracks = sorted([os.path.splitext(t)[0] for t in tracks])

        return tracks

    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          IDMT-SMT-Guitar track name

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
            xml_path = self.get_xml_path(track)

            # Load the notes by string from the XML file (nominal)
            stacked_notes_tab = load_stacked_notes_xml(xml_path)

            # Represent the string-wise notes with nominal pitches as a stacked multi pitch array
            stacked_multi_pitch_tab = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes_tab, times, self.profile)

            # Convert the stacked multi pitch array into tablature
            tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch_tab, self.profile)

            # Load the notes by string from the XML file (absolute_
            stacked_notes_mpe = load_stacked_notes_xml(xml_path, True)

            # Represent the string-wise notes with absolute pitches as a stacked multi pitch array
            stacked_multi_pitch_mpe = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes_mpe, times, self.profile)

            # Convert the stacked multi pitch array into a single representation
            multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch_mpe)

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
          IDMT-SMT-Guitar track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Break apart the track in order to reconstruct the path
        section_dir, file_name = os.path.dirname(track), os.path.basename(track)

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, section_dir, 'audio', f'{file_name}.wav')

        return wav_path

    def get_xml_path(self, track):
        """
        Get the path for a track's annotations.

        Parameters
        ----------
        track : string
          IDMT-SMT-Guitar track name

        Returns
        ----------
        xml_path : string
          Path to the specified track's annotations
        """

        # Break apart the track in order to reconstruct the path
        section_dir, file_name = os.path.dirname(track), os.path.basename(track)

        if 'Lick11' in file_name:
            # Correct apparent naming error
            file_name += 'VSBHD'

        # Get the path to the XML annotations
        xml_path = os.path.join(self.base_dir, section_dir, 'annotation', f'{file_name}.xml')

        return xml_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits.

        Returns
        ----------
        splits : list of strings
          Different sections of dataset
        """

        splits = ['isolated_clean', 'isolated_techniques', 'licks', 'pieces', 'licks_train', 'licks_val', 'licks_test']

        return splits

    @classmethod
    def dataset_name(cls):
        """
        Obtain a string representing the dataset.

        Returns
        ----------
        tag : str
          Dataset name with dashes
        """

        # Obtain class name and replace underscores with dashes
        tag = super().dataset_name().replace('_', '-')

        return tag

    @staticmethod
    def download(save_dir):
        """
        Download the IDMT-SMT-Guitar dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of IDMT-SMT-Guitar
        """

        # Create top-level directory
        TranscriptionDataset.download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        url = 'https://zenodo.org/record/7544110/files/IDMT-SMT-GUITAR_V2.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        tools.stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        tools.unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        tools.change_base_dir(save_dir, os.path.join(save_dir, 'IDMT-SMT-GUITAR_V2'))


if __name__ == '__main__':
    tmp = IDMT_SMT_Guitar(base_dir='/Users/yizhong/Documents/gits/guitar-tab/IDMT-SMT-GUITAR-dataset')
    # IDMT_SMT_Guitar.download('/Users/yizhong/Documents/gits/guitar-tab/IDMT-SMT-GUITAR-dataset')