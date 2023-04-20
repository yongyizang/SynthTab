# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import TranscriptionDataset

import amt_tools.tools as tools

# Regular imports
import os


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
        TODO
        """

        # TODO
        tracks = None

        return tracks

    def load(self, track):
        """
        TODO
        """

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = super().load(track)

        # If the track data is being instantiated, it will not have the 'audio' key
        if not tools.query_dict(data, tools.KEY_AUDIO):
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            audio, fs = tools.load_normalize_audio(wav_path,
                                                   fs=self.sample_rate,
                                                   norm=self.audio_norm)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(audio)

            # TODO - obtain annotations

            # Add all relevant ground-truth to the dictionary
            data.update({tools.KEY_FS : fs,
                         tools.KEY_AUDIO : audio,
                         tools.KEY_TABLATURE : tablature,
                         tools.KEY_MULTIPITCH : multi_pitch})

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

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

        # Get the path to the audio
        # TODO
        wav_path = None

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

        # Get the path to the annotations
        # TODO
        xml_path = None

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

        splits = ['isolated', 'licks', 'pieces']

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
        tools.change_base_dir(save_dir, 'IDMT-SMT-GUITAR_V2')
