# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import librosa
import os


def profile_audio_dir(top_dir, exts=['.wav']):
    """
    Determine the amount of audio contained within a directory.

    Parameters
    ----------
    top_dir : string
      Path to top-level directory to profile
    exts : list of string
      Extensions to search

    Returns
    ----------
    total_time : float
      Total duration of all audio in seconds
    """

    total_time = 0

    # Parse through everything contained within the top-level directory
    for root, dirs, files in os.walk(top_dir):
        # Loop through files
        for f in files:
            # Check if the file has one of the specified extensions
            if any(ext in f for ext in exts):
                # Load the audio file with the original sampling rate
                audio, fs = librosa.load(os.path.join(root, f), sr=None)
                # Accumulate the time of the audio file
                total_time += audio.size / fs

    return total_time
