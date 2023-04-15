# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import InhibitionMatrixTrainer
from amt_tools.features import CQT
from SynthTab import SynthTab

import amt_tools.tools as tools

# Regular imports
from sacred import Experiment

import os


DEBUG = 0 # (0 - remote | 1 - desktop)

ex = Experiment('Obtain an inhibition matrix for the SynthTab training set')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of frets supported
    num_frets = 22

    # Flag to include silence associations
    silence_activations = True

    # Select the power for boosting
    boost = 128


@ex.automain
def train_matrix(sample_rate, hop_length, num_frets, silence_activations, boost):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=num_frets)

    if DEBUG:
        # Point to the default location of dataset
        synthtab_base_dir = os.path.join('..', 'demo_data')

        # Keep all cached data/features here
        cache_dir = os.path.join('.', 'generated', 'data')

        # Construct a path for saving the inhibition matrix
        save_path = os.path.join('.', 'generated', 'matrices', f'synthtab_train_p{boost}.npz')
    else:
        # Navigate to the location of the full data
        synthtab_base_dir = os.path.join('/', 'media', 'finch', 'SSD2', 'SynthTab')

        # Keep all cached data/features here
        cache_dir = os.path.join('/', 'media', 'finch', 'SSD2', 'precomputed')

        # Construct a path for saving the inhibition matrix
        save_path = os.path.join('/', 'media', 'finch', 'SSD2', f'synthtab_train_p{boost}.npz')

    # Create a CQT feature extraction module
    # spanning 8 octaves w/ 2 bins per semitone
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=192,
                    bins_per_octave=24)

    # Instantiate the SynthTab training partition
    synthtab_train = SynthTab(base_dir=synthtab_base_dir,
                              splits=['train'],
                              guitars=['luthier'],
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              data_proc=data_proc,
                              profile=profile,
                              reset_data=False,
                              store_data=False,
                              save_data=True,
                              save_loc=cache_dir)

    # Obtain an inhibition matrix from the DadaGP data
    InhibitionMatrixTrainer(profile=profile,
                            silence_activations=silence_activations,
                            boost=boost,
                            save_path=save_path).train(synthtab_train, residual_threshold=None)
