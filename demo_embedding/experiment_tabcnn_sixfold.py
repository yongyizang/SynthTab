# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT
from train import train

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               MultipitchEvaluator, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate, \
                               append_results, \
                               average_results

import amt_tools.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from datetime import datetime
from sacred import Experiment

import numpy as np
import torch
import os


DEBUG = 0 # (0 - remote | 1 - desktop)
RECURRENT = 0 # (0 - no recurrence | 1 - recurrence)
LOGISTIC = 0 # (0 - softmax output layer | 1 - logistic output layer)

if RECURRENT and LOGISTIC:
    from guitar_transcription_inhibition.models import TabCNNLogisticRecurrent as TabCNN
elif RECURRENT:
    from guitar_transcription_inhibition.models import TabCNNRecurrent as TabCNN
elif LOGISTIC:
    from guitar_transcription_inhibition.models import TabCNNLogistic as TabCNN
else:
    from amt_tools.models import TabCNN

EX_NAME = '_'.join([TabCNN.model_name(),
                    GuitarSet.dataset_name(),
                    CQT.features_name(),
                    datetime.now().strftime("%m-%d-%Y@%H:%M")])

ex = Experiment('6-fold Cross Validation for TabCNN w/ CQT on GuitarSet')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 500

    # Number of epochs
    epochs = 1000

    # Number batches in between checkpoints
    checkpoints = 100

    # Number of samples to gather for a batch
    batch_size = 15

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate
    # features (useful if testing out different parameters)
    reset_data = False

    # Multiplier for inhibition loss if applicable
    lmbda = 10

    # Path to inhibition matrix if applicable
    matrix_path = None

    # The random seed for this experiment
    seed = 0

    # Path to a pre-trained model checkpoint to load each for fold
    pre_trained_path = None

    # Number of threads to use for data loading
    n_workers = 0 if DEBUG else 5

    # Create the root directory for the experiment files
    if DEBUG:
        root_dir = os.path.join('.', 'generated', 'experiments', EX_NAME)
    else:
        root_dir = os.path.join('/', 'home', 'frank', 'experiments', EX_NAME)

    # Make sure the directory exists
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def tabcnn_cross_val(sample_rate, hop_length, num_frames, epochs, checkpoints,
                     batch_size, gpu_id, reset_data, lmbda, matrix_path, seed,
                     pre_trained_path, n_workers, root_dir):
    # Initialize the default guitar profile
    profile = tools.GuitarProfile(num_frets=19)

    # Create a CQT feature extraction module
    # spanning 8 octaves w/ 2 bins per semitone
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=192,
                    bins_per_octave=24)

    # Initialize the estimation pipeline (Tablature -> Stacked Multi Pitch -> Multi Pitch)
    validation_estimator = ComboEstimator([TablatureWrapper(profile=profile),
                                           StackedMultiPitchCollapser(profile=profile)])

    # Initialize the evaluation pipeline (Loss | Multi Pitch | Tablature)
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           TablatureEvaluator(profile=profile),
                                           SoftmaxAccuracy()])

    if DEBUG:
        # Point to the default location of dataset
        gset_base_dir = None

        # Keep all cached data/features here
        gset_cache = os.path.join('.', 'generated', 'data')
    else:
        # Navigate to the location of the full data
        gset_base_dir = os.path.join('/', 'media', 'finch', 'SSD2', 'GuitarSet')

        # Keep all cached data/features here
        gset_cache = os.path.join('/', 'media', 'finch', 'SSD2', 'precomputed')

    # Initialize an empty dictionary to hold the average results across folds
    results = dict()

    # Perform each fold of cross-validation
    for k in range(6):
        # Seed everything with the same seed
        tools.seed_everything(seed)

        # Set validation patterns for logging during training
        validation_evaluator.set_patterns(['loss', 'pr', 're', 'f1', 'tdr', 'acc'])

        # Allocate training/validation/testing splits
        train_splits = GuitarSet.available_splits()
        val_splits = [train_splits.pop(k - 1)]
        test_splits = [train_splits.pop(k)]

        print('Loading training partition...')

        # Instantiate the GuitarSet training partition
        gset_train = GuitarSet(base_dir=gset_base_dir,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=num_frames,
                               audio_norm=np.inf,
                               data_proc=data_proc,
                               profile=profile,
                               store_data=True,
                               save_data=True,
                               reset_data=(reset_data and k == 0),
                               save_loc=gset_cache)

        # Create a PyTorch data loader for the dataset
        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers,
                                  drop_last=True)

        print(f'Loading validation partition (player {val_splits[0]})...')

        # Instantiate the GuitarSet validation partition
        gset_val = GuitarSet(base_dir=gset_base_dir,
                             splits=val_splits,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             num_frames=num_frames,
                             audio_norm=np.inf,
                             data_proc=data_proc,
                             profile=profile,
                             store_data=True,
                             save_data=True,
                             reset_data=reset_data,
                             save_loc=gset_cache)

        print(f'Loading testing partition (player {test_splits[0]})...')

        # Instantiate the GuitarSet testing partition
        gset_test = GuitarSet(base_dir=gset_base_dir,
                              splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              audio_norm=np.inf,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=False,
                              save_data=True,
                              save_loc=gset_cache)

        print('Initializing model...')

        if pre_trained_path is not None:
            # Load the pre-trained model
            tabcnn = torch.load(pre_trained_path, map_location='cpu')
        else:
            # Initialize a new instance of the model
            if LOGISTIC:
                tabcnn = TabCNN(dim_in=data_proc.get_feature_size(),
                                profile=profile,
                                in_channels=data_proc.get_num_channels(),
                                matrix_path=matrix_path,
                                silence_activations=True,
                                lmbda=lmbda,
                                device=gpu_id)
            else:
                tabcnn = TabCNN(dim_in=data_proc.get_feature_size(),
                                profile=profile,
                                in_channels=data_proc.get_num_channels(),
                                device=gpu_id)
        tabcnn.change_device(gpu_id)
        tabcnn.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adadelta(tabcnn.parameters(), lr=1.0)

        print('Training model...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Enter the training loop
        tabcnn = train(model=tabcnn,
                       train_loader=train_loader,
                       optimizer=optimizer,
                       epochs=epochs,
                       checkpoints=checkpoints,
                       log_dir=model_dir,
                       val_set=gset_val,
                       estimator=validation_estimator,
                       evaluator=validation_evaluator)

        print(f'Transcribing and evaluating test partition (player {test_splits[0]})...')

        # Add a save directory to the evaluators
        validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
        # Reset the evaluation patterns to log everything
        validation_evaluator.set_patterns(None)

        # Compute the average results for the fold
        fold_results = validate(tabcnn, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

        # Add the results to the tracked fold results
        results = append_results(results, fold_results)

        # Reset the results for the next fold
        validation_evaluator.reset_results()

        # Log the average results for the fold in metrics.json
        ex.log_scalar('Fold Results', fold_results, k)

    # Log the average results across all folds in metrics.json
    ex.log_scalar('Overall Results', average_results(results), 0)
