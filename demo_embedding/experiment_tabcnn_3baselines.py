# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
import pdb

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT
from IDMT_SMT_Guitar import IDMT_SMT_Guitar
from EGDB import EGDB
from train import train
# from guitar_transcription_continuous.datasets import GuitarSetPlus as GuitarSet


from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               MultipitchEvaluator, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate

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
                    CQT.features_name(),
                    datetime.now().strftime("%m-%d-%Y@%H:%M")])

ex = Experiment('Train TabCNN w/ CQT on 3 Baseline datasets and Evaluate on GuitarSet')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 250

    # Number of epochs
    epochs = 1000

    # Number batches in between checkpoints
    checkpoints = 250

    # Number of samples to gather for a batch
    batch_size = 32

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate
    # features (useful if testing out different parameters)
    reset_data = False

    # Flag to augment audio during training
    augment = False

    # Multiplier for inhibition loss if applicable
    lmbda = 10

    # Path to inhibition matrix if applicable
    matrix_path = None

    # The random seed for this experiment
    seed = 505

    # Number of threads to use for data loading
    n_workers = 0 if DEBUG else 12

    # Create the root directory for the experiment files
    if DEBUG:
        root_dir = os.path.join('.', 'generated', 'experiments', EX_NAME)
    else:
        root_dir = os.path.join('.', 'generated', 'experiments', EX_NAME)

        # root_dir = os.path.join('/home/finch/terryssd/guitar-tab/EXPS/', EX_NAME)

    # Make sure the directory exists
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def synthtab_experiment(sample_rate, hop_length, num_frames, epochs, checkpoints,
                        batch_size, gpu_id, reset_data, augment, lmbda, matrix_path,
                        seed, n_workers, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

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

    # Set validation patterns for logging during training
    validation_evaluator.set_patterns(['loss', 'pr', 're', 'f1', 'tdr', 'acc'])

    if DEBUG:
        # Point to the default location of dataset
        synthtab_base_dir = os.path.join('..', 'demo_data')
        gset_base_dir = None

        # Keep all cached data/features here
        cache_dir = os.path.join('.', 'generated', 'data')
    else:
        # Navigate to the location of the full data
        gset_base_dir = os.path.join('/home/finch/terryssd/guitar-tab/datasets', 'GuitarSet')
        egdb_base_dir = os.path.join('/home/finch/terryssd/guitar-tab/datasets', 'EGDB')
        idmt_base_dir = os.path.join('/home/finch/terryssd/guitar-tab/datasets', 'IDMT-SMT-GUITAR-dataset')

        # Keep all cached data/features here
        cache_dir = os.path.join('//home/finch/terryssd/guitar-tab/EXPS', 'precomputed')



    # Set validation patterns for logging during training
    # Allocate training/validation/testing splits


    print('Loading training partition...')
    # Create a dataset corresponding to the training partition

    train_splits = GuitarSet.available_splits()
    test_splits = [train_splits.pop()]
    val_splits = [train_splits.pop()]

    trainset = \
        GuitarSet(base_dir=gset_base_dir,
                  splits=train_splits,
                  hop_length=hop_length,
                  sample_rate=sample_rate,
                  num_frames=num_frames,
                  data_proc=data_proc,
                  profile=profile,
                  reset_data=reset_data,
                  save_loc=cache_dir,
                  seed=seed,
                  )

        # EGDB(base_dir=egdb_base_dir,
        #      splits=['DI_train'],
        #      hop_length=hop_length,
        #      sample_rate=sample_rate,
        #      num_frames=num_frames,
        #      data_proc=data_proc,
        #      profile=profile,
        #      reset_data=reset_data,
        #      save_loc=cache_dir,
        #      seed=seed,
        #      )

        # IDMT_SMT_Guitar(base_dir=idmt_base_dir,
        #                    splits=['licks_train'],
        #                    hop_length=hop_length,
        #                    sample_rate=sample_rate,
        #                    num_frames=num_frames,
        #                    data_proc=data_proc,
        #                    profile=profile,
        #                    reset_data=reset_data,
        #                    save_loc=cache_dir,
        #                    seed=seed,
        #                    )


    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=n_workers,
                              drop_last=True)


    valset = \
        GuitarSet(base_dir=gset_base_dir,
                  splits=val_splits,
                  hop_length=hop_length,
                  sample_rate=sample_rate,
                  num_frames=None,
                  data_proc=data_proc,
                  profile=profile,
                  store_data=False,
                  reset_data=reset_data,
                  save_loc=cache_dir,
                  seed=seed)

        # EGDB(base_dir=egdb_base_dir,
        #      splits=['DI_val'],
        #      hop_length=hop_length,
        #      sample_rate=sample_rate,
        #      num_frames=None,
        #      data_proc=data_proc,
        #      profile=profile,
        #      store_data=False,
        #      reset_data=reset_data,
        #      save_loc=cache_dir,
        #      seed=seed)

        # IDMT_SMT_Guitar(base_dir=idmt_base_dir,
        #                   splits=['licks_val'],
        #                   hop_length=hop_length,
        #                   sample_rate=sample_rate,
        #                   num_frames=None,
        #                   data_proc=data_proc,
        #                   profile=profile,
        #                   store_data=False,
        #                   reset_data=reset_data,
        #                   save_loc=cache_dir,
        #                   seed=seed)

    testset = \
        GuitarSet(base_dir=gset_base_dir,
                  splits=test_splits,
                  hop_length=hop_length,
                  sample_rate=sample_rate,
                  num_frames=None,
                  data_proc=data_proc,
                  profile=profile,
                  store_data=False,
                  reset_data=reset_data,
                  save_loc=cache_dir,
                  seed=seed)

        # EGDB(base_dir=egdb_base_dir,
        #           splits=['DI_test'],
        #           hop_length=hop_length,
        #           sample_rate=sample_rate,
        #           num_frames=None,
        #           data_proc=data_proc,
        #           profile=profile,
        #           store_data=False,
        #           reset_data=reset_data,
        #           save_loc=cache_dir,
        #           seed=seed)

        # IDMT_SMT_Guitar(base_dir=idmt_base_dir,
        #                   splits=['licks_test'],
        #                   hop_length=hop_length,
        #                   sample_rate=sample_rate,
        #                   num_frames=None,
        #                   data_proc=data_proc,
        #                   profile=profile,
        #                   store_data=False,
        #                   reset_data=reset_data,
        #                   save_loc=cache_dir,
        #                   seed=seed)

    print('Initializing model...')

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
    tabcnn.change_device()
    tabcnn.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adadelta(tabcnn.parameters(), lr=1.0)
    # optimizer = torch.optim.AdamW(tabcnn.parameters(), lr=1e-3, weight_decay=1e-5)
    #
    # # Also adding a lr scheduler
    # train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print('Training model...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Enter the training loop
    tabcnn = train(model=tabcnn,
                   train_loader=train_loader,
                   optimizer=optimizer,
                   epochs=epochs,
                   checkpoints=checkpoints,
                   log_dir=model_dir,
                   val_set=valset,
                   estimator=validation_estimator,
                   evaluator=validation_evaluator,
                   #scheduler = train_scheduler,
                   )

    print(f'Transcribing and evaluating ...')

    # Add a save directory to the evaluators
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    # Reset the evaluation patterns to log everything
    validation_evaluator.set_patterns(None)

    # Compute the average results on GuitarSet
    results = validate(tabcnn, testset, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('Overall Results', results, 0)