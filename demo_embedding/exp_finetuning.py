from IDMT_SMT_Guitar import IDMT_SMT_Guitar
from EGDB import EGDB
from GuitarSet import GuitarSet

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
    # from amt_tools.models import TabCNN
    from tabcnn import TabCNNx4, TabCNN

FINETUNE_DATASET = 'GuitarSet' # valid: GuitarSet, EGDB, IDMT

EX_NAME = '_'.join([TabCNN.model_name(),
                    FINETUNE_DATASET,
                    CQT.features_name(),
                    datetime.now().strftime("%m-%d-%Y@%H:%M")])

ex = Experiment('Baseline for TabCNN w/ CQT on ' + FINETUNE_DATASET)


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 250 if FINETUNE_DATASET == 'IDMT' else 500

    # Number of epochs
    epochs = 1000

    # Number batches in between checkpoints
    checkpoints = 100

    # Number of samples to gather for a batch
    batch_size = 8

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

    pre_trained_path = None
    assert pre_trained_path is not None, 'Please specify the path to the model to use as the pretrained weights'
    # Number of threads to use for data loading
    n_workers = 0 if DEBUG else 5

    # Create the root directory for the experiment files
    if DEBUG:
        root_dir = os.path.join('.', 'generated', 'experiments', EX_NAME)
    else:
        root_dir = os.path.join('.', 'generated_baseline', 'experiments', EX_NAME)

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
        gset_base_dir = os.path.join('/home/finch/external_ssd', 'GuitarSet')
        idmt_base_dir = os.path.join('/home/finch/external_ssd', 'IDMT-SMT-GUITAR-dataset')
        egdb_base_dir = os.path.join('/home/finch/external_ssd', 'EGDB')

        # Keep all cached data/features here
        cache_dir = os.path.join('/home/finch/external_ssd/guitar-tab/EXPS', 'precomputed')
    # Initialize an empty dictionary to hold the average results across folds
    results = dict()

    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Set validation patterns for logging during training
    validation_evaluator.set_patterns(['loss', 'pr', 're', 'f1', 'tdr', 'acc'])

    # Allocate training/validation/testing splits


    if FINETUNE_DATASET == 'GuitarSet':
    # Instantiate the GuitarSet training partition
        print('Loading training partition... GuitarSet')
        train_splits = GuitarSet.available_splits()
        val_splits = [train_splits.pop()]
        test_splits = [train_splits.pop()]
        trainset = GuitarSet(base_dir=gset_base_dir,
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
                               save_loc=cache_dir)



        print(f'Loading validation partition (player {val_splits[0]})...')

        # Instantiate the GuitarSet validation partition
        valset = GuitarSet(base_dir=gset_base_dir,
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
                             save_loc=cache_dir)

        print(f'Loading testing partition (player {test_splits[0]})...')

        # Instantiate the GuitarSet testing partition
        testset = GuitarSet(base_dir=gset_base_dir,
                              splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              audio_norm=np.inf,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=False,
                              save_data=True,
                              save_loc=cache_dir)

    elif FINETUNE_DATASET == 'EGDB':

        print('Loading training partition... EGDB')

        trainset = EGDB(base_dir=egdb_base_dir,
                          splits=['DI_train'],
                          hop_length=hop_length,
                          sample_rate=sample_rate,
                          num_frames=num_frames,
                          data_proc=data_proc,
                          profile=profile,
                          reset_data=reset_data,
                          save_loc=cache_dir,
                          seed=seed,
                          )



        valset = EGDB(base_dir=egdb_base_dir,
                        splits=['DI_val'],
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        num_frames=None,
                        data_proc=data_proc,
                        profile=profile,
                        store_data=False,
                        reset_data=reset_data,
                        save_loc=cache_dir,
                        seed=seed)

        testset = EGDB(base_dir=egdb_base_dir,
                         splits=['DI_test'],
                         hop_length=hop_length,
                         sample_rate=sample_rate,
                         num_frames=None,
                         data_proc=data_proc,
                         profile=profile,
                         store_data=False,
                         reset_data=reset_data,
                         save_loc=cache_dir,
                         seed=seed)


    elif FINETUNE_DATASET == 'IDMT':
        print('Loading training partition... IDMT')

        trainset = IDMT_SMT_Guitar(base_dir=idmt_base_dir,
                        splits=['licks_train'],
                        hop_length=hop_length,
                        sample_rate=sample_rate,
                        num_frames=num_frames,
                        data_proc=data_proc,
                        profile=profile,
                        reset_data=reset_data,
                        save_loc=cache_dir,
                        seed=seed,
                        )

        valset = IDMT_SMT_Guitar(base_dir=idmt_base_dir,
                                   splits=['licks_val'],
                                   hop_length=hop_length,
                                   sample_rate=sample_rate,
                                   num_frames=None,
                                   data_proc=data_proc,
                                   profile=profile,
                                   store_data=False,
                                   reset_data=reset_data,
                                   save_loc=cache_dir,
                                   seed=seed)

        testset = IDMT_SMT_Guitar(base_dir=idmt_base_dir,
                                    splits=['licks_test'],
                                    hop_length=hop_length,
                                    sample_rate=sample_rate,
                                    num_frames=None,
                                    data_proc=data_proc,
                                    profile=profile,
                                    store_data=False,
                                    reset_data=reset_data,
                                    save_loc=cache_dir,
                                    seed=seed)

    print('Initializing model...')
    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              drop_last=True)


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
                            model_complexity=4,
                            in_channels=data_proc.get_num_channels(),
                            device=gpu_id)
    tabcnn.change_device(gpu_id)
    tabcnn.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adadelta(tabcnn.parameters(), lr=1.0)

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
                   evaluator=validation_evaluator)

    print(f'Transcribing and evaluating test partition (player {test_splits[0]})...')

    # Add a save directory to the evaluators
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    # Reset the evaluation patterns to log everything
    validation_evaluator.set_patterns(None)

    # Compute the average results for the fold
    results = validate(tabcnn, testset, evaluator=validation_evaluator, estimator=validation_estimator)

    # Reset the results for the next fold
    validation_evaluator.reset_results()

    # Log the results for the fold in metrics.json
    ex.log_scalar('Overall Results', results, 0)
    #
    # # Log the average results across all folds in metrics.json
    # ex.log_scalar('Overall Results', average_results(results), 0)
