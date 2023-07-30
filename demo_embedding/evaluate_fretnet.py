# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from IDMT_SMT_Guitar import IDMT_SMT_Guitar
from EGDB import EGDB
from amt_tools.datasets import GuitarSet
from amt_tools.features import HCQT

from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               MultipitchEvaluator, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate

# Regular imports
import librosa
import torch
import os


# Construct the path to the model to evaluate
model_path = os.path.join('.', 'generated', 'experiments', '<EXPERIMENT_DIR>', 'models', 'model-<CHECKPOINT>.pt')

# Number of samples per second of audio
sample_rate = 22050
# Number of samples between frames
hop_length = 512
# Flag to re-acquire ground-truth data and re-calculate features
reset_data = False
# Choose the GPU on which to perform evaluation
gpu_id = 0

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load the model onto the specified device
model = torch.load(model_path, map_location=device)
model.change_device(device)
model.eval()

# Create an HCQT feature extraction module comprising
# the first five harmonics and a sub-harmonic, where each
# harmonic transform spans 4 octaves w/ 3 bins per semitone
data_proc = HCQT(sample_rate=sample_rate,
                 hop_length=hop_length,
                 fmin=librosa.note_to_hz('E2'),
                 harmonics=[0.5, 1, 2, 3, 4, 5],
                 n_bins=144, bins_per_octave=36)

# Default location of datasets
gset_base_dir = None
idmt_base_dir = None
egdb_base_dir = None

# Keep all cached data/features here
cache_dir = os.path.join('.', 'generated', 'data')

# Instantiate GuitarSet for testing
gset_test = GuitarSet(base_dir=gset_base_dir,
                      hop_length=hop_length,
                      sample_rate=sample_rate,
                      num_frames=None,
                      data_proc=data_proc,
                      profile=model.profile,
                      store_data=False,
                      reset_data=reset_data,
                      save_loc=cache_dir)

# Instantiate IDMT-SMT-Guitar for testing
idmt_test = IDMT_SMT_Guitar(base_dir=idmt_base_dir,
                            splits=['licks', 'pieces'],
                            hop_length=hop_length,
                            sample_rate=sample_rate,
                            num_frames=None,
                            data_proc=data_proc,
                            profile=model.profile,
                            store_data=False,
                            reset_data=reset_data,
                            save_loc=cache_dir)

# Instantiate EGDB (direct input only) for testing
egdb_test = EGDB(base_dir=egdb_base_dir,
                 splits=['DI'],
                 hop_length=hop_length,
                 sample_rate=sample_rate,
                 num_frames=None,
                 data_proc=data_proc,
                 profile=model.profile,
                 store_data=False,
                 reset_data=reset_data,
                 save_loc=cache_dir)

# Initialize the estimation pipeline (Tablature -> Stacked Multi Pitch -> Multi Pitch)
validation_estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                                       StackedMultiPitchCollapser(profile=model.profile)])

# Initialize the evaluation pipeline (Loss | Multi Pitch | Tablature)
validation_evaluator = ComboEvaluator([LossWrapper(),
                                       MultipitchEvaluator(),
                                       TablatureEvaluator(profile=model.profile),
                                       SoftmaxAccuracy()])

# Compute the average results on GuitarSet
gset_results = validate(model, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

print(f'Results on GuitarSet: {gset_results}')

# Need to intialize each time if you only want results on one dataset
validation_estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                                       StackedMultiPitchCollapser(profile=model.profile)])

validation_evaluator = ComboEvaluator([LossWrapper(),
                                       MultipitchEvaluator(),
                                       TablatureEvaluator(profile=model.profile),
                                       SoftmaxAccuracy()])

# Compute the average results on IDMT-SMT-Guitar
idmt_results = validate(model, idmt_test, evaluator=validation_evaluator, estimator=validation_estimator)

print(f'Results on IDMT-SMT-Guitar: {idmt_results}')

# Need to intialize each time if you only want results on one dataset
validation_estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                                       StackedMultiPitchCollapser(profile=model.profile)])

validation_evaluator = ComboEvaluator([LossWrapper(),
                                       MultipitchEvaluator(),
                                       TablatureEvaluator(profile=model.profile),
                                       SoftmaxAccuracy()])

# Compute the average results on EGDB
egdb_results = validate(model, egdb_test, evaluator=validation_evaluator, estimator=validation_estimator)

print(f'Results on EGDB: {egdb_results}')
