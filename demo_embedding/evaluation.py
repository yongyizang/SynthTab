# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.datasets import GuitarSet
from amt_tools.features import CQT

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

# Initialize the estimation pipeline (Tablature -> Stacked Multi Pitch -> Multi Pitch)
validation_estimator = ComboEstimator([TablatureWrapper(profile=model.profile),
                                       StackedMultiPitchCollapser(profile=model.profile)])

# Initialize the evaluation pipeline (Loss | Multi Pitch | Tablature)
validation_evaluator = ComboEvaluator([LossWrapper(),
                                       MultipitchEvaluator(),
                                       TablatureEvaluator(profile=model.profile),
                                       SoftmaxAccuracy()])

# Create a CQT feature extraction module
# spanning 8 octaves w/ 2 bins per semitone
data_proc = CQT(sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=192,
                bins_per_octave=24)

# Default location of GuitarSet
gset_base_dir = None

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

# Compute the average results on GuitarSet
results = validate(model, gset_test, evaluator=validation_evaluator, estimator=validation_estimator)

print(f'Results on GuitarSet: {results}')
